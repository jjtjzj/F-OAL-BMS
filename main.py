import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import timm
import numpy as np
from collections import Counter
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Hyperparameters (from paper, adjusted for resources)
D = 512  # Projection dimension (reduced from 1000 for speed/memory)
gamma = 1.0  # Regularization
batch_size = 10
num_tasks = 10
num_classes = 100
imb_factor = 0.01  # Change this for different α (e.g., 1.0, 0.1, 0.05, 0.01)
use_balanced_sampling = True  # Set to False for baseline without BMS

# Mean and std for normalization
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

# Load datasets (without transform, apply manually)
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=None)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=None)

train_data = train_dataset.data  # (50000, 32, 32, 3) numpy uint8
train_targets = np.array(train_dataset.targets)  # (50000,)

# Apply long-tail imbalance if imb_factor < 1
if imb_factor < 1:
    img_max = len(train_targets) // num_classes  # 500
    class_indices = [[] for _ in range(num_classes)]
    for idx, lbl in enumerate(train_targets):
        class_indices[lbl].append(idx)
    selected_indices = []
    for c in range(num_classes):
        num_samples = int(img_max * (imb_factor ** (c / (num_classes - 1.0))))
        num_samples = max(1, num_samples)
        selected = np.random.choice(class_indices[c], num_samples, replace=False)
        selected_indices.extend(selected)
    train_data = train_data[selected_indices]
    train_targets = train_targets[selected_indices]
    print(f"Applied long-tail with α={imb_factor}, total samples: {len(train_targets)}")

# Compute class sample counts for head/medium/tail
class_samples = Counter(train_targets)

# Convert train data to tensor, transpose, float, /255
train_data_tensor = torch.from_numpy(train_data.transpose(0, 3, 1, 2)).float() / 255.0

# Test data
test_data = torch.from_numpy(test_dataset.data.transpose(0, 3, 1, 2)).float() / 255.0
test_targets = torch.tensor(test_dataset.targets)

# Load backbone: ViT-S/16, frozen, features dim=384
backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
backbone = backbone.to(device)
backbone.eval()  # Frozen

# Random projection P (384 -> D), frozen
feat_dim = 384  # ViT-S
P = torch.randn(feat_dim, D, device=device) / np.sqrt(feat_dim)  # Normalized
P.requires_grad_(False)

# Function to extract fused features (average CLS from cumulative blocks)
def extract_fused_features(model, x):
    features = []
    def hook(module, input, output):
        features.append(output[:, 0, :])  # CLS token
    handles = []
    for block in model.blocks:
        h = block.register_forward_hook(hook)
        handles.append(h)
    _ = model.forward_features(x)  # Run forward
    for h in handles:
        h.remove()
    fused = sum(features) / len(features)  # Average over layers
    return fused

# Initialize R (D x D) and W (D x 0)
R = torch.eye(D, device=device) / gamma  # Init as I / gamma, but gamma=1 -> I
W = torch.zeros(D, 0, device=device)

# Track seen classes and class-to-col map
seen_classes = []
class_to_col = {}

# Accuracy matrix for metrics (num_tasks x num_tasks)
acc_matrix = np.zeros((num_tasks, num_tasks))

# Precompute mean and std tensors
mean_tensor = torch.tensor(mean, device=device).view(3, 1, 1)
std_tensor = torch.tensor(std, device=device).view(3, 1, 1)

# For each task
for task_id in range(num_tasks):
    print(f"\nStarting Task {task_id + 1}/{num_tasks}")
    
    # New classes for this task
    new_classes = list(range(task_id * 10, (task_id + 1) * 10))
    seen_classes.extend(new_classes)
    for i, c in enumerate(new_classes):
        class_to_col[c] = len(seen_classes) - len(new_classes) + i
    
    # Select task data (only new classes)
    task_mask = np.isin(train_targets, new_classes)
    task_data_tensor = train_data_tensor[task_mask]
    task_labels = train_targets[task_mask]
    
    task_labels_tensor = torch.from_numpy(task_labels)
    
    # Dataset
    task_dataset = TensorDataset(task_data_tensor, task_labels_tensor)
    
    # Loader with/without balanced sampling

    # if use_balanced_sampling:
    #     weights = [1.0 / class_samples[l.item()] for l in task_labels_tensor]
    #     sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    #     task_loader = DataLoader(task_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    # else:
    #     task_loader = DataLoader(task_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if use_balanced_sampling and len(task_labels_tensor) > 0:
        # Calculate weights based on the labels IN THIS TASK ONLY
        task_class_counts = Counter(task_labels_tensor.cpu().numpy())
        weights = [1.0 / task_class_counts[l.item()] for l in task_labels_tensor]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        task_loader = DataLoader(task_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        task_loader = DataLoader(task_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
 
        
    
    # Expand W for new classes
    W = torch.cat([W, torch.zeros(D, len(new_classes), device=device)], dim=1)
    
    # First batch flag
    is_first_batch = True
    prev_R_for_task = R.clone()  # For first batch of new task
    
    # Online updates: One-pass through loader
    for batch_idx, (batch_x, batch_y) in enumerate(task_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Apply resize and normalize (batch_x is (bs, 3, 32, 32), float [0,1])
        batch_x_resized = F.resize(batch_x, size=[224, 224], interpolation=InterpolationMode.BILINEAR)
        batch_x_transformed = (batch_x_resized - mean_tensor) / std_tensor
        
        with torch.no_grad():
            fused_feats = extract_fused_features(backbone, batch_x_transformed)
            phi = torch.sigmoid(fused_feats @ P)  # (bs, D)
        
        bs = phi.shape[0]
        
        # Prepare Y (bs x current_num_classes), one-hot
        Y = torch.zeros(bs, len(seen_classes), device=device)
        for i in range(bs):
            col = class_to_col[batch_y[i].item()]
            Y[i, col] = 1.0
        
        # RLS update
        Xa = phi  # (bs, D)
        XaT = phi.t()  # (D, bs)
        
        if is_first_batch:
            # Theorem 3.1: Use prev_R_for_task (from end of last task)
            inner = torch.eye(bs, device=device) + Xa @ prev_R_for_task @ XaT  # (bs, bs)
            inv_inner = torch.inverse(inner)
            R = prev_R_for_task - prev_R_for_task @ XaT @ inv_inner @ Xa @ prev_R_for_task
            error = Y - Xa @ W
            W = W + R @ XaT @ error
            is_first_batch = False
        else:
            # Theorem 3.2: Use current R
            inner = torch.eye(bs, device=device) + Xa @ R @ XaT
            inv_inner = torch.inverse(inner)
            R = R - R @ XaT @ inv_inner @ Xa @ R
            error = Y - Xa @ W
            W = W + R @ XaT @ error
    
    # After task, evaluate on all seen test data
    seen_test_mask = np.isin(test_targets.cpu().numpy(), seen_classes)
    seen_test_data = test_data[seen_test_mask]
    seen_test_labels = test_targets[seen_test_mask]
    
    seen_test_dataset = TensorDataset(seen_test_data, seen_test_labels)
    test_loader = DataLoader(seen_test_dataset, batch_size=32, shuffle=False)
    
    preds = []
    true = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            # Apply resize and normalize
            batch_x_resized = F.resize(batch_x, size=[224, 224], interpolation=InterpolationMode.BILINEAR)
            batch_x_transformed = (batch_x_resized - mean_tensor) / std_tensor
            fused_feats = extract_fused_features(backbone, batch_x_transformed)
            phi = torch.sigmoid(fused_feats @ P)
            logit = phi @ W
            pred = torch.argmax(logit, dim=1)
            preds.extend(pred.cpu().tolist())
            true.extend(batch_y.cpu().tolist())
    
    overall_acc = np.mean(np.array(preds) == np.array(true))
    print(f"After Task {task_id + 1}: Overall Acc = {overall_acc:.4f}")
    
    # Update acc_matrix: Acc per past task
    for past_task in range(task_id + 1):
        past_classes = list(range(past_task * 10, (past_task + 1) * 10))
        past_mask = np.isin(seen_test_labels.cpu().numpy(), past_classes)
        if past_mask.sum() > 0:
            past_preds = np.array(preds)[past_mask]
            past_true = np.array(true)[past_mask]
            task_acc = np.mean(past_preds == past_true)
            acc_matrix[task_id, past_task] = task_acc

# Final metrics
T = num_tasks
A_avg = np.mean(acc_matrix[T-1, :T])
max_past_acc = np.max(acc_matrix[:T-1, :T-1], axis=0)  # Max acc for each past task
final_acc = acc_matrix[T-1, :T-1]
F = np.mean(max_past_acc - final_acc) if T > 1 else 0.0

print(f"\nFinal Metrics:")
print(f"Average Incremental Accuracy (A_avg): {A_avg:.4f}")
print(f"Forgetting Rate (F): {F:.4f}")

# Head/Medium/Tail Accuracy
sorted_classes = sorted(class_samples, key=class_samples.get, reverse=True)
num_c = len(sorted_classes)
head = sorted_classes[:int(0.2 * num_c)]
medium = sorted_classes[int(0.2 * num_c):int(0.8 * num_c)]
tail = sorted_classes[int(0.8 * num_c):]

# Compute acc per group on final test
head_mask = np.isin(seen_test_labels.cpu().numpy(), head)
medium_mask = np.isin(seen_test_labels.cpu().numpy(), medium)
tail_mask = np.isin(seen_test_labels.cpu().numpy(), tail)

head_acc = np.mean(np.array(preds)[head_mask] == np.array(true)[head_mask]) if head_mask.sum() > 0 else 0
medium_acc = np.mean(np.array(preds)[medium_mask] == np.array(true)[medium_mask]) if medium_mask.sum() > 0 else 0
tail_acc = np.mean(np.array(preds)[tail_mask] == np.array(true)[tail_mask]) if tail_mask.sum() > 0 else 0

print(f"Head Accuracy: {head_acc:.4f}")
print(f"Medium Accuracy: {medium_acc:.4f}")
print(f"Tail Accuracy: {tail_acc:.4f}")
