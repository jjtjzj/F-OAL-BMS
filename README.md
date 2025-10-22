# F-OAL-BMS

# Does Class-Balanced Sampling Help Forward-Only OCIL? An Empirical Study with F-OAL under Long-Tail Streams

## Overview

This repository contains the code and experimental results for a research paper that investigates the surprising ineffectiveness of class-balanced sampling for the Forward-only Online Analytic Learning (F-OAL) framework. F-OAL is a novel approach to class incremental learning that boasts fast training and a low memory footprint by using a frozen feature extractor and a linear classifier updated with recursive least squares.

Our research extends the F-OAL paper (NeurIPS 2024) by addressing its identified limitation in handling long-tail class distributions. The initial hypothesis was that incorporating class-balanced sampling during mini-batch processing would improve performance on imbalanced datasets. However, our experiments demonstrate that this common technique provides no significant improvement and can even slightly degrade performance.

This README details the experimental design, presents the results, and provides an in-depth analysis of why class-balanced sampling fails in this context, highlighting the critical role of the frozen feature extractor as a performance bottleneck.

## Research Question and Hypothesis

**Research Gap:** The original F-OAL paper does not explicitly address the challenge of long-tail class distributions, where class imbalance can bias the learning process towards majority classes.

**Hypothesis:** Incorporating class-balanced sampling during mini-batch processing in the F-OAL framework will improve the average incremental accuracy (A_avg) and reduce the forgetting rate (F) on long-tail datasets without increasing the memory footprint or requiring back-propagation.

**Outcome:** The hypothesis was proven incorrect, leading to the central question of this research: *Why does class-balanced sampling, a standard technique for imbalanced learning, fail to improve the performance of F-OAL?*

## Experiment Design

### Dataset
*   **CIFAR-100:** Organized into 10 incremental tasks.

### Evaluation Metrics
*   **Average Incremental Accuracy (A_avg):** The primary metric to measure overall performance across all tasks.
*   **Forgetting Rate (F):** Quantifies the extent of catastrophic forgetting.
*   **Head/Medium/Tail Accuracy:** Class-wise accuracy is broken down into three groups based on the number of samples per class to directly assess the impact on classes with varying representation.

### Imbalance Ratios
To simulate different degrees of class imbalance, we introduced a long-tail distribution to the training data with the following imbalance ratios (α), ranging from mild to extreme:
*   α = 1.0 (Mild)
*   α = 0.1 (Moderate)
*   α = 0.05 (Strong)
*   α = 0.01 (Extreme)

### Modifications to the Original F-OAL Implementation
Due to limited computational resources (experiments were run on a Google Colab T4-GPU), the following modifications were made to the original paper's setup:
*   **Backbone:** Vision Transformer (ViT) was changed from ViT-B/16 to a smaller ViT-S/16.
*   **Projection Dimension (D):** Reduced from 1000 to 512.
*   **Number of Runs:** Conducted 1 run for each experiment instead of 3.

## Experimental Results

The experiments were conducted with and without a class-balanced sampler for each of the specified imbalance ratios.

### Without Class-Balanced Sampler
| Imbalance Ratio (α) | A_avg | F | Head Acc. | Medium Acc. | Tail Acc. |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1.0 | 0.7700 | 0.0870 | 0.7655 | 0.7740 | 0.7625 |
| 0.1 | 0.5893 | 0.0312 | 0.8705 | 0.6110 | 0.2430 |
| 0.05 | 0.5073 | 0.0234 | 0.8810 | 0.5273 | 0.0735 |
| 0.01 | 0.3964 | 0.0133 | 0.8925 | 0.3632 | 0.0000 |

### With Class-Balanced Sampler
| Imbalance Ratio (α) | A_avg | F | Head Acc. | Medium Acc. | Tail Acc. |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1.0 | 0.7666 | 0.0886 | 0.7655 | 0.7707 | 0.7555 |
| 0.1 | 0.5859 | 0.0261 | 0.8725 | 0.6065 | 0.2375 |
| 0.05 | 0.5073 | 0.0238 | 0.8775 | 0.5300 | 0.0690 |
| 0.01 | 0.4034 | 0.0108 | 0.8900 | 0.3738 | 0.0055 |

## Analysis of Results: The Bottleneck of Fixed Representations

The experimental results clearly show that class-balanced sampling does not lead to any meaningful improvement in the performance of F-OAL on long-tail datasets. This counter-intuitive finding points to a fundamental mismatch between the proposed solution and the architectural constraints of the F-OAL framework.

### Why Class-Balancing is Ineffective for F-OAL

1.  **The Frozen Feature Extractor:** The core of the issue lies in F-OAL's reliance on a **frozen, pre-trained feature extractor** (in this case, a ViT). This backbone is not fine-tuned on the incoming data stream. In traditional deep learning models, class-balanced sampling is effective because oversampling minority classes allows the network to learn more discriminative features for those classes through backpropagation. In F-OAL, no matter how many times an image from a tail class is shown, the feature vector produced by the frozen ViT remains identical.

2.  **The Nature of Recursive Least Squares (RLS):** F-OAL employs a Recursive Least Squares (RLS) solver to update its linear classifier. RLS is an analytical solver, not a gradient-based optimizer like SGD. It finds a precise mathematical solution to fit the given feature vectors to their corresponding labels. When you repeatedly feed the same ambiguous feature vectors from tail classes (due to the frozen backbone), the RLS solver doesn't necessarily learn a better representation. Instead, it can lead to overfitting on the specific examples of the tail classes it has seen, potentially at the expense of a more generalized solution.

### An Analogy

Imagine the frozen ViT backbone is a camera with a fixed, slightly blurry focus. The RLS classifier is an expert trying to identify objects from the photos taken by this camera.

*   **Traditional Training (with backpropagation):** This is like allowing the expert to adjust the camera's focus (the feature extractor) to get clearer pictures of rare objects.
*   **F-OAL Training:** This is akin to locking the camera's focus. Showing the expert the same blurry photo of a rare object multiple times (class-balanced sampling) doesn't help them identify it any better. They might become very good at recognizing that specific blurry photo but not at identifying the rare object in general.

## Conclusion

This research demonstrates a critical limitation of the F-OAL framework: its performance on long-tail distributions is bottlenecked by the quality of its frozen feature extractor. While class-balanced sampling is a powerful technique for models that learn representations, it is ineffective for a forward-only, analytic learning approach like F-OAL. The findings suggest that for such models to effectively handle imbalanced data, the focus should be on improving the quality and adaptability of the feature representations themselves, rather than on data-level balancing techniques that rely on representation learning.

## How to Run the Code

```bash
# Clone the repository
git clone https://github.com/jjtjzj/F-OAL-BMS
cd F-OAL-BMS

# Install dependencies
pip install -r requirements.txt

# Run the experiments
python main.py
