# Spatial Transcriptomics with Graph Neural Networks

## Introduction

This repository provides a comprehensive benchmarking framework for applying Graph Neural Networks (GNNs) to the 10x Visium spatial transcriptomics dataset. Our pipeline evaluates multiple GNN architectures under a unified preprocessing and evaluation strategy.

---

## Datasets

We use the **10x Visium spatial transcriptomics dataset**, available at:

📂 **Download link**: [https://figshare.com/articles/dataset/10x\_visium\_datasets/22548901](https://figshare.com/articles/dataset/10x_visium_datasets/22548901)

The dataset includes **12 tissue slices**, grouped into three sub-datasets:

* `151507`, `151508`, `151509`, `151510`
* `151669`, `151670`, `151671`, `151672`
* `151673`, `151674`, `151675`, `151676`

---

## Data Preprocessing Pipeline

To ensure consistency across all tissue slices, we apply the following preprocessing steps:

1. **High Variation Gene (HVG) Selection**:
   For each slice, we identify the top **4,096 highly variable genes (HVGs)**.

2. **Gene Set Intersection**:
   We compute the **intersection of HVG sets across all slices**, ensuring that each model processes slices with the same gene set.

3. **Leave-One-Slice-Out Cross-Validation**:
   For model evaluation, we use a **leave-one-slice-out** strategy:

   * Train on **three slices**
   * Test on the **remaining slice** (unseen during training)
   * Repeat this for each slice in a group

---

## Model Evaluation

### GNN Architectures Benchmarked

We evaluate **seven GNN architectures**:

1. **APPNP**
2. **ChebNet**
3. **GAT (Graph Attention Network)**
4. **GatedGNN**
5. **GCN (Graph Convolutional Network)**
6. **GIN (Graph Isomorphism Network)**
7. **GraphSAGE**

### Hyperparameter Optimization

For each model, we perform hyperparameter tuning using a dedicated search script: `main_hyperopt`.

**Search space includes:**

* Number of GNN layers
* Hidden channels
* Number of neighbors (for message passing)
* Dropout rate
* Learning rate
* Optimizer type

**Search strategy:**

* **10 training epochs per trial** (`num_epoch_search = 10`)
* **100 hyperparameter trials per model**

### Final Training and Evaluation

After selecting the best configuration:

* Retrain the model for **200 epochs** (`num_epochs_evals = 200`)
* Evaluate on the held-out slice

### Evaluation Tasks

1. **Region Classification**

   * Metric: **Accuracy**
   * Objective: Classify spatial transcriptomics spots into anatomical regions.

2. **Clustering Analysis**

   * Extract **embeddings from the final GNN layer** (before classification head)
   * Metrics:

     * **ARI** (Adjusted Rand Index)
     * **NMI** (Normalized Mutual Information)


