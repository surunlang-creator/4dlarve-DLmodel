# 4dlarve-DLmodel
ACmix+Swin-Transformer
# ACmix-Swin-WGCNA: Integrated Deep Learning Framework for Transcriptomic Phenotype Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **English** | [中文](#chinese-readme)

## Overview

ACmix-Swin-WGCNA is an integrated deep learning pipeline that combines **Wasserstein GAN with Gradient Penalty (WGAN-GP)** for data augmentation, **ACmix-Swin hybrid architecture** for phenotype classification, and **Weighted Gene Co-expression Network Analysis (WGCNA)** for biological interpretation. This framework is specifically designed for small-sample transcriptomic studies where traditional deep learning approaches struggle due to the high-dimensionality, low-sample-size (HDLSS) problem.

### Key Features

-  **WGAN-GP Data Augmentation**: Generates high-fidelity synthetic transcriptomic samples with quality filtering
-  **ACmix-Swin Architecture**: Hybrid CNN-Transformer model combining local feature extraction and global dependency modeling
-  **WGCNA Integration**: Python implementation of weighted co-expression network analysis
-  **Comprehensive Visualization**: Automated generation of 15+ publication-quality figures (PDF)
-  **Hub Gene Selection**: Dual-scoring system integrating deep learning importance and network topology
-  **Multi-strategy Augmentation**: Combines WGAN-GP, SMOTE, Gaussian noise, and Mixup
-  **Interpretable Results**: Gene-level importance scores and phenotype-specific biomarkers

### Architecture Diagram

```
Input Data (Gene Expression Matrix)
    ↓
[Feature Selection] → Top N genes by variance/mutual information
    ↓
[WGAN-GP Augmentation] → Synthetic sample generation + Quality filtering
    ↓                      
[Multi-strategy Fusion] → WGAN-GP (30%) + SMOTE (15%) + Noise (35%) + Mixup (20%)
    ↓
[ACmix-Swin Classifier]
    ├── Feature Embedding → FC layers with LayerNorm & GELU
    ├── ACmix Hybrid Layer
    │   ├── Swin Window Attention (Global dependencies)
    │   └── Depthwise Separable Conv (Local features)
    │   └── Dynamic weighted fusion (α × Attention + β × Conv)
    └── Classification Head → Adaptive pooling + Dropout + FC
    ↓
[WGCNA Network Analysis] → Module detection + Trait correlation
    ↓
[Hub Gene Selection] → Combined scoring (DL + WGCNA)
    ↓
Output: Predictions + Importance scores + Network files + Visualizations
```

---

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training acceleration)

### Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn scipy matplotlib
pip install openpyxl  # For Excel file export

# Optional but recommended
pip install tensorboard  # For training monitoring
```

### Clone Repository

```bash
git clone https://github.com/yourusername/acmix-swin-wgcna.git
cd acmix-swin-wgcna
```

---

## Quick Start

### Basic Usage

```bash
python acmix_swin_wgcna2.py \
    --expr expression_matrix.csv \
    --samples sample_groups.txt \
    --output ./results
```

### Input File Formats

#### 1. Expression Matrix (`expression_matrix.csv`)
```csv
Gene_ID,Sample1,Sample2,Sample3,...
Gene1,5.23,4.87,6.12,...
Gene2,3.45,3.78,2.91,...
Gene3,7.89,8.23,7.45,...
...
```
- **Rows**: Genes (with gene IDs in first column)
- **Columns**: Samples (with sample names in header)
- **Values**: Gene expression levels (e.g., log2(TPM+1), log2(FPKM+1), normalized counts)

#### 2. Sample Groups (`sample_groups.txt`)
```
Sample1	Drone
Sample2	Queen
Sample3	Worker
Sample4	Drone
...
```
- **Format**: Tab-separated (Sample_ID\tPhenotype)
- **Phenotype**: Must match exactly across samples (e.g., "Drone", "Queen", "Worker")

---

## Advanced Usage

### Custom Model Parameters

```bash
python acmix_swin_wgcna2.py \
    --expr exp.csv \
    --samples samples.txt \
    --output ./results \
    --n_features 100 \              # Number of features to select
    --samples_per_class 50 \        # Target samples per class after augmentation
    --embed_dim 64 \                # Embedding dimension (32, 64, 128)
    --num_heads 4 \                 # Number of attention heads
    --window_size 7 \               # Swin window size
    --dropout 0.3 \                 # Dropout rate
    --lr 1e-4 \                     # Learning rate
    --epochs 300 \                  # Max training epochs
    --patience 50 \                 # Early stopping patience
    --use_mixup \                   # Enable Mixup augmentation
    --mixup_alpha 0.2 \             # Mixup alpha parameter
    --label_smoothing 0.05          # Label smoothing factor
```

### Custom Hub Gene Selection

```bash
# Option 1: Same number of hub genes for all phenotypes
python acmix_swin_wgcna2.py \
    --expr exp.csv \
    --samples samples.txt \
    --n_overall_hub 30 \            # Overall hub genes
    --n_phenotype_hub 15            # Hub genes per phenotype

# Option 2: Different numbers for each phenotype
python acmix_swin_wgcna2.py \
    --expr exp.csv \
    --samples samples.txt \
    --n_overall_hub 30 \
    --n_phenotype_hub "Drone:20,Queen:15,Worker:18"
```

---

## Parameter Reference

### Data Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--expr` | str | **Required** | Path to expression matrix CSV file |
| `--samples` | str | None | Path to sample grouping file (tab-separated) |
| `--output` | str | `./output_wgcna` | Output directory path |
| `--n_features` | int | 100 | Number of features to select |

### Augmentation Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--samples_per_class` | int | 50 | Target samples per class after augmentation |
| `--gan_epochs` | int | 600 | WGAN-GP training epochs |

### Model Architecture Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--embed_dim` | int | 64 | Embedding dimension (recommended: 32, 64, 128) |
| `--num_heads` | int | 4 | Number of attention heads in Swin layer |
| `--window_size` | int | 7 | Window size for Swin attention |
| `--dropout` | float | 0.3 | Dropout rate (0.1-0.6) |

### Training Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | 300 | Maximum training epochs |
| `--batch_size` | int | 16 | Batch size for training |
| `--lr` | float | 1e-4 | Learning rate |
| `--weight_decay` | float | 1e-3 | Weight decay (L2 regularization) |
| `--patience` | int | 50 | Early stopping patience |
| `--use_mixup` | flag | False | Enable Mixup augmentation during training |
| `--mixup_alpha` | float | 0.2 | Mixup alpha parameter |
| `--label_smoothing` | float | 0.05 | Label smoothing factor |

### Hub Gene Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n_overall_hub` | int | 20 | Number of overall hub genes |
| `--n_phenotype_hub` | str/int | 10 | Hub genes per phenotype (e.g., "10" or "Drone:15,Queen:10,Worker:12") |

---

## Output Files

The pipeline generates a comprehensive set of output files organized as follows:

```
./output_wgcna/
├── Input/                              # R visualization input files
│   ├── node.xlsx                       # Network nodes
│   ├── edge.xlsx                       # Network edges
│   ├── layout.xlsx                     # Network layout
│   ├── module_correlation_matrix.csv   # Module-module correlation
│   ├── module_pvalue_matrix.csv        # Module-module p-values
│   ├── module_trait_correlation.csv    # Module-trait correlation
│   ├── module_trait_pvalue.csv         # Module-trait p-values
│   ├── groups.csv                      # Sample grouping
│   ├── metabolite_types.xlsx           # Gene types
│   └── WGCNA_results.pkl               # Complete WGCNA results
│
├── Visualizations/                     # Publication-quality PDFs
│   ├── training_curves.pdf             # Loss and accuracy curves
│   ├── wgan_training.pdf               # WGAN-GP training dynamics
│   ├── confusion_matrix.pdf            # Classification confusion matrix
│   ├── acmix_weights.pdf               # ACmix fusion weights
│   ├── augmentation_summary.pdf        # Data augmentation statistics
│   ├── data_original.pdf               # Original data PCA
│   ├── data_augmented.pdf              # Augmented data PCA
│   ├── data_comparison.pdf             # Original vs. augmented comparison
│   ├── distance_analysis.pdf           # Intra/inter-class distances
│   ├── gene_importance.pdf             # Top gene importance scores
│   ├── gene_importance_by_class.pdf    # Class-specific importance
│   ├── gene_heatmap.pdf                # Gene contribution heatmap
│   ├── wgcna_module_trait.pdf          # Module-trait heatmap
│   └── wgcna_module_correlation.pdf    # Module correlation network
│
├── Data/                               # CSV data files
│   ├── training_history.csv            # Training metrics per epoch
│   ├── gene_importance.csv             # Overall gene importance
│   ├── gene_importance_by_class.csv    # Phenotype-specific importance
│   ├── sample_gene_contribution.csv    # Sample-wise gene contribution
│   ├── gene_scores_combined.csv        # Combined DL + WGCNA scores
│   ├── predictions.csv                 # Model predictions and probabilities
│   └── selected_features.csv           # Selected feature list
│
└── model.pth                           # Trained model weights
```

### Key Output Files Description

#### Network Files (for Cytoscape visualization)
- **node.xlsx**: Node attributes (gene IDs, types, annotations)
- **edge.xlsx**: Edge list with weights (gene-phenotype associations)
- **layout.xlsx**: Pre-computed layout coordinates

#### Gene Importance Files
- **gene_importance.csv**: Overall gene importance scores (gradient × input method)
- **gene_importance_by_class.csv**: Phenotype-specific importance for each gene
- **gene_scores_combined.csv**: Integrated scores (DL importance + WGCNA topology)

#### WGCNA Results
- **module_trait_correlation.csv**: Correlation between gene modules and phenotypes
- **module_correlation_matrix.csv**: Inter-module correlation matrix
- **WGCNA_results.pkl**: Complete WGCNA object (module assignments, eigengenes, etc.)

---

## Methodology

### 1. WGAN-GP Data Augmentation

The pipeline uses Wasserstein GAN with Gradient Penalty to generate synthetic transcriptomic samples that preserve biological correlations while expanding the training set.

**Key Features:**
- **Generator**: 2-layer MLP with conditional embedding (noise_dim=64, hidden_dim=128)
- **Critic**: 3-layer spectral-normalized network with label conditioning
- **Gradient Penalty**: λ_gp = 10 to enforce Lipschitz constraint
- **Quality Filtering**: Removes synthetic samples with excessive distance from real data

**Training Parameters:**
- Optimizer: Adam (lr=1e-4, β=(0.0, 0.9))
- Critic iterations per generator update: 5
- Training epochs: 600

### 2. ACmix-Swin Hybrid Architecture

The classification model integrates convolutional operations and self-attention mechanisms through a novel fusion strategy.

**Architecture Components:**

**(a) Feature Embedding**
```
Input → FC(dim → 2×embed_dim) → LayerNorm → GELU → Dropout → FC(2×embed_dim → embed_dim×window_size)
```

**(b) ACmix Fusion Layer**

Two parallel branches with dynamic weighted fusion:

- **Attention Branch**: Swin Window Attention with relative position bias
  ```
  Attention(Q,K,V) = softmax(QK^T/√d_k + B)V
  ```
  
- **Convolution Branch**: Depthwise Separable Convolution
  ```
  DepthConv(kernel=3) → PointConv(kernel=1) → BatchNorm → GELU
  ```

- **Fusion**: `Output = α × Attention_out + β × Conv_out`
  - α and β are learnable parameters (initialized at 0.5)
  - Automatically balances global and local feature extraction

**(c) Classification Head**
```
LayerNorm → AdaptiveAvgPool1d → Dropout(0.3) → FC(embed_dim → num_classes)
```

### 3. WGCNA Network Analysis

Python implementation of weighted gene co-expression network analysis following the WGCNA protocol.

**Pipeline:**
1. **Soft-thresholding**: Select power β to approximate scale-free topology (R² > 0.85)
2. **Adjacency Matrix**: `a_ij = |cor(x_i, x_j)|^β`
3. **TOM Similarity**: Topological Overlap Matrix for robust similarity
4. **Module Detection**: Hierarchical clustering with dynamic tree cut
5. **Module Eigengenes**: First principal component of module expression
6. **Trait Association**: Correlation between module eigengenes and phenotypes

### 4. Hub Gene Selection

Integrates deep learning importance and network topology through dual scoring:

```
DL_score = Normalized(|∂Loss/∂x_i × x_i|)
WGCNA_score = Normalized(max(GS_i) × max(MM_i))
Combined_score = w_DL × DL_score + (1 - w_DL) × WGCNA_score
```

Where:
- **GS** (Gene Significance): Correlation with phenotype
- **MM** (Module Membership): Correlation with module eigengene
- **w_DL**: Weight for deep learning importance (default: 1.0)

**Hub Gene Categories:**
- **Overall Hub**: Top N genes by combined score across all phenotypes
- **Phenotype-specific Hub**: Top M genes for each phenotype based on class-specific DL importance

---

## Training Strategy

### Optimizer and Learning Rate
- **Optimizer**: AdamW (weight_decay=1e-3)
- **Base Learning Rate**: 1e-4
- **Warmup**: Linear warmup for first 10 epochs
- **Scheduler**: Cosine annealing after warmup

### Regularization Techniques
1. **Dropout**: 0.3 in embedding and classification layers
2. **Label Smoothing**: 0.05 (softens one-hot labels)
3. **Mixup Augmentation**: Optional (α=0.2)
4. **Gradient Clipping**: max_norm=1.0
5. **Early Stopping**: Patience=50 based on validation loss

### Data Split
- Training : Test = 80% : 20%
- Stratified sampling to maintain class balance

---

## Example Workflow

### 1. Prepare Your Data

```python
import pandas as pd

# Load expression data
expr_df = pd.read_csv('your_expression_data.csv', index_col=0)
# Rows = Genes, Columns = Samples

# Create sample groups file
samples = expr_df.columns.tolist()
groups = ['Drone', 'Queen', 'Worker', ...]  # Your phenotype labels

sample_groups = pd.DataFrame({'Sample': samples, 'Group': groups})
sample_groups.to_csv('sample_groups.txt', sep='\t', index=False, header=False)
```

### 2. Run the Pipeline

```bash
python acmix_swin_wgcna2.py \
    --expr your_expression_data.csv \
    --samples sample_groups.txt \
    --output ./my_results \
    --n_features 150 \
    --samples_per_class 60 \
    --embed_dim 64 \
    --epochs 300 \
    --use_mixup
```

### 3. Visualize Results

The pipeline automatically generates all visualizations in PDF format. For interactive network visualization:

**Option 1: Use Cytoscape**
```r
# In Cytoscape:
# 1. Import network from ./my_results/Input/edge.xlsx
# 2. Import node attributes from ./my_results/Input/node.xlsx
# 3. Apply layout from ./my_results/Input/layout.xlsx
```

**Option 2: Use R visualization script** (if available)
```r
# Place Input/ folder in R script directory
source('visualization_script.R')
```

### 4. Interpret Results

**Key files to examine:**

1. **Model Performance**: `confusion_matrix.pdf`, `training_curves.pdf`
2. **Gene Importance**: `gene_importance.csv`, `gene_importance_by_class.pdf`
3. **Hub Genes**: `gene_scores_combined.csv` (sorted by combined_score)
4. **Module Analysis**: `wgcna_module_trait.pdf`, `module_trait_correlation.csv`
5. **Predictions**: `predictions.csv` (with class probabilities)

---

## Troubleshooting

### Common Issues

**1. Out of Memory Error**
```bash
# Reduce batch size and embedding dimension
--batch_size 8 --embed_dim 32
```

**2. Model Overfitting**
```bash
# Increase dropout and weight decay
--dropout 0.5 --weight_decay 1e-2 --use_mixup
```

**3. Poor Convergence**
```bash
# Try different learning rate and longer warmup
--lr 5e-5 --warmup_epochs 20
```

**4. WGCNA Module Detection Fails**
```bash
# Adjust WGCNA parameters in code:
# min_module_size=10 (increase if too many small modules)
# merge_cut_height=0.25 (decrease for more modules)
```

**5. ImportError: openpyxl**
```bash
pip install openpyxl
# Or the code will automatically fall back to CSV format
```

---

## Performance Benchmarks

### Test Environment
- **Hardware**: NVIDIA RTX 3090 (24GB), Intel i9-12900K
- **Dataset**: 199 samples, 20,000 genes, 3 phenotypes
- **Configuration**: embed_dim=64, window_size=7, 100 selected features

### Runtime
| Stage | Time |
|-------|------|
| Feature Selection | ~5 seconds |
| WGAN-GP Training (600 epochs) | ~3 minutes |
| Data Augmentation | ~10 seconds |
| Model Training (300 epochs) | ~8 minutes |
| WGCNA Analysis | ~2 minutes |
| Visualization | ~30 seconds |
| **Total** | **~14 minutes** |

### Model Performance (Example)
- **Test Accuracy**: 87.6%
- **F1-Score (weighted)**: 0.86
- **Parameters**: ~45,000
- **Inference Time**: <1ms per sample

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{acmix_swin_wgcna,
  title={ACmix-Swin-WGCNA: Integrated Deep Learning Framework for Transcriptomic Phenotype Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/acmix-swin-wgcna}
}
```

**Key References:**

1. **WGAN-GP**:
   - Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein generative adversarial networks. In ICML.
   - Gulrajani, I., et al. (2017). Improved training of Wasserstein GANs. In NeurIPS.

2. **ACmix**:
   - Pan, X., et al. (2022). On the integration of self-attention and convolution. In CVPR.

3. **Swin Transformer**:
   - Liu, Z., et al. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV.

4. **WGCNA**:
   - Langfelder, P., & Horvath, S. (2008). WGCNA: an R package for weighted correlation network analysis. BMC bioinformatics.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
git clone https://github.com/yourusername/acmix-swin-wgcna.git
cd acmix-swin-wgcna

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- WGCNA methodology inspired by Peter Langfelder and Steve Horvath
- ACmix architecture based on the work by Pan et al. (2022)
- Special thanks to all contributors and users

---

## Contact

For questions, issues, or collaborations:
- **GitHub Issues**: [https://github.com/yourusername/acmix-swin-wgcna/issues](https://github.com/yourusername/acmix-swin-wgcna/issues)
- **Email**: your.email@institution.edu

---

<a name="chinese-readme"></a>

# 中文说明文档

## 概述

ACmix-Swin-WGCNA 是一个整合深度学习的转录组表型分类框架，专为小样本高维生物数据设计。该工具结合了 **WGAN-GP 数据增强**、**ACmix-Swin 混合架构**和**加权基因共表达网络分析(WGCNA)**，为转录组研究提供从数据预处理到生物学解释的完整流程。

### 核心特性

- 🧬 **WGAN-GP 数据增强**：生成高保真合成样本，缓解小样本问题
- 🤖 **ACmix-Swin 混合架构**：融合CNN局部特征提取和Transformer全局依赖建模
- 🔗 **WGCNA 网络分析**：Python 实现的加权共表达网络分析
- 📊 **全面可视化**：自动生成15+张高质量图表(PDF格式)
- 🎯 **Hub 基因筛选**：整合深度学习重要性和网络拓扑的双重评分
- 📈 **多策略增强**：结合 WGAN-GP、SMOTE、高斯噪声和 Mixup
- 🔬 **可解释结果**：基因水平重要性评分和表型特异性标志物

## 快速开始

### 安装依赖

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib openpyxl
```

### 基础用法

```bash
python acmix_swin_wgcna2.py \
    --expr 表达矩阵.csv \
    --samples 样本分组.txt \
    --output ./结果输出
```

### 输入文件格式

#### 表达矩阵 (CSV格式)
```csv
基因ID,样本1,样本2,样本3,...
基因1,5.23,4.87,6.12,...
基因2,3.45,3.78,2.91,...
```

#### 样本分组 (制表符分隔)
```
样本1	工蜂
样本2	蜂王
样本3	雄蜂
```

## 高级用法

### 自定义模型参数

```bash
python acmix_swin_wgcna2.py \
    --expr 表达矩阵.csv \
    --samples 样本分组.txt \
    --output ./结果 \
    --n_features 100 \              # 特征选择数量
    --samples_per_class 50 \        # 每类增强后样本数
    --embed_dim 64 \                # 嵌入维度
    --num_heads 4 \                 # 注意力头数
    --window_size 7 \               # Swin窗口大小
    --dropout 0.3 \                 # Dropout率
    --lr 1e-4 \                     # 学习率
    --epochs 300 \                  # 最大训练轮数
    --use_mixup                     # 启用Mixup增强
```

### 自定义Hub基因数量

```bash
# 方式1：统一数量
--n_overall_hub 30 --n_phenotype_hub 15

# 方式2：分别指定
--n_phenotype_hub "工蜂:20,蜂王:15,雄蜂:18"
```

## 参数说明

### 数据参数
- `--expr`: 表达矩阵文件路径（必需）
- `--samples`: 样本分组文件路径
- `--output`: 输出目录（默认：./output_wgcna）
- `--n_features`: 特征选择数量（默认：100）

### 增强参数
- `--samples_per_class`: 每类目标样本数（默认：50）
- `--gan_epochs`: WGAN-GP训练轮数（默认：600）

### 模型参数
- `--embed_dim`: 嵌入维度（默认：64，可选32/64/128）
- `--num_heads`: 注意力头数（默认：4）
- `--window_size`: Swin窗口大小（默认：7）
- `--dropout`: Dropout率（默认：0.3）

### 训练参数
- `--epochs`: 最大训练轮数（默认：300）
- `--batch_size`: 批大小（默认：16）
- `--lr`: 学习率（默认：1e-4）
- `--patience`: 早停耐心值（默认：50）
- `--use_mixup`: 启用Mixup增强
- `--label_smoothing`: 标签平滑系数（默认：0.05）

## 输出文件

```
./output_wgcna/
├── Input/                              # R可视化输入文件
│   ├── node.xlsx                       # 网络节点
│   ├── edge.xlsx                       # 网络边
│   ├── module_trait_correlation.csv    # 模块-表型相关性
│   └── WGCNA_results.pkl               # 完整WGCNA结果
│
├── Visualizations/                     # 可视化图表(PDF)
│   ├── training_curves.pdf             # 训练曲线
│   ├── confusion_matrix.pdf            # 混淆矩阵
│   ├── gene_importance.pdf             # 基因重要性
│   ├── wgcna_module_trait.pdf          # 模块-表型热图
│   └── ...（共15+张图表）
│
├── Data/                               # 数据文件(CSV)
│   ├── gene_importance.csv             # 基因重要性评分
│   ├── gene_scores_combined.csv        # 综合评分(DL+WGCNA)
│   ├── predictions.csv                 # 模型预测结果
│   └── ...
│
└── model.pth                           # 训练好的模型权重
```

## 方法学原理

### 1. WGAN-GP 数据增强
- 使用Wasserstein距离和梯度惩罚训练生成对抗网络
- 生成高保真合成转录组样本
- 质量过滤确保生成样本的生物学合理性

### 2. ACmix-Swin 混合架构
- **注意力分支**：Swin窗口注意力捕获全局依赖
- **卷积分支**：深度可分离卷积提取局部特征
- **动态融合**：可学习权重 α 和 β 自适应平衡两个分支

### 3. WGCNA 网络分析
- 软阈值选择
- 拓扑重叠矩阵(TOM)计算
- 模块检测和特征基因提取
- 模块-表型关联分析

### 4. Hub 基因筛选
整合深度学习和WGCNA的双重评分：
```
综合评分 = w_DL × DL重要性 + (1-w_DL) × WGCNA评分
```

## 性能基准

### 测试环境
- 硬件：NVIDIA RTX 3090, Intel i9-12900K
- 数据集：199样本，20000基因，3表型

### 运行时间
- 特征选择：~5秒
- WGAN-GP训练：~3分钟
- 模型训练：~8分钟
- WGCNA分析：~2分钟
- **总计：~14分钟**

### 模型性能（示例）
- 测试准确率：87.6%
- F1分数：0.86
- 参数量：~45,000

## 常见问题

### 内存不足
```bash
--batch_size 8 --embed_dim 32
```

### 模型过拟合
```bash
--dropout 0.5 --weight_decay 1e-2 --use_mixup
```

### 收敛困难
```bash
--lr 5e-5 --patience 80
```

## 引用

如果您在研究中使用了本工具，请引用：

还未发表
```

## 联系方式
- Email: surunlang@gmail.com

---

**注**：本工具仅供学术研究使用。
