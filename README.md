
<h1 align="center">Prototype-Conditioned Graph Neural Networks (PCGNN)</h1>
<p align="center">
  <b>Official Implementation of "Group Rationale: Prototype-Conditioned Graph Neural Networks"</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-1.12+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-2.3+-green.svg" alt="PyG">
</p>

---


## 📖 Overview

This repository contains the official PyTorch implementation of **"Group Rationale: Prototype-Conditioned Graph Neural Networks (PCGNN)"**.

PCGNN employs prototypes to discover entity-group behavioral mappings within an integrated framework unifying information sources, information propagation, and human-readable concept translation of model outputs. It is tailored to explain model decisions based on the associations between prototypes and graph entities.

### Key Features

- 🧠 **Prototype Learning**
- 📊 **Group Rationale**
- 🔍 **Interpretability**
- 🌐 **Visualization Companion**

---

## 🔗 Companion Visualization Tool

> 🌐 **Interactive Prototype Visualizer**: [Open Visualization Tool](https://github.com/goalsetting/PROTOTYPE_VISUALIZER)
>
> This companion web tool allows you to:
> - Visualize entity-prototype allocation networks
> - Inspect LLM-generated prototype concepts
> - Review statistical evidence profiles
> - Annotate and validate prototype quality with expert marks
>
> The visualizer runs entirely in your browser with **no data uploaded to any server**.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pcgnn.git
cd pcgnn

# Create conda environment (recommended)
conda create -n pcgnn python=3.9
conda activate pcgnn

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install PyTorch Geometric and dependencies
pip install torch-geometric==2.3.1
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

### 2. Test on Cora Dataset

The Cora dataset will be automatically downloaded via PyTorch Geometric on first run.

```bash
# Run with default config (Cora dataset)
python main_E2E.py --config config.yaml
```

Expected output:
```
[2024-01-15 10:30:45] Loading Cora dataset...
[2024-01-15 10:30:46] Dataset: Cora, Nodes: 2708, Edges: 10556, Classes: 7
[2024-01-15 10:30:46] Model: PCGNN, Prototypes: 10, Hidden: 128
[2024-01-15 10:30:47] Epoch 001/200 | ....
...
[2024-01-15 10:35:12] Epoch 200/200 | ....
```

---

## ⚙️ Configuration (config.yaml)

All hyperparameters and runtime settings are managed through `config.yaml`. Below is the full configuration template:

```yaml
# =============================================================================
# PCGNN Configuration File
# =============================================================================

# -----------------------------------------------------------------------------
# 2. Model Architecture
# -----------------------------------------------------------------------------
model:
  in_feats: 100
  hidden_dim: 256
  out_dim: 32
  num_layers: 3
prototype:
  num_prototypes: 7
  context_hidden: 256
  expl_strategy: 'entropy_topk'  # 或 'threshold', 'topk'
  expl_k: 2                          
ot:
  epsilon: 0.05
  max_iter: 5
  tol: 1e-6
graph:
  diffusion_alpha: 0
loss:
  w_centroid: 0.5
  w_contrast: 0.3
  w_suff: 0.1     
  w_nec: 0.1 
optim:
  lr: 0.001
  weight_decay: 0.0001
device: cuda
conditioned_gnn:
  num_layers: 3    
  hidden_dim: 32
  out_dim: 32         
  use_hypernet: true     
  hyper_hidden: 32       
  residual: False       
  pc_attn_mode: "simple"
```

### Common Config Modifications

**Change number of prototypes:**
```yaml
prototype:
    num_prototypes: 12
```

---

## 🖥️ Command Line Interface

### Basic Usage

```bash
# Run with default config
python main_E2E.py --config config.yaml

# Run with custom config file
python main_E2E.py --config configs/cora_gat.yaml

# Override specific parameters via CLI
python main_E2E.py --config config.yaml --prototype.num_prototypes 12

```

---

## 🌐 Using the Visualization Tool

After training, export the visualization data:

```bash
python main_E2E.py --config config.yaml --export_viz --viz_dir ./my_viz_data
```

Then open the visualizer:

```bash
# Open in browser
cd visualizer
open index.html

# Or use a local server
cd visualizer
python -m http.server 8000
# Visit http://localhost:8000
```

Upload the exported files (`concepts.json`, `evidence.json`, `mapping.csv`, `labels.csv`) into the visualizer to explore your trained prototypes interactively.

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{pcgnn2026,
  title={Group Rationale: Prototype-Conditioned Graph Neural Networks},
  author={Author Names},
  booktitle={Undefined},
  year={2026}
}
```

---

## 📜 License

[MIT License](LICENSE)

---

## 🙏 Acknowledgments

- [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/) for the graph learning framework
- [D3.js](https://d3js.org/) for the companion visualization tool

</details>

