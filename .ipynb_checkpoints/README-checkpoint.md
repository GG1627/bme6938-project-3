# Clinical NLP - PubMed RCT Sentence Classification

> NLP pipeline for classifying clinical trial abstract sentences into rhetorical roles using a bidirectional LSTM baseline and fine-tuned DistilBERT. Built on the PubMed RCT 20k dataset.

**GitHub:** [bme6938-project-3](https://github.com/GG1627/bme6938-project-3)
**Course:** Medical AI (BME6938) - University of Florida

---

## Clinical Context

Randomized controlled trials (RCTs) are the gold standard for clinical evidence, but the sheer volume of published literature makes manual review impractical. Automatically classifying sentences in RCT abstracts into their rhetorical roles - Background, Objective, Methods, Results, and Conclusions - enables faster literature synthesis, structured summarization, and downstream clinical decision support. This project trains and compares a recurrent baseline and a pretrained transformer model on this sentence classification task.

---

## Results Summary

| Model | Accuracy | Macro F1 | Weighted F1 | Params |
|-------|:---:|:---:|:---:|:---:|
| Bidirectional LSTM | 83% | 0.77 | 0.83 | 6.2M |
| DistilBERT | 85% | 0.79 | 0.85 | 67M |

### Per-Class F1

| Class | LSTM | DistilBERT |
|-------|:---:|:---:|
| background | 0.63 | 0.69 |
| objective | 0.64 | 0.63 |
| methods | 0.91 | 0.93 |
| results | 0.89 | 0.90 |
| conclusions | 0.78 | 0.81 |

---

## Repository Structure

```
bme6938-project-3/
├── figures/           # generated plots and visualizations
├── models/            # saved model checkpoints
├── eda.ipynb          # exploratory data analysis
├── train.ipynb        # LSTM + DistilBERT training pipeline
├── demo.ipynb         # load trained models, run inference, error analysis
├── requirements.txt
└── README.md
```

---

## Quick Start

**Requirements:** Python 3.10+, at least 1 GPU recommended

### HiPerGator (UF)

When launching a Jupyter session on [ood.rc.ufl.edu](https://ood.rc.ufl.edu), use these settings:

| Setting | Value |
|---|---|
| Cluster Partition | `default` |
| Number of CPUs | `4` |
| Memory (GB) | `16` |
| GRES | `gpu:1` |

After the session launches, select the **PyTorch-2.8.0** kernel from the Jupyter kernel selector in the top right of the notebook. Each notebook installs its own dependencies in the first cell - no manual installation needed.

Then run the notebooks in order:
1. `eda.ipynb`
2. `train.ipynb`
3. `demo.ipynb`

---

## Data

- **Source:** [PubMed RCT 20k](https://huggingface.co/datasets/armanc/pubmed-rct20k) via Hugging Face
- **Size:** 235,892 sentences (176,642 train / 29,672 val / 29,578 test)
- **Task:** 5-class sentence classification
- **Classes:** Background, Objective, Methods, Results, Conclusions
- **License:** Open access
- **Citation:** Dernoncourt & Lee, 2017

> The dataset is loaded automatically via the Hugging Face `datasets` library - no manual download needed.

```python
from datasets import load_dataset
ds = load_dataset('armanc/pubmed-rct20k')
```

---

## Methods Overview

- **Preprocessing:** Lowercased text, `@` tokens retained (anonymized numbers), class weights computed to handle imbalance
- **LSTM Baseline:** Custom vocabulary (30k tokens), bidirectional LSTM (2 layers, hidden dim 256), embedding dim 128, dropout 0.3
- **DistilBERT:** `distilbert-base-uncased` fine-tuned for sequence classification, max length 128, lr 2e-5
- **Training:** Adam optimizer, ReduceLROnPlateau scheduler, gradient clipping, best checkpoint saved by val loss
- **Evaluation:** Accuracy, per-class Precision/Recall/F1, Macro F1, Weighted F1, Confusion Matrix

---

## Environment

```
Python        3.10
PyTorch       2.8.0
transformers  4.57.3
datasets      4.3.0
scikit-learn  1.7.2
```

Full list of dependencies in `requirements.txt`.

---

## Authors & Contributions

| Name | Role |
|------|------|
| Gael Garcia | Project lead, LSTM + DistilBERT training pipeline, model architecture, GitHub setup |
| Jada | Data preprocessing, EDA notebook, class imbalance analysis, visualizations |
| Dylan | Literature review, evaluation metrics, error analysis, demo notebook |

---

## AI Disclosure

Claude was used as a coding assistant during development. All code was reviewed, tested, and understood by the team. AI assistance is documented per course guidelines.