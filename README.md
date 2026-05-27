# Multimodal Data Fusion: Embedding Models vs. Generative VLMs

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository provides a comprehensive evaluation framework for multimodal data fusion, focusing on how to effectively utilize foundation models. It explores the trade-offs between lightweight embedding-based models and large zero-shot generative Vision-Language Models (VLMs) under various computational and domain constraints.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Data](#data)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Key Findings](#key-findings)
- [License](#license)
- [Citation](#citation)
- [Contact & Support](#contact--support)

## Introduction

This repository explores two dominant paradigms for utilizing foundation models in multimodal AI:

1. **Lightweight Embedding Fusion:** Extracts embeddings using pretrained foundation models, leveraging representations that act as a common format to unify different data modalities (e.g., CLIP, SigLIP, BiomedCLIP, MedSigLIP). These representations are combined via shallow neural networks (Linear Probes, Early/Late Fusion), significantly reducing computational demands and facilitating data integration in complex domains.

2. **Generative Vision-Language Models:** Employs state-of-the-art multimodal LLMs for direct zero-shot inference (e.g., Gemma-3-27B, MedGemma-27B). While highly capable without additional training, their performance depends heavily on model scale and application domain, limiting their use in resource-constrained settings.

Through systematic evaluation across **Recipes5k** (food concept recognition), **Fakeddit** (misinformation detection), and **mBRSET** (diabetic retinopathy prediction using clinical metadata), this framework provides a practical evaluation for selecting multimodal fusion strategies according to domain specificity, modality alignment, and computational resource constraints.

## Setup

### Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.12.7
- Required Python packages (specified in `requirements.txt`)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/dsrestrepo/Embedding-Alignment.git
cd Embedding-Alignment
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate 
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key if you plan to use GPT as a foundation model:

Create a `.env` file in the root directory.

Add your OpenAI API key to the `.env` file:

```makefile
OPENAI_API_KEY=your_api_key_here
```
Make sure you have a valid OpenAI API key to access the language model.

For local Hugging Face models such as Gemma or MedGemma, make sure the model
weights are available locally or that your Hugging Face credentials are
configured in the environment where the jobs run.

## Data

This repository focuses on **three primary datasets** for comprehensive multimodal fusion evaluation:

### 1. Recipes5k Dataset

[Recipes5k](http://www.ub.edu/cvub/recipes5k/): A multi-label ingredient prediction task from images and recipe metadata. Contains 4,826 recipes with 101 food types, addressing both intra- and inter-class variability. This use case demonstrates efficient embedding fusion for high-throughput, structured classification where abundant labeled training data is available.

### 2. Fakeddit Dataset

[Fakeddit](https://fakeddit.netlify.app/): A large-scale multimodal misinformation detection benchmark with over 1 million samples spanning satire, misinformation, and fabricated news. Features text, images, and metadata. This adversarial dataset explicitly tests fusion models' robustness to cross-modal contradictions (mismatched image-text pairs) and reveals when embedding-based approaches outperform generative models through early fusion strategies.

### 3. mBRSET Dataset

[mBRSET](https://www.nature.com/articles/s41597-025-04627-3): A mobile-friendly version of the Brazilian Ophthalmological dataset, capturing 5,164+ retinal fundus images from handheld cameras in LMIC screening campaigns, paired with clinical metadata (age, sex, systemic conditions, treatment history). This medical use case exemplifies deployment constraints and demonstrates how domain-specific embedding backbones (MedSigLIP) and late fusion architectures achieve clinical-grade diagnostics on edge devices without massive server infrastructure.

## Usage

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys (if needed for generative model inference)
cp .env.example .env
# Edit .env with your OpenAI, Hugging Face, or other API credentials
```

### 2. Download Datasets

```bash
# Download and preprocess the three primary datasets
sbatch jobs/download_datasets.sh
```

The jobs read `configs/paper_experiments.yaml`, where datasets, paths, model
backbones, and enabled experiments can be adjusted without editing the SLURM
scripts.

The same configured workflow can be launched directly:

```bash
python scripts/download_datasets.py --config configs/paper_experiments.yaml
```

### 3. Extract Embeddings

Extract vision and text embeddings using pre-trained foundation models:

```bash
# Generate embeddings using CLIP, SigLIP, BiomedCLIP, MedSigLIP, etc.
sbatch jobs/job_vlm_embeddings.sh
```

Embeddings are stored as CSV files under the `embeddings_dir` configured in
`configs/paper_experiments.yaml` for downstream fusion training.

Direct script entrypoint:

```bash
python scripts/generate_embeddings_all.py --config configs/paper_experiments.yaml
```

### 4. Train Fusion Models

Evaluate embedding fusion architectures (Linear Probe, Early Fusion, Late Fusion) and compare against zero-shot generative models:

```bash
# Train and evaluate embedding fusion pipelines
sbatch jobs/job_framework_fusion.sh

# Evaluate zero-shot Gemma models
sbatch jobs/job_gemma_zero_shot.sh
```

Direct script entrypoints:

```bash
python scripts/train_fusion_framework.py --config configs/paper_experiments.yaml
python scripts/eval_gemma_zero_shot.py --config configs/paper_experiments.yaml
```

### 5. Visualize and Analyze Results

```bash
# Generate performance summary plots and tables
sbatch jobs/job_plot_results.sh
```

Results are saved to `outputs/framework_fusion/` and `outputs/gemma_zero_shot/`.

## Directory Structure

```
configs/                       # YAML experiment definitions
├── paper_experiments.yaml     # Main paper configuration: paths, datasets, models, methods
└── __init__.py

src/                          # Core framework code
├── embeddings.py             # Embedding utilities and model-specific embedding logic
├── classifiers.py            # Classifier/fusion model components
├── datasets.py               # Dataset loaders for Recipes5k, Fakeddit, mBRSET
├── data_utils.py             # Preprocessing and utilities
├── get_data.py               # Dataset download and preprocessing helpers
├── models.py                 # Generative VLM wrappers (Gemma, Qwen, LLaVA, OpenAI, Gemini, etc.)
├── prompts.py                # Zero-shot prompting templates for generative models
├── quantization.py           # BitsAndBytes quantization helpers
├── vlm_embeddings.py         # Embedding extraction CLI used by torchrun
├── vlm_models.py             # Additional VLM model definitions/utilities
├── vlm_utils.py              # VLM dataset utilities
├── framework/
│   └── fusion.py             # FusionTrainerFramework for linear, early MLP, late MLP fusion
└── ...

scripts/                       # Main experiment pipelines
├── download_datasets.py       # Download and preprocess datasets
├── extract_data_samples.py    # Build small sample subsets
├── generate_embeddings_all.py # Extract embeddings across datasets/models
├── train_fusion_framework.py  # Train embedding fusion classifiers
├── eval_gemma_zero_shot.py    # Evaluate zero-shot generative models
├── correct_fakeddit_metrics.py # Utility for correcting Fakeddit result parsing
└── plot_framework_results.py  # Generate result summaries and tables

jobs/                          # SLURM job submission scripts
├── check_env.sh               # Environment sanity checks
├── download_datasets.sh       # Download Recipes5k, Fakeddit, mBRSET
├── job_extract_samples.sh     # Generate sample data subsets
├── job_vlm_embeddings.sh      # Extract foundation model embeddings
├── job_framework_fusion.sh    # Train fusion architectures
├── job_gemma_zero_shot.sh     # Run zero-shot Gemma inference
├── job_plot_results.sh        # Visualize and summarize results
└── ...

data_samples/                  # Generated sample subsets for quick inspection/testing

outputs/                       # Result CSVs, plots, and SLURM logs
├── framework_fusion/          # Fusion model results
└── gemma_zero_shot/           # Generative model results

main.tex, appendix.tex         # Paper manuscript sources
requirements.txt               # Python dependency list
```

## Key Findings

- **Embedding Fusion vs. Generative Models:** Lightweight MLP-based fusion of foundation embeddings provides a practical alternative to direct multimodal generative inference, achieving competitive or superior performance while requiring lighter downstream models.
- **Parameter Scaling:** Model scale remains a practical limitation for routine deployment. Reducing Gemma from 27B to 4B parameters causes significant accuracy drops on canonical tasks and near-random performance on adversarial (Fakeddit) data.
- **Domain Adaptation:** Medical-specific backbones (MedSigLIP) on mBRSET achieve superior performance compared to general-domain embeddings. This highlights the modularity of embedding fusion: practitioners can swap backbones to fit domain constraints.
- **Architectural Insights:**
  - **Early Fusion:** Performed best in Fakeddit, where the model needs to learn image-text interactions directly, including cases where captions and images may conflict.
  - **Late Fusion:** Performed best in Recipes5k and mBRSET, where modalities provide complementary information, preserving modality-specific signals before final integration.


## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{restrepo2024dfdm,
  author = {Restrepo, D. and Wu, C. and V\'{a}squez-Venegas, C. and Nakayama, L.F. and Celi, L.A. and L\'{o}pez, D.M.},
  title = {DF-DM: A foundational process model for multimodal data fusion in the artificial intelligence era},
  journal = {Res Sq},
  year = {2024},
  month = {Apr},
  day = {23},
  note = {Preprint},
  doi = {10.21203/rs.3.rs-4277992/v1},
  pmid = {38746100},
  pmcid = {PMC11092829}
}
```

## Contact & Support

For questions, issues, or suggestions:

- **Email:** davidres@mit.edu
