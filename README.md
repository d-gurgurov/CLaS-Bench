# CLaS-Bench

## **Cross-Lingual Alignment and Steering Benchmark**

A lightweight parallel-question benchmark for evaluating language-steering behavior in LLMs across 32 languages.

---

## Overview

Understanding and controlling the behavior of large language models (LLMs) is an increasingly important topic in multilingual NLP. Beyond prompting or fine-tuning, *language steering*, manipulating internal representations during inference, has emerged as a more efficient and interpretable technique for adapting models to a target language. Yet, no dedicated benchmarks or evaluation protocols exist to quantify the effectiveness of these techniques.

**CLaS-Bench** addresses this gap by providing:

- A standardized benchmark covering **32 languages**
- Systematic evaluation of multilingual steering methods
- Two-axis performance measurement: **language control** and **semantic relevance**
- A unified **harmonic-mean steering score**

---

## Supported Models

This repository supports two models benchmarked in the paper:

| Model | Identifier |
|-------|------------|
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Aya Expanse 8B | `CohereLabs/aya-expanse-8b` |

> **Note:** The model choice must be specified in every shell script. The scripts can be extended to support other models with some effort.

---

## Evaluated Steering Methods

- Residual-stream DiffMean interventions
- Probe-derived directions
- Language-specific neurons
- PCA/LDA vectors
- Sparse Autoencoders (SAEs)
- Prompting baselines

---

## Getting Started

### Prerequisites

Log in to Hugging Face before running any code:

```bash
HF_TOKEN=<your_token>
huggingface-cli login --token $HF_TOKEN
```

---

## Usage

The benchmark consists of two parts: **(I) Identification** of language steering components and **(II) Steering** evaluation. Below, we detail the steps to reproduce the DiffMean results. Other methods can be run analogously using the appropriate scripts.

### Part I: Identification (Optional)

> Most identified components are already computed and included in this repository.

```bash
cd CLaS-Bench/identification

# Step 1: Download CulturaX data for all 32 languages
./0-load_data.sh

# Step 2: Compute average activations and steering vectors
./2-diffmean.sh

# Step 3: Generate layer-wise inspection plots
python vis_diffmean.py
```

### Part II: Steering Evaluation

#### 1. Run Baseline (Optional)

Evaluates two prompting strategies:
- Requesting target language response with English instruction
- Requesting target language response with native instruction

```bash
./1-baseline.sh
```

#### 2. Run DiffMean Steering

Scripts sweep through layers with different intervention strengths (α):

```bash
./2-diffmean-3.sh  # α = 5.0
# Additional scripts: -1, -2, -4 for other α levels
```

#### 3. Evaluate Language Forcing

```bash
./X-evaluate-language-forcing-diffmean.sh
```

#### 4. Evaluate Semantic Relevance

```bash
./Y-evaluate-judge-diffmean-3.sh
# Additional scripts: -0, -1, -2 for other α levels
```

#### 5. Generate Visualization

Produces plots for:
1. Language forcing success rate
2. Judge score (semantic relevance)
3. Harmonic mean of both scores

```bash
./B-vis-ablate-diffmean.sh
```

#### 6. Get Final Steering Scores

Runs the best configuration for each method based on ablation results:

```bash
./Z-get-steering-score-1.sh
```

> To run only for DiffMean, comment out other methods in the script.

---

## Repository Structure

```
CLaS-Bench/
├── identification/              # Scripts for identifying steering components
│   ├── 0-load_data.sh
│   ├── 2-diffmean.sh
│   └── vis_diffmean.py
├── data/                        # Data for all supported languages 
│   └── lang.txt                 
├── 1-baseline.sh                # Prompting baselines
├── 2-diffmean-*.sh              # DiffMean steering scripts
├── X-evaluate-language-forcing-*.sh
├── Y-evaluate-judge-*.sh
├── B-vis-ablate-*.sh
├── Z-get-steering-score-*.sh
└── README.md
```
