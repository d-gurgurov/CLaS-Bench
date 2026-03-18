# 🧭 Language Steering

A Gradio-based interface for steering LLM responses between languages using activation engineering techniques.

## Overview

This app allows you to manipulate the internal activations of language models to steer their outputs from one language to another. It supports two steering methods:

- **Diffmean**: Modifies hidden states by adding the difference between language-specific steering vectors
- **Neurons**: Directly manipulates MLP gate activations for language-specific neurons

## Supported Models

| Model | HuggingFace ID |
|-------|----------------|
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Aya-Expanse-8B | `CohereForAI/aya-expanse-8b` |

## Supported Languages

English, Arabic, Tibetan, Danish, German, Spanish, French, Hindi, Italian, Japanese, Korean, Maltese, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Turkish, Chinese, Slovak, Greek, Kazakh, Swahili, Georgian, Ukrainian, Persian, Thai, Indonesian, Vietnamese, Czech, Romanian

## Steering Methods

### Diffmean Steering

Steers the model by adding the difference between target and source language vectors to the hidden states.

**Parameters:**
- **Steering Strength** (0-10): Controls how strongly the steering is applied
- **Start/End Layer** (0-31): Which layers to apply the steering to

### Neuron Steering

Directly manipulates language-specific neurons in the MLP gate projections using additive intervention.

**Parameters:**
- **Neuron Percentage K** (1-5%): Top K% of language-specific neurons to manipulate
- **Activation Strength** (0-10): How much to boost target language neurons
- **Deactivation Value** (-5 to 5): Value to set source language neurons to
- **Deactivate Source**: Toggle to also suppress source language neurons

## Installation

```bash
pip install gradio torch transformers huggingface_hub
```

## Usage

1. Set your HuggingFace token (required for gated models):
   ```bash
   export HF_TOKEN=your_token_here
   ```

2. Run the app:
   ```bash
   python app.py
   ```

3. Open the Gradio interface in your browser

4. Load a model, select source/target languages, choose a steering method, and generate!

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token for accessing gated models |
