# Interpretability Framework for Vision-Language Models — STCG

Research project conducted at **POSTECH (GSAI)**, 2025.

## Problem

Vision-Language Models (VLMs) such as BLIP-2 achieve strong performance on multimodal tasks, but their internal reasoning remains opaque. Existing interpretability tools (Grad-CAM, token importance maps) provide static, correlation-based insights that fail to capture the causal and dynamic nature of multimodal decision-making.

## Approach

We introduce the **Structured Temporal Counterfactual Graph (STCG)**, an interpretability framework that represents VLM reasoning as a causally grounded graph combining three dimensions:

1. **Structural alignment** — links between textual tokens and visual regions via cross-attention
2. **Counterfactual sensitivity** — causal importance of specific connections measured through perturbation (center mask, color shift, patch shuffle, peripheral mask, saliency patch mask)
3. **Temporal evolution** — how attention patterns shift during response generation

The framework is applied to **BLIP-2 (Salesforce/blip2-flan-t5-xl)** on the **VQA-v2** dataset.

## Key Results

- Perturbations disrupting spatial organization (patch shuffling, peripheral masking) produce the largest internal weight variation ΔW
- Color-shift perturbations induce strong answer probability changes, consistent with chromatic attribution patterns for color-identification tasks
- A dissociation was identified between internal disruption and semantic decision: Gaussian noise produces comparable ΔW values without affecting the final answer
- Content-bearing tokens ("colour", "car") establish stronger and more localized connections to visual regions than function tokens ("what", "the")

## Limitations

Evaluation was conducted on a limited number of examples. Causal faithfulness (−0.31) and Jaccard stability (0.23) indicate that explanations derived from a single forward pass remain fragile — a known challenge in VLM interpretability that motivates further work.

## Repository Structure

```
├── STCG_dynamic_counterfactual.ipynb   # Main notebook (BLIP-2 + STCG pipeline)
└── README.md
```

## How to Use

Open the notebook in **Google Colab** (recommended — requires GPU for real model).

**Mock mode** (default, no GPU needed):
- Runs the full STCG pipeline with synthetic deterministic attention weights
- Validates the visualization and counterfactual pipeline without downloading multi-GB checkpoints
- Set `USE_MOCK_MODEL = True` in Cell 3

**Real mode** (requires ~12 GB disk + GPU):
- Set `STCG_FORCE_REAL=1` or `USE_MOCK_MODEL = False`
- Default model: `Salesforce/blip2-flan-t5-xl` — use `Salesforce/blip2-flan-t5-base` for a lighter alternative
- Provide your own image via `STCG_IMAGE_PATH` or use the built-in synthetic demo image

## Stack

Python · PyTorch · HuggingFace Transformers · BLIP-2 · Plotly · NetworkX · OpenCV · ipywidgets

## Report

*Preprint — link to be added.*

## Context

Student research project, POSTECH Graduate School of AI (GSAI), 2025.  
Authors: Zachari Arnaud, Noé Stefani.
