# TARGE: Token Approximation and Reduction for Generation Efficiency

TARGE is a multimodal architecture designed to reduce visual token count while preserving semantic and pixel-level fidelity for efficient generation and reasoning in vision-language models (VLMs).

---

## Overview

Modern VLMs struggle with long visual token sequences, leading to high compute cost and poor scalability.

TARGE addresses this by:
- Selecting the most informative tokens
- Condensing redundant information
- Aligning visual tokens for efficient LLM consumption

The system combines:
- Joint-hybrid image encoding (pixel + semantic)
- Token selection via attention-based scoring
- Learnable query-based condensation
- Efficient connector into an LLM backbone

---

## System Architecture

### 1. Input

- Image

---

### 2. Joint-Hybrid Encoder

Encodes images into two complementary representations:

#### Pixel Encoder
- Model: DINOv3 ViT (~0.3B params)
- Output: `N` pixel tokens

#### Semantic Encoder
- Model: SigLIP v2 Base (~0.4B params)
- Output: `N` semantic tokens

#### Combined Output
- Total: `2N` tokens
- Concatenated along sequence dimension
- Each token includes:
  - Modality indicator (pixel vs semantic)
  - RoPE positional encoding

---

### 3. Selector

Selects the most important tokens.

#### Input
- `2N` tokens

#### Method
- Start with dense attention
- Use attention scores as importance metric
- Select top-`K` tokens

#### Future Extension
- Replace dense attention with DeepSeek Sparse Attention (DSA)

#### Output
- `K` selected tokens
- `Q` tokens (for condensation)

---

### 4. Condenser

Compresses information using learnable queries.

#### Input
- `Q` tokens

#### Method
- Cross-attention:
  - Query: Learnable Query Vectors (LQV)
  - Key/Value: Q tokens

#### Output
- `LQ` learnable query vectors

---

### 5. Connector

Aligns tokens for LLM input.

#### Input
- `K` tokens
- `LQ` vectors

#### Method
- Shared token-wise MLP

#### Output
- Aligned visual tokens (`K + LQ`)

---

### 6. LLM Backbone

Consumes text and reduced visual tokens.

#### Input
- Text tokens
- Visual tokens (`K + LQ`)

#### Options

- Llama 3.2  
  - 3B or 8B  
  - Baseline model  

- Llama 4  
  - 17B MoE (61B total)  
  - Higher performance option  

---

## Training Strategy

### Stage 1: Core Training (Required)

Train:
- Selector
- Condenser
- Connector

Freeze:
- Encoders
- LLM

Goal:
- Learn token selection, compression, and alignment

---

### Stage 2: Sparse Attention (Optional)

- Replace selector with DSA
- Freeze all other components
- Train DSA to approximate full attention

---

### Stage 3: LLM Adaptation (Optional)

- Unfreeze LLM
- Apply LoRA fine-tuning

Goal:
- Adapt LLM to compressed token distribution

---

## Data

- MS-COCO  
  - Standard benchmark dataset  

- CC3M  
  - Used in training CLIP, BLIP, LLaVA  

---

## Hardware Requirements

- FP8 support recommended:
  - NVIDIA Hopper / Blackwell GPUs  
  - Newer TPUs  

Notes:
- Enables efficient large-scale training  
- May increase iteration cost  
- Colab may provide limited access  

---

## Framework

- Swift  
  - Used for efficient VLM loading and execution  

---

## Evaluation

### 1. Multimodal Understanding
- Measures semantic comprehension

### 2. Long-Sequence Performance
- Evaluates efficiency gains from token reduction

### 3. Image Reconstruction / Generation
- Tests pixel fidelity of hybrid encoding

### 4. Dense Scene Understanding
- Measures information loss from token selection

---

## Key Idea

TARGE separates visual processing into:
- **Selection (what matters)**
- **Compression (how to represent it)**
- **Alignment (how to use it)**

This enables scalable multimodal modeling without the quadratic cost of attention over large token sets.