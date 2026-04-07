# Training Approach Comparison: Your DST-Unified vs LUAS vs D0T

## Overview

This document compares how three dialogue state tracking (DST) projects approach model training:
- **Your Project** (DST-Unified): Unified research pipeline supporting T5 and Llama models
- **LUAS** (ParticleMedia): LLM-backed User-Agent Simulation for DST
- **D0T** (emorynlp): Diverse 0-Shot Tracking

---

## 1. Training Architecture

### Your Approach (DST-Unified)

**Models Supported:**
- T5 (seq2seq): `google/flan-t5-base`, `google/flan-t5-large`
- Llama (causal LM): `meta-llama/Llama-2-7b-chat-hf`, etc.

**Training Framework:**
- **Single-GPU focused**: Uses HuggingFace `Trainer` class
- **LoRA for Llama**: Keeps base model frozen, only trains adapter weights (~5% of parameters)
- **Full fine-tuning for T5**: All model weights are trainable
- **No distributed training**: Designed for single GPU (A100 or CPU for small models)

**Data Processing:**
```
JSONL Input → Prompt + Target Normalization → Tokenization → Training
```

---

### LUAS Approach (LLM-backed User-Agent Simulation)

**Models Supported:**
- Llama 2: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`
- Custom sizes (7B, 13B, 70B)

**Training Framework:**
- **Multi-GPU distributed**: Uses `torchrun` with FSDP (Fully Sharded Data Parallel)
- **Enterprise-scale**: Supports distributed training across multiple nodes
- **LoRA + Quantization**: 8-bit quantization + LoRA for efficiency
- **FSDP Configuration**: Mixed precision (pure bfloat16), activation checkpointing

**Scaling:**
```bash
# 4-GPU single machine training
torchrun --nproc_per_node 4 llama_finetuning.py
# Can scale to multi-node with --nnodes parameter
```

**Key Features:**
- Gradient accumulation: 4 steps
- Learning rate: 2e-5 (lower, more conservative)
- Batch size: 4 per GPU (16 effective with 4 GPUs + accumulation)
- Epochs: 1 (careful optimization per epoch)
- Enable FSDP with pure_bf16 precision

---

### D0T Approach (Diverse 0-Shot Tracking)

**Models Supported:**
- Llama 3.1: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- T5 (alternative)

**Training Framework:**
- Custom framework with `LlamaTracker` and `T5Tracker` classes
- **Generator-based training**: Training yields results at configurable epoch intervals (not full epochs)
- No distributed/multi-GPU setup (similar to your approach)
- **Zero-shot focus**: Explicitly drops training domains for evaluation

**Key Innovation: Domain-Aware Evaluation**
- **Cross-domain testing**: Test on domains held out during training (true zero-shot)
- **Negative slot targets**: During eval, tests ALL slots per domain (even unseen ones)
- **Domain dropping**: Removes specified domains from training to simulate zero-shot scenario

**Data Processing:**
```
Load Training Data → Drop test domains → Add negative slot targets (in-domain) 
→ Training → Evaluation on all domains
```

---

---

## 2. Data Pipeline Comparison

### Your Approach

**Input Format:**
```json
{
  "dataset": "multiwoz24|d0t|luas",
  "dialogue_context": "Turn 0: user: ...\nTurn 1: system: ...",
  "slot_name": "domain-slot",
  "slot_description": "Human description of slot",
  "target_value": "slot_value"
}
```

**Preprocessing:**
- Unified JSONL format for all datasets
- Normalization: `"", "none", "not mentioned"` → `"none"`
- Balancing strategy: **50% none / 50% non-none** examples
- Prompt template: `(dialogue_context + slot_name + slot_description) → slot_value`

**Dataset Sizes:**
- T5 min: 200 examples (debug mode)
- T5 balanced: 400-8000 examples
- Llama balanced: 8000 examples (default)

---

### LUAS Approach

**Input Format:**
- Custom JSON format from generated agent simulations
- Path: `generation/multiwoz/converters/woz.2.2.gen/train.act.json`

**Preprocessing:**
- **Two-stage training approach:**
  1. **Agent Simulation Stage**: Generates synthetic conversations using LLM-backed agents
  2. **Fine-tuning Stage**: Trains decision makers on simulated + real data

**Agent-Based Prompting:**
```
System Role: Local guide handling services (hotel, restaurant, train, taxi, police)
Task: Determine next action (search vs. chat) + extract service slots
Format: JSON output with {"current_service": ..., "slots": {...}}
```

**Data Characteristics:**
- ~1000 training examples (for agent_sft_act_dataset)
- ~100 validation examples
- Context length: 1024 tokens

---

## 3. Training Configuration Comparison

| Aspect | **Your DST-Unified** | **LUAS** | **D0T** |
|--------|---------------------|---------|---------|
| **Distributed Training** | ❌ Single GPU | ✅ FSDP Multi-GPU | ❌ Single GPU |
| **Quantization** | Optional QLoRA (Llama) | ✅ 8-bit + int8 training | ✅ nf4 (different) |
| **LoRA Support** | ✅ Yes (Llama) | ✅ Yes (configurable) | ✅ Yes (tiny: r=2) |
| **Precision** | fp16 or float32 | pure_bf16 (FSDP) | Not specified |
| **Batch Size** | 4-8 per GPU | 4 per GPU (16 eff.) | 16/GPU (256 effective) |
| **Learning Rate** | 2e-4 (Llama), 1e-4 (T5) | 2e-5 (conservative) | 1e-2 (high, AdaFactor) |
| **Optimizer** | AdamW (torch) | AdamW + custom | **AdaFactor** (unique!) |
| **Epochs** | Variable (3-5 for T5) | 1 (careful tuning) | 1 (yield every 0.001 epochs) |
| **LR Scheduler** | Cosine (Llama), Linear (T5) | StepLR + Warmup | Not specified |
| **Task Focus** | Pure slot value prediction | DST + action selection | **Zero-shot generalization** |
| **Evaluation** | JGA, slot accuracy | JGA, slot accuracy | JGA + slot update accuracy |
| **Zero-shot Setup** | ❌ No | ❌ No | ✅ Domain hold-out test |

### Training Parameters Example

**Your Llama Training:**
```python
--model meta-llama/Llama-2-7b-chat-hf
--steps 500
--batch_size 4
--grad_accum 4  # Effective batch = 16
--lr 2e-4
--lora_r 16
--lora_alpha 32
--warmup_steps 50
--max_length 512
```

**LUAS Training:**
```bash
--model_name meta-llama/Llama-2-7b-hf
--train_batch_size 4
--micro_batch_size 4
--num_epochs 1
--lr 2e-5
--evaluation_steps 200
--pure_bf16  # Important for FSDP
--enable_fsdp
```

**D0T Training (Llama 3.1 8B):**
```python
base = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
train_batch_size = 16
gradient_accumulation_steps = 16  # Effective batch = 256
learning_rate = 1e-2  # High! Uses AdaFactor optimizer
optimizer = 'adafactor'  # Not AdamW!
lora_r = 2              # Much smaller than yours (16)
lora_alpha = 4          # Smaller than yours (32)
lora_dropout = 0.0
quantize = 'nf4'        # Different from int8/4-bit
epochs = 1
max_sequence_length = 512
protected_input_length = 400  # Unique: protects prompt tokens
yield_every_x_epochs = 0.001  # Fine-grained evaluation intervals
test_domains = ['restaurant']  # Zero-shot: hide from training
```

---

## 4. Model Architecture & Task Formulation

### Your Approach

**For T5 (Seq2Seq):**
```
Encoder: Dialogue context + slot name + slot description
Decoder: Predict slot value (max 32 tokens)
Loss: Only on target tokens (padding masked with -100)
```

**For Llama (Causal LM):**
```
Template: "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{dialogue_context}\nSlot: {slot_name}\nDescription: {slot_description}\n[/INST] {target_value}"
Loss: Only on target value tokens (prompt masked with -100)
Left-pad: Ensures causal masking works correctly
```

---

### LUAS Approach

**Two-stage Decision Making:**

1. **Agent Simulation Stage:**
   - Role-playing agents (e.g., "hotel reservation staff", "local operator")
   - Generate dialogue moves → Supervision signal
   - Simulates diverse user behaviors

2. **Fine-tuning Stage:**
   - Task: Slot extraction + action selection
   - Prompt includes persona + dialogue history
   - Output: JSON with actions and slot values

**Key Difference:**
- Not just predicting slot values
- Also predicting **what action to take** (search vs. chat)
- Combines DST with policy/action selection

---

### D0T Approach (Zero-shot Domain Generalization)

**Llama 3.1 Instruction-Following:**
```
Template: [Instruction + dialogue context + slot description] → [slot value]
Protected Input: First 400 tokens are "protected" - only last 112 tokens trained
LoRA: Minimal adapter (r=2) on top of frozen base
```

**Key Innovation: Protected Input Tokens**
- `protected_input_length: 400` safeguards prompt understanding
- Only learns on the final output portion (selective fine-tuning)
- Reduces adapter size significantly (r=2 vs. typical r=8-16)

**Generator-based Training Loop:**
```python
# Training yields control every 0.001 epochs for fine-grained evaluation
for epoch_ppl in tracker.training(data, yield_every_x_epochs=0.001):
    # Evaluate after tiny epoch intervals
    # Allows assessment of cross-domain generalization in progress
    perplexity = epoch_ppl
```

**Zero-shot Evaluation Setup:**
```
1. Drop test domain(s) from training set (e.g., 'restaurant')
2. Add negative slot targets (all valid domain-slot pairs)
3. Train on in-domain data only
4. Evaluate on held-out domain (zero-shot test)
5. Metrics: JGA, slot accuracy, slot update accuracy
```

---

## 5. Resource Requirements

### Your Approach

**Llama 7B Training:**
- GPU Memory: 40GB (A100) with LoRA + 4-bit quantization
- or 80GB (A100) with LoRA only
- Single GPU, simple setup

**T5-Base Training:**
- GPU Memory: 16-24GB
- Can run on consumer GPUs (RTX 3090, A10)

---

### LUAS Approach

**Llama 7B (4 GPUs):**
- Total Memory: 160GB+ (4 × 40GB GPUs)
- With FSDP sharding: Each GPU handles 1/4 of model
- More stable gradient updates, better scaling

**Example Parallelization:**
```
GPU0: Model layers 1-10, Optimizer states 1-10
GPU1: Model layers 11-20, Optimizer states 11-20
GPU2: Model layers 21-30, Optimizer states 21-30
GPU3: Model layers 31-40, Optimizer states 31-40
```

---

### D0T Approach

**Llama 3.1 8B Training:**
- GPU Memory: 24-32GB (single A100/H100)
- Lightweight LoRA: r=2 (1/8 of typical config)
- Protected tokens: Only final ~112 tokens computed for loss
- Effective batch: 256 (16 batch × 16 accumulation)
- AdaFactor optimizer: Memory-efficient (no momentum buffer)
- Single GPU setup, but aggressive batch/accumulation size

**Why Smaller LoRA with Large Batch?**
- Protected input + tiny LoRA reduces gradient variance
- Large batch accumulation compensates for smaller model updates
- AdaFactor requires less memory than AdamW

---

## 6. Key Differences Summary

| Aspect | Your Approach | LUAS | D0T |
|--------|---------------|------|-----|
| **Scale** | Research-oriented (single GPU) | Enterprise-scale (multi-GPU) | Research-focused (single GPU) |
| **Optimization** | Balanced dataset (50/50 none/non-none) | Agent simulation for data augmentation | **Zero-shot domain hold-out** |
| **Task Focus** | Pure value prediction | Value prediction + action selection | **Cross-domain generalization** |
| **Training Efficiency** | LoRA + QLoRA for memory | FSDP + distributed for speed | **Larger batch, AdaFactor** |
| **Framework** | HuggingFace Trainer | Custom training loop with torchrun | **Custom tracker (generator-based)** |
| **Data Source** | Unified real datasets | LLM-simulated + real data | DSG5K (diverse domains) |
| **Reproducibility** | Simple, single command | Requires multi-GPU setup | Moderate (custom framework) |
| **LoRA Configuration** | r=16, alpha=32 | Configurable | **r=2, alpha=4 (minimal)** |
| **Optimizer** | AdamW | AdamW | **AdaFactor (gradient-free)** |

---

## 7. When to Use Each Approach

### Use Your DST-Unified When:
- ✅ Limited GPU resources (single A100 or RTX GPU)
- ✅ Need rapid prototyping and research
- ✅ Want unified comparison across datasets
- ✅ Evaluating different model sizes/architectures
- ✅ Simple to reproduce and debug
- ✅ Need balanced training data handling

### Use LUAS When:
- ✅ Have multi-GPU cluster available
- ✅ Need production-scale training
- ✅ Want to leverage LLM-generated synthetic data
- ✅ Need joint optimization (DST + action selection)
- ✅ Can afford enterprise-level infrastructure

### Use D0T When:
- ✅ Focus on **zero-shot/cross-domain generalization**
- ✅ Want to test on domains unseen during training
- ✅ Interested in lightweight adapter training (r=2)
- ✅ Prefer AdaFactor optimizer (gradient-free, memory-efficient)
- ✅ Working with diverse domains (DSG5K-style data)
- ✅ Single GPU setup is fine
- ✅ Need fine-grained evaluation (yield every 0.001 epochs)

---

## 8. Recommendations

### To Improve Your Training:

1. **Consider FSDP for Llama:**
   ```bash
   # Current: Single GPU with LoRA
   # Potential: FSDP + LoRA on multi-GPU cluster
   torchrun --nproc_per_node 4 train_llama.py --enable_fsdp
   ```

2. **Adopt agent-based data augmentation:**
   - Use GPT-4/Claude to generate diverse dialogue variations
   - Mix synthetic + real data (like LUAS does)

3. **Hybrid approach:**
   - Keep your unified evaluation framework
   - Optionally adopt LUAS's two-stage training
   - Test on all three datasets with both methods

4. **Efficient scaling:**
   - Your current LoRA approach is optimal for single GPU
   - For cluster training, FSDP + mixed precision is better
   - Consider adding LUAS-style agent simulation as optional stage

---

## 9. Code Comparison: Key Files

| Task | Your Repo | LUAS Repo |
|------|-----------|----------|
| Model Loading | `src/dst/models/llama_dst.py` | `training_scripts/llama_finetuning.py` |
| Data Loading | `src/dst/data/jsonl_dataset.py` | `training_scripts/ft_datasets/agent_sft_act_dataset.py` |
| Training Loop | `src/dst/runners/train_llama.py` | `training_scripts/llama_finetuning.py::main()` |
| Config | `configs/*.yaml` | `training_scripts/configs/*` |

---

## Summary

Your **DST-Unified** approach prioritizes **simplicity and reproducibility** with a unified pipeline, while **LUAS** prioritizes **scale and data augmentation** through agent simulation. Both are valid—yours is ideal for research, LUAS for production systems.
