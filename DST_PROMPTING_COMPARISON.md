# DST Prompting Comparison: dst-unified vs D0T vs LUAS (Corrected)

## Your Implementation: dst-unified

### Prompt Structure

```python
def format_slot_prompt(dialogue_context: str, slot_name: str, slot_description: str) -> str:
    return (
        "Dialogue:\n"
        f"{dialogue_context}\n\n"
        "Slot:\n"
        f"{slot_name}\n\n"
        "Description:\n"
        f"{slot_description}\n\n"
        "Instructions:\n"
        "- Extract the exact slot value as mentioned or implied in the dialogue.\n"
        "- Use only the most recent value if it changes across turns.\n"
        "- If the user doesn't care about the value, answer 'dontcare'.\n"
        "- If the slot is not mentioned at all, answer 'none'.\n"
        "- Reply with the slot value only — no explanation.\n"
    )
```

### Characteristics
- **Format**: Structured 5-section layout (Dialogue | Slot | Description | Instructions | response)
- **Descriptions**: Always included (non-configurable)
- **Output**: Raw value ("none", "dontcare", or extracted text)
- **Philosophy**: "Explicit is better than implicit"
- **Compatibility**: Model-agnostic (T5, Llama, etc.)

---

## D0T Approach (Diverse 0-Shot Tracking)

### Prompt Template

From `dextrous/tracker.py`:

```
{dialogue_context}

Identify the information from the above dialogue:
{slot_name}{: description if included} {[candidates] or (e.g. val1, val2)?}?
```

### Concrete Example (with all options)

```
B: I'm looking for a cheap restaurant
A: We have several options...

Identify the information from the above dialogue:
restaurant-pricerange: The price range the user is looking for. (e.g. cheap, moderate, expensive)?
```

### Characteristics
- **Format**: Ultra-minimal template - dialogue + one-liner instruction + slot
- **Descriptions**: Highly configurable
  - Training: `train_percent_with_description` (default 1.0 = 100%)
  - Evaluation: Same as training
- **Candidate values**: Optional
  - Training: `train_percent_with_value_exs` (default 0.0 = 0%)
  - Eval: `eval_with_value_exs` (default True = 100%)
- **Modes**:
  - Description-only: slot name omitted, description only
  - Category mode: `[val1, val2, val3]?` (for categorical slots)
  - Example mode: `(e.g. val1, val2)?` (for open-ended slots)
- **Output**: Raw value
- **Efficiency**: Protected input (first 400 tokens frozen), minimal LoRA (r=2, α=4), nf4 quantization

---

## LUAS Approach (LLM-backed User-Agent Simulation)

### Actual Training Prompt (from `training_scripts/ft_datasets/agent_sft_act_dataset.py`)

```python
ANSWER_TYPE_PROMPT['act_selection_baseline_dst_2.4'] = (
    'You are a local guide online, primarily handling local services like:\n'
    'find the user\'s place (such as attraction, hotel, train, restaurant or hospital), and calling taxis, contacting the police, or other convenient services.\n'
    'Your service is efficient and of high quality, earning widespread praise from the local community.\n'
    'Given the conversion history, Your task is to help determine whether the next response can be directly replied to or not.\n'
    'Please output the current_service based on the user last utterence.\n'
    'Please noted that your responses are not used in the action selection, except the hotel name and restaurant name that you provided.\n'
    'And also please output all the services\' information that need pay attention to from the whole conversion.\n'
    'Here is the conversion history:\n{history}\n'
    'the user lastest utterence: \n{user_utterence}\n'
    'The output should be in JSON format like {{"slots": {{"service": {{"slot_key": "slot_val"}}}}}}\n'
    'Please give your decision:\n'
)
```

### Concrete Example

```
You are a local guide online, primarily handling local services like:
find the user's place (such as attraction, hotel, train, restaurant or hospital), 
and calling taxis, contacting the police, or other convenient services.
Your service is efficient and of high quality, earning widespread praise from the local community.
Given the conversion history, Your task is to help determine whether the next response can be directly replied to or not.
Please output the current_service based on the user last utterence.

Here is the conversion history:
[
  "user: I need a cheap restaurant in the south",
  "you: We have several options..."
]

the user lastest utterence: 
"Are there any with good food?"

The output should be in JSON format like {"slots": {"restaurant": {"area": "south", "pricerange": "cheap"}}}

Please give your decision:
```

### Characteristics
- **Format**: Task-driven narrative with persona context
- **Dialogue history**: Plain text list (not JSON structure)
- **Output**: JSON with service nesting: `{"slots": {"restaurant": {...}, "hotel": {...}}}`
- **Instruction style**: Implicit (embedded in persona narrative)
- **Descriptions**: N/A (implicit in persona context)
- **Training stages**: Two-stage approach
  1. Stage 1: Train on synthetic LLM-generated data
  2. Stage 2: Fine-tune on real MultiWOZ data
- **Efficiency**: Full fine-tuning (not parameter-optimized)
- **Multi-domain**: Handles multiple services simultaneously in JSON output
---

## Comparison Matrix

| Dimension | dst-unified | D0T | LUAS |
|-----------|-------------|-----|------|
| **Format Focus** | Slot extraction (explicit) | Slot extraction (sparse) | Multi-service state (JSON) |
| **Prompt Structure** | 5 labeled sections | Minimal template | Narrative task |
| **Description Handling** | Always 100% | Configurable (0-100%) | Fixed (implicit in persona) |
| **Candidate Values** | None | Optional by split | N/A (fixed output) |
| **Output Format** | Raw value | Raw value | JSON with services |
| **Instruction Style** | Explicit bullet list | Minimal prefix | Implicit via persona |
| **Token Count** | ~250-300 | ~150-200 | ~350+ |
| **Token Efficiency** | Unknown | High (protected input) | Low (full model) |
| **Training Stages** | 1 | 1 | 2 (synthetic → real) |
| **Zero-shot Design** | No | Yes (domain dropped) | No |
| **Model Type** | Generic | Generic | LLM-specific |
| **Multi-domain Handling** | Independent slots | Independent slots | Unified JSON output |
| **Parameter Optimization** | Unknown | Yes (LoRA) | No (full finetune) |

---

## Key Differences

### Prompting Philosophy

**dst-unified**: Clarity and explicitness
- Clear labels make it easy to understand what model sees
- Verbose instructions are unambiguous
- Good for debugging and analysis
- Verbose (more tokens)

**D0T**: Minimalism and efficiency
- Sparse format reduces token overhead
- Configurable components tune efficiency vs. effectiveness
- Protected input freezes dialogue (no gradient updates)
- Extremely parameter-efficient (r=2 LoRA)

**LUAS**: Task context and multi-domain
- Persona sets implicit task context
- Handles multiple services simultaneously
- Task-driven rather than slot-driven
- Two-stage training improves data efficiency

### Component Configurability

| Component | dst-unified | D0T | LUAS |
|-----------|-------------|-----|------|
| Description | Fixed (on) | Configurable | Fixed (implicit) |
| Examples | Fixed (off) | Configurable | Fixed (off) |
| Instruction | Fixed | Fixed prefix | Fixed persona |
| Output Schema | Fixed (raw) | Fixed (raw) | Fixed (JSON) |

### When to Use Each Approach

**Use dst-unified when:**
- Interpretability is critical (debugging DST models)
- Working with diverse model architectures (T5, Llama, etc.)
- Slot extraction is the main task (not multi-service)
- Token budget is not the primary concern

**Use D0T when:**
- Parameter efficiency is critical (mobile/edge deployment)
- Zero-shot cross-domain generalization matters
- You want to tune description/example ratios
- Fine-tuning resources are limited

**Use LUAS when:**
- Multi-service/multi-domain tracking is needed
- You have access to LLM for synthetic data generation
- Task-driven narrative prompting helps model performance
- You can afford full model fine-tuning

## 3. Key Differences

### Comparison Table

| Aspect | D0T | LUAS |
|--------|-----|------|
| **Primary Goal** | Zero-shot DST generalization | Data generation + fine-tuning |
| **Model** | Llama 3.1 8B (or T5) | LLaMA 2 |
| **Adapter Training** | LoRA (r=2, α=4) | LoRA (if used) + full fine-tuning |
| **Quantization** | nf4 | int8 or bfloat16 |
| **Prompt Type** | Slot-value extraction | JSON DST prediction |
| **Input Format** | Structured: context + slot + description + examples | Natural language: role + history + latest turn |
| **Output Format** | Slot value directly | JSON with slots structure |
| **Training Data** | Domain-diverse synthetic (DSG5K) | Generated via LLM + real data |
| **Training Stages** | Single stage per dataset | Two-stage (synthetic → real) |
| **Description Inclusion** | ~100% for training/eval | N/A (uses role description instead) |
| **Examples/Categories** | Optional, configurable | N/A (slot info in history context) |
| **Dialogue Context** | Reversed chronological view | JSON array of turns |

### Novel Techniques

#### D0T Innovations:
1. **Protected Input Length** - Freeze first 400 tokens, focus LoRA on last tokens
2. **AdaFactor Optimizer** - Memory-efficient gradient-free optimization
3. **nf4 Quantization** - Novel quantization approach vs standard int8
4. **Minimal LoRA Configuration** - r=2, α=4 for extreme parameter efficiency
5. **Dynamic Prompt Composition** - Hyperparameter-driven component inclusion for training vs evaluation
6. **Zero-shot Domain Generalization** - Explicitly drops test domains from training data

#### LUAS Innovations:
1. **Two-Stage Fine-tuning**:
   - Stage 1: LLM-generated synthetic data with diverse domains
   - Stage 2: Real data to ground synthetic knowledge
2. **LLM-backed Simulation** - Uses GPT-4 to generate realistic user/agent interactions
3. **JSON Schema Output** - Structured prediction instead of free-form values
4. **Multi-service Multi-slot Nesting** - Hierarchical slot structure per service
5. **Adaptive Domain Extension** - Can generate data for new domains on-the-fly
6. **Role-based Prompting** - System persona instead of slot descriptions

---

## 4. Prompting Strategy Summary

### D0T: Slot-Centric Extraction Prompting
- **Formulation**: `(dialogue_context + slot_name + description) → value`
- **Focus**: Individual slot value extraction
- **Instruction Style**: Task-oriented, direct
- **Scalability**: Per-slot independent prediction
---

## Recommendations for dst-unified

### Keep Your Current Approach If:
- Interpretability and transparency are top priorities
- You're working with diverse model architectures
- Debugging and understanding model behavior matters
- Your primary task is single-slot extraction per example

### Adopt D0T Elements If:
- Token efficiency becomes critical
- Parameter count is constrained
- You want to tune description/example ratios
- Zero-shot generalization is desired

### Adopt LUAS Elements If:
- Multi-service/multi-domain joint prediction is needed
- You want two-stage training (synthetic→real)
- Access to LLM for data generation is available
- Task-driven narrative helps your models

### Suggested Enhancements for dst-unified:
1. **Add description toggle** - `include_description: bool = True`
2. **Add candidate values option** - `candidate_values: list[str] | None`
3. **Stay with explicit format** - it's clearer than D0T's sparse approach
4. **Keep raw output** - simpler than LUAS's JSON (unless multi-service needed)
5. **Consider two-stage training** - could improve robustness without changing prompts

---

## Summary

| Approach | Best For | Output | Philosophy |
|----------|----------|--------|-----------|
| **dst-unified** | Interpretability, diversity | Explicit sections | Clear is better |
| **D0T** | Efficiency, zero-shot | Minimal template | Less is more |
| **LUAS** | Multi-service, LLM data | JSON structure | Context matters |

Your current explicit approach is **good for analysis and understanding**. D0T excels at **parameter efficiency and zero-shot**. LUAS excels at **structured multi-service output and synthetic data generation**. Choose based on your primary constraints and goals.

