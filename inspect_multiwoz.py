import json
from pathlib import Path
from collections import Counter

print("=" * 100)
print("MULTIWOZ DATASET INSPECTION")
print("=" * 100)

data_dir = Path("data_unified/multiwoz24")

# Check if files exist
print("\n1. CHECKING FILES")
print("-" * 100)
for split in ["train", "val", "test"]:
    file_path = data_dir / f"{split}.jsonl"
    if file_path.exists():
        size = file_path.stat().st_size / (1024*1024)
        print(f"✓ {split}.jsonl exists ({size:.1f} MB)")
    else:
        print(f"✗ {split}.jsonl NOT FOUND")

# Load all splits
print("\n2. LOADING DATA")
print("-" * 100)

train_examples = []
val_examples = []
test_examples = []

with open(data_dir / "train.jsonl") as f:
    train_examples = [json.loads(line) for line in f]

with open(data_dir / "val.jsonl") as f:
    val_examples = [json.loads(line) for line in f]

with open(data_dir / "test.jsonl") as f:
    test_examples = [json.loads(line) for line in f]

print(f"Train examples: {len(train_examples):,}")
print(f"Val examples: {len(val_examples):,}")
print(f"Test examples: {len(test_examples):,}")
print(f"Total examples: {len(train_examples) + len(val_examples) + len(test_examples):,}")

# Show first example
print("\n3. FIRST EXAMPLE STRUCTURE")
print("-" * 100)
ex = train_examples[0]
print(f"Example 1 fields:")
for key, value in ex.items():
    if key == "dialogue_context":
        print(f"  {key}: {str(value)[:100]}...")
    elif isinstance(value, str) and len(value) > 50:
        print(f"  {key}: {value[:50]}...")
    else:
        print(f"  {key}: {value}")

# Analyze structure
print("\n4. DATASET STATISTICS (TRAIN SPLIT)")
print("-" * 100)

# Unique values
dialogue_ids = len(set(ex.get("dialogue_id") for ex in train_examples))
slot_names = len(set(ex.get("slot_name") for ex in train_examples))
speakers = Counter(ex.get("speaker") for ex in train_examples)
slots = Counter(ex.get("slot_name") for ex in train_examples)
turn_ids = [ex.get("turn_id") for ex in train_examples if ex.get("turn_id") is not None]

print(f"Unique dialogue IDs: {dialogue_ids:,}")
print(f"Unique slot names: {slot_names:,}")
print(f"Total train examples: {len(train_examples):,}")
print(f"Turn ID range: {min(turn_ids) if turn_ids else 'N/A'} to {max(turn_ids) if turn_ids else 'N/A'}")

print(f"\nSpeaker distribution:")
for speaker, count in speakers.most_common():
    print(f"  {speaker}: {count:,} ({count/len(train_examples)*100:.1f}%)")

# Slots - Top 20
print(f"\nTop 20 slots by frequency:")
for slot, count in slots.most_common(20):
    print(f"  {slot}: {count:,}")

# None values
none_count = sum(1 for ex in train_examples if ex.get("target_value") == "none")
print(f"\n'none' values: {none_count:,} ({none_count/len(train_examples)*100:.2f}%)")

# Context length
context_lengths = [len(ex.get("dialogue_context", "")) for ex in train_examples]
print(f"\nContext length statistics:")
print(f"  Min: {min(context_lengths)} chars")
print(f"  Max: {max(context_lengths):,} chars")
print(f"  Mean: {sum(context_lengths)/len(context_lengths):.0f} chars")

# Value appearance
print("\n5. VALUE APPEARANCE IN DIALOGUE")
print("-" * 100)
non_none_examples = [ex for ex in train_examples if ex.get("target_value") != "none"]
exact_match = 0
case_insensitive = 0

for ex in non_none_examples:
    target = str(ex.get("target_value", ""))
    context = ex.get("dialogue_context", "")
    if target in context:
        exact_match += 1
    elif target.lower() in context.lower():
        case_insensitive += 1

total_found = exact_match + case_insensitive
print(f"Non-'none' values: {len(non_none_examples):,}")
print(f"Exact match in dialogue: {exact_match:,} ({exact_match/len(non_none_examples)*100:.2f}%)")
print(f"Case-insensitive match: {case_insensitive:,} ({case_insensitive/len(non_none_examples)*100:.2f}%)")
print(f"Total found: {total_found:,} ({total_found/len(non_none_examples)*100:.2f}%)")
print(f"NOT found: {len(non_none_examples)-total_found:,} ({(len(non_none_examples)-total_found)/len(non_none_examples)*100:.2f}%)")

# Sample by domain
print("\n6. SAMPLE DIALOGUES BY DOMAIN")
print("-" * 100)

domains = {}
for ex in train_examples[:5000]:
    domain = ex.get("slot_name", "").split("-")[0]
    if domain not in domains:
        domains[domain] = []
    if len(domains[domain]) < 3:
        domains[domain].append(ex)

for domain in sorted(domains.keys()):
    print(f"\n{domain.upper()} DOMAIN:")
    for i, ex in enumerate(domains[domain], 1):
        print(f"  Example {i}:")
        print(f"    Slot: {ex['slot_name']}")
        print(f"    Value: '{ex['target_value']}'")
        print(f"    Context (first 150 chars): {ex['dialogue_context'][:150]}...")

# Slot distribution by domain
print("\n7. SLOTS BY DOMAIN")
print("-" * 100)

slots_by_domain = {}
for slot, count in slots.items():
    domain = slot.split("-")[0]
    if domain not in slots_by_domain:
        slots_by_domain[domain] = {}
    slots_by_domain[domain][slot] = count

for domain in sorted(slots_by_domain.keys()):
    print(f"\n{domain.upper()} ({len(slots_by_domain[domain])} slots):")
    for slot, count in sorted(slots_by_domain[domain].items(), key=lambda x: x[1], reverse=True):
        print(f"  {slot}: {count:,}")

# Top target values
print("\n8. TOP 15 TARGET VALUES")
print("-" * 100)
target_values = Counter(ex.get("target_value") for ex in train_examples)
for value, count in target_values.most_common(15):
    print(f"  '{value}': {count:,}")

print("\n" + "=" * 100)
print("INSPECTION COMPLETE")
print("=" * 100)
