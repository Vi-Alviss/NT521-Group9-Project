import json

filepath = "results/title/title_few-shot_test.json"

with open(filepath, 'r') as f:
    data = json.load(f)

ids = [item['id'] for item in data]
unique_ids = set(ids)

print("="*60)
print("CHECKPOINT ANALYSIS")
print("="*60)
print(f"Total items: {len(ids)}")
print(f"Unique IDs: {len(unique_ids)}")
print(f"Min ID: {min(ids)}")
print(f"Max ID: {max(ids)}")
print(f"Has duplicates: {len(ids) != len(unique_ids)}")
print(f"\n✅ Resume from ID: {max(ids) + 1}")
print("="*60)
