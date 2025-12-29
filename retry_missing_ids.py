import json
import asyncio
import os
from dotenv import load_dotenv
from src import prompt
from src.request import async_api_requests

load_dotenv()

print("="*60)
print("RETRY MISSING IDs")
print("="*60)

# Step 1: Find missing IDs in Main file
print("\n[STEP 1] Analyzing Main file...")
with open('results/title/title_few-shot_test.json', 'r') as f:
    main_data = json.load(f)

main_ids = [int(item['id'].split('-')[1]) for item in main_data]
main_ids_set = set(main_ids)
min_id, max_id = min(main_ids), max(main_ids)
expected = set(range(min_id, max_id + 1))
missing = sorted(expected - main_ids_set)

print(f"✅ Main file: {len(main_data)} items")
print(f"   Range: title-{min_id} → title-{max_id}")
print(f"   Missing: {len(missing)} items")

# Step 2: Check if missing IDs exist in original dataset
print("\n[STEP 2] Checking original dataset...")
dataset_path = 'data/title/title_itape-test.json'

if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found: {dataset_path}")
    exit(1)

with open(dataset_path, 'r') as f:
    dataset_raw = json.load(f)

# Extract actual data from dict structure
if isinstance(dataset_raw, dict) and 'title_itape' in dataset_raw:
    original_data = dataset_raw['title_itape']
    print(f"✅ Original dataset: {len(original_data)} items (dict format)")
else:
    original_data = dataset_raw
    print(f"✅ Original dataset: {len(original_data)} items")

# Step 3: Generate prompts for missing IDs only
print("\n[STEP 3] Generating prompts for missing IDs...")
task = "title"
dataset = "title_itape"
method = "few-shot"
test_val = "test"

# Generate all prompts first
all_prompts = prompt.generate_prompt(
    root=os.path.join(os.getcwd(), 'data'),
    task=task,
    dataset=dataset,
    method=method,
    TEST=test_val,
    testNum=33438
)

# Filter only missing IDs
missing_prompts = []
for p in all_prompts:
    prompt_id = int(p['id'].split('-')[1])
    if prompt_id in missing:
        missing_prompts.append(p)

print(f"✅ Found {len(missing_prompts)} prompts for missing IDs")

if len(missing_prompts) == 0:
    print("\n✅ No missing IDs to retry!")
    exit(0)

print(f"\n📋 Sample missing IDs:")
for i in range(min(10, len(missing_prompts))):
    print(f"   {missing_prompts[i]['id']}")
if len(missing_prompts) > 10:
    print(f"   ... and {len(missing_prompts) - 10} more")

# Step 4: Confirm retry
print(f"\n⚠️  This will retry {len(missing_prompts)} missing items.")
print(f"   Estimated time: ~{len(missing_prompts)/500*60:.1f} minutes @ 500 RPM")
response = input("\nProceed? (yes/no): ").strip().lower()
if response != 'yes':
    print("Cancelled.")
    exit(0)

# Step 5: API Configuration
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\n[ERROR] OPENAI_API_KEY not found in .env file!")
    exit(1)

model = "gpt-4o-mini"
rpm = 500
tpm = 200000

# Step 6: Run retry
print(f"\n[STEP 4] Starting API requests for missing IDs...")
result_output_path = os.path.join(os.getcwd(), 'results', task)
retry_filename = f"{task}_{method}_{test_val}_retry_missing"

asyncio.run(
    async_api_requests(
        max_requests_per_minute=rpm,
        max_tokens_per_minute=tpm,
        request_url="https://api.openai.com/v1/chat/completions",
        api_key=api_key,
        root_path=os.path.join(os.getcwd(), 'data'),
        result_file_path=result_output_path,
        result_file_name=retry_filename,
        task=task,
        dataset=dataset,
        model=model,
        dataNum=0,
        testNum=len(missing_prompts),
        method=method,
        data=missing_prompts
    )
)

# Step 7: Merge retry results with Main
print(f"\n[STEP 5] Merging retry results with Main file...")
retry_file = os.path.join(result_output_path, retry_filename + ".json")

if not os.path.exists(retry_file):
    print("❌ Retry file not found!")
    exit(1)

with open(retry_file, 'r') as f:
    retry_data = json.load(f)

print(f"✅ Retry results: {len(retry_data)} items")

# Merge
import shutil
shutil.copy('results/title/title_few-shot_test.json',
            'results/title/title_few-shot_test_before_retry.json')

merged = main_data + retry_data
print(f"✅ Merged: {len(main_data)} + {len(retry_data)} = {len(merged)} items")

# Save
with open('results/title/title_few-shot_test.json', 'w') as f:
    json.dump(merged, f, indent=4)

print(f"\n✅ Main file updated!")

# Final summary
merged_ids = sorted([int(item['id'].split('-')[1]) for item in merged])
print("\n" + "="*60)
print("RETRY COMPLETE!")
print("="*60)
print(f"📊 Total items: {len(merged)}")
print(f"   Range: title-{merged_ids[0]} → title-{merged_ids[-1]}")
print(f"   Target: 33,438")
print(f"   Progress: {len(merged)/33438*100:.1f}%")
print("="*60)