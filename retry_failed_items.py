import json
import asyncio
import os
from dotenv import load_dotenv
from src import prompt
from src.request import async_api_requests

load_dotenv()

print("="*60)
print("RETRY FAILED ITEMS (Rate Limited)")
print("="*60)

# Step 1: Find failed items
print("\n[STEP 1] Finding failed items...")
with open('results/title/title_few-shot_test.json', 'r') as f:
    data = json.load(f)

failed_ids = []
for item in data:
    res = item.get('response')
    if not isinstance(res, dict) or 'choices' not in res:
        failed_ids.append(item['id'])

print(f"✅ Found {len(failed_ids)} failed items")

if len(failed_ids) == 0:
    print("✅ No failed items!")
    exit(0)

# Step 2: Generate prompts for failed IDs
print("\n[STEP 2] Generating prompts...")
task = "title"
dataset = "title_itape"
method = "few-shot"
test_val = "test"

all_prompts = prompt.generate_prompt(
    root=os.path.join(os.getcwd(), 'data'),
    task=task,
    dataset=dataset,
    method=method,
    TEST=test_val,
    testNum=33438
)

failed_prompts = [p for p in all_prompts if p['id'] in failed_ids]
print(f"✅ Found {len(failed_prompts)} prompts for failed items")

# Step 3: Confirm
print(f"\n⚠️  This will retry {len(failed_prompts)} failed items")
print(f"   Using same rate limits:")
print(f"   - RPM: 500")
print(f"   - TPM: 200000")
print(f"   Estimated time: ~{len(failed_prompts)/500*60:.1f} minutes")

response = input("\nProceed? (yes/no): ").strip().lower()
if response != 'yes':
    print("Cancelled.")
    exit(0)

# Step 4: API Config
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\n[ERROR] OPENAI_API_KEY not found!")
    exit(1)

model = "gpt-4o-mini"
rpm = 500
tpm = 200000

# Step 5: Run retry
print(f"\n[STEP 3] Starting retry...")
result_output_path = os.path.join(os.getcwd(), 'results', task)
retry_filename = f"{task}_{method}_{test_val}_retry_failed"

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
        testNum=len(failed_prompts),
        method=method,
        data=failed_prompts
    )
)

# Step 6: Replace failed items in Main
print(f"\n[STEP 4] Replacing failed items in Main file...")
retry_file = os.path.join(result_output_path, retry_filename + ".json")

if not os.path.exists(retry_file):
    print("❌ Retry file not found!")
    exit(1)

with open(retry_file, 'r') as f:
    retry_data = json.load(f)

print(f"✅ Retry results: {len(retry_data)} items")

# Create lookup dict
retry_dict = {item['id']: item for item in retry_data}

# Replace failed items
import shutil
shutil.copy('results/title/title_few-shot_test.json',
            'results/title/title_few-shot_test_before_retry_failed.json')

updated_data = []
replaced_count = 0

for item in data:
    if item['id'] in retry_dict:
        # Replace with new result
        updated_data.append(retry_dict[item['id']])
        replaced_count += 1
    else:
        # Keep original
        updated_data.append(item)

print(f"✅ Replaced {replaced_count} failed items")

# Save
with open('results/title/title_few-shot_test.json', 'w') as f:
    json.dump(updated_data, f, indent=4)

# Summary
successful = sum(1 for item in updated_data if isinstance(item.get('response'), dict) and 'choices' in item.get('response'))
print("\n" + "="*60)
print("RETRY COMPLETE!")
print("="*60)
print(f"📊 Total items: {len(updated_data)}")
print(f"   Successful: {successful}")
print(f"   Failed: {len(updated_data) - successful}")
print(f"   Success rate: {successful/len(updated_data)*100:.1f}%")
print("="*60)