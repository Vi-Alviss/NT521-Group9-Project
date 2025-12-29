import json
import shutil
import os

print("="*60)
print("MERGE PART8_NEW → MAIN FILE")
print("="*60)

# Backup Main file
print("\n📦 Creating backup...")
shutil.copy('results/title/title_few-shot_test.json',
            'results/title/title_few-shot_test_backup.json')
print("✅ Backup saved")

# Load Main file
print("\n📂 Loading files...")
with open('results/title/title_few-shot_test.json', 'r') as f:
    main = json.load(f)
print(f"✅ Main file: {len(main)} items")

main_ids = [item['id'] for item in main]
print(f"   Range: {min(main_ids)} → {max(main_ids)}")

# Load Part8_NEW
with open('results/title/title_few-shot_test_part8_new.json', 'r') as f:
    part8_new = json.load(f)
print(f"✅ Part8_NEW: {len(part8_new)} items")
print(f"   Range: {part8_new[0]['id']} → {part8_new[-1]['id']}")
# Merge
print("\n🔀 Merging Main + Part8_NEW...")
merged = main + part8_new
print(f"✅ Merged: {len(main)} + {len(part8_new)} = {len(merged)} items")

# Check duplicates
ids = [item['id'] for item in merged]
unique = set(ids)
if len(ids) != len(unique):
    print(f"⚠️  WARNING: Found {len(ids) - len(unique)} duplicates!")
else:
    print(f"✅ No duplicates")

# Save merged Main
print("\n💾 Saving merged Main file...")
with open('results/title/title_few-shot_test.json', 'w') as f:
    json.dump(merged, f, indent=4)

print("✅ Main file updated!")

# Clean up Part8_NEW (NOT part4_new!)
print("\n🗑️  Cleaning up...")
os.remove('results/title/title_few-shot_test_part8_new.json')  # ← SỬA ĐÂY
print("✅ Removed: part8_new.json")

# Summary
sorted_ids = sorted([int(id.split('-')[1]) for id in ids])
print("\n" + "="*60)
print("MERGE COMPLETE!")
print("="*60)
print(f"📊 Total items: {len(merged)}")
print(f"   Range: title-{sorted_ids[0]} → title-{sorted_ids[-1]}")
print(f"   Target: 33,438 items")
print(f"   Progress: {len(merged)/33438*100:.1f}%")
print(f"   Remaining: {33438 - len(merged)} items")
print(f"\n✅ Next resume from: title-{sorted_ids[-1] + 1}")
print(f"   Update resume_from_checkpoint.py:")
print(f"   start_from_index = {sorted_ids[-1] + 1}")
print("="*60)