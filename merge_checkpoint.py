import json
import shutil

print("="*60)
print("MERGE PART2 + PART2_NEW")
print("="*60)

# Backup Part2 cũ
print("\n📦 Creating backup...")
shutil.copy('results/title/title_few-shot_test_part2.json',
            'results/title/title_few-shot_test_part2_backup.json')
print("✅ Backup saved: title_few-shot_test_part2_backup.json")

# Load cả 2 files
print("\n📂 Loading files...")
with open('results/title/title_few-shot_test_part2.json', 'r') as f:
    part2_old = json.load(f)
print(f"✅ Part2 (old): {len(part2_old)} items")

with open('results/title/title_few-shot_test_part2_new.json', 'r') as f:
    part2_new = json.load(f)
print(f"✅ Part2 (new): {len(part2_new)} items")

# Merge
print("\n🔀 Merging...")
merged = part2_old + part2_new
print(f"✅ Merged: {len(part2_old)} + {len(part2_new)} = {len(merged)} items")
print(f"   Range: {merged[0]['id']} → {merged[-1]['id']}")

# Ghi đè Part2
print("\n💾 Saving merged Part2...")
with open('results/title/title_few-shot_test_part2.json', 'w') as f:
    json.dump(merged, f, indent=4)

print("✅ Part2 updated successfully!")

# Cleanup Part2_NEW
print("\n🗑️  Cleaning up...")
import os
os.remove('results/title/title_few-shot_test_part2_new.json')
print("✅ Removed: title_few-shot_test_part2_new.json")

print("\n" + "="*60)
print("MERGE COMPLETE!")
print("="*60)
print(f"\n📊 Next resume from: title-17222")
print(f"   Update resume_from_checkpoint.py:")
print(f"   start_from_index = 17222")
print("="*60)