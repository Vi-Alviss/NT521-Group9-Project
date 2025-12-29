import asyncio
import os
import json
from dotenv import load_dotenv
from src import prompt
from src.request import async_api_requests
from src.metrics import calculate_title_metrics

load_dotenv()

def print_token_summary(filepath):
    """Parses the result JSON to calculate total tokens and estimated cost."""
    if not os.path.exists(filepath):
        return

    total_prompt = 0
    total_completion = 0
    successful_requests = 0

    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return

    for item in data:
        res = item.get('response')
        if isinstance(res, dict) and 'usage' in res:
            usage = res['usage']
            total_prompt += usage.get('prompt_tokens', 0)
            total_completion += usage.get('completion_tokens', 0)
            successful_requests += 1

    input_cost = (total_prompt / 1_000_000) * 0.15
    output_cost = (total_completion / 1_000_000) * 0.60
    total_cost = input_cost + output_cost

    summary_output = (
        f"\n" + "="*55 + "\n"
        f"RUN SUMMARY (Token Usage & Estimated Cost)\n"
        f"{'-' * 55}\n"
        f"Successful Requests:  {successful_requests}\n"
        f"Total Prompt Tokens:  {total_prompt:,}\n"
        f"Total Completion:     {total_completion:,}\n"
        f"Total Tokens Used:    {total_prompt + total_completion:,}\n"
        f"Estimated Cost:       ${total_cost:.6f} (USD)\n"
        f"{'='*55}"
    )
    print(summary_output)
    return summary_output

def main():
    print("="*60)
    print("RESUME: ChatGPT Vulnerability Management from Checkpoint")
    print("="*60)

    # Configuration from previous run
    task = "title"
    dataset = "title_itape"
    method = "few-shot"
    test_val = "test"
    
    # CHECKPOINT SETTINGS (based on analysis)
    start_from_index = 32424  # Array index to start from
    total_items = 33438  # Total dataset size
    
    print(f"\n[CHECKPOINT INFO]")
    print(f"  Task: {task}")
    print(f"  Dataset: {dataset}")
    print(f"  Method: {method}")
    print(f"  Resuming from index: {start_from_index} (ID: title-{start_from_index})")
    print(f"  Remaining items: {total_items - start_from_index}")
    print(f"  Progress: {start_from_index}/{total_items} ({start_from_index/total_items*100:.1f}%)")
    
    # API Configuration
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY not found in .env file!")
        return
    
    model = "gpt-4o-mini"
    rpm = 500
    tpm = 200000
    
    print(f"\n[API CONFIG]")
    print(f"  Model: {model}")
    print(f"  Rate limits: {rpm} RPM, {tpm} TPM")
    
    # Setup paths
    root_data_path = os.path.join(os.getcwd(), 'data')
    result_output_path = os.path.join(os.getcwd(), 'results', task)
    
    # File paths
    temp_filename = f"{task}_{method}_{test_val}_part9"
    temp_new_filename = f"{task}_{method}_{test_val}_part9_new"  # New temp file
    final_filename = f"{task}_{method}_{test_val}"
    
    temp_result_path = os.path.join(result_output_path, temp_filename + ".json")
    temp_new_result_path = os.path.join(result_output_path, temp_new_filename + ".json")
    final_result_path = os.path.join(result_output_path, final_filename + ".json")
    
    # Backup existing Part 2 if it exists
    if os.path.exists(temp_result_path):
        backup_part2 = temp_result_path.replace('.json', '_before_resume.json')
        import shutil
        shutil.copy(temp_result_path, backup_part2)
        print(f"✅ Part 2 backed up to: {backup_part2}")
        
        # Load existing part2 data
        with open(temp_result_path, 'r') as f:
            existing_part2 = json.load(f)
        print(f"✅ Existing Part 2 has {len(existing_part2)} items")
    else:
        existing_part2 = []
        print("⚠️  No existing Part 2 found, starting fresh")
    
    print(f"\n--- Generating prompts for remaining items ---")
    generated_prompts = prompt.generate_prompt(
        root=root_data_path,
        task=task,
        dataset=dataset,
        method=method,
        TEST=test_val,
        testNum=total_items
    )
    
    print(f"Total prompts generated: {len(generated_prompts)}")
    
    # Extract only remaining items (from index start_from_index onwards)
    remaining_prompts = generated_prompts[start_from_index:]
    print(f"Remaining prompts to process: {len(remaining_prompts)}")
    print(f"First ID in remaining batch: {remaining_prompts[0]['id']}")
    print(f"Last ID in remaining batch: {remaining_prompts[-1]['id']}")
    
    # Confirm before proceeding
    print(f"\n⚠️  This will process {len(remaining_prompts)} items.")
    print(f"   Estimated time: ~{len(remaining_prompts) / rpm * 60:.1f} minutes")
    print(f"   New items will be APPENDED to existing Part 2")
    response = input("\nProceed? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Cancelled.")
        return
    
    print(f"\n--- Starting API Requests ---")
    request_url = "https://api.openai.com/v1/chat/completions"
    
    # Process new items into a separate temp file
    asyncio.run(
        async_api_requests(
            max_requests_per_minute=rpm,
            max_tokens_per_minute=tpm,
            request_url=request_url,
            api_key=api_key,
            root_path=root_data_path,
            result_file_path=result_output_path,
            result_file_name=temp_new_filename,  # Use separate temp file
            task=task,
            dataset=dataset,
            model=model,
            dataNum=0,
            testNum=len(remaining_prompts),
            method=method,
            data=remaining_prompts
        )
    )
    
    # Merge existing Part 2 + new results
    print(f"\n--- Merging Part 2: existing + new results ---")
    
    if os.path.exists(temp_new_result_path):
        with open(temp_new_result_path, 'r') as f:
            new_part2_results = json.load(f)
        print(f"✅ New batch has {len(new_part2_results)} items")
    else:
        print("ERROR: New results file not found!")
        return
    
    # Combine existing + new
    combined_part2 = existing_part2 + new_part2_results
    print(f"✅ Combined Part 2: {len(existing_part2)} + {len(new_part2_results)} = {len(combined_part2)} items")
    
    # Save combined Part 2
    with open(temp_result_path, 'w') as f:
        json.dump(combined_part2, f, indent=4)
    print(f"✅ Part 2 updated: {temp_result_path}")
    
    # Clean up temporary new file
    os.remove(temp_new_result_path)
    
    # Final merge: Part 1 + Part 2 (combined)
    print(f"\n--- Final merge: Part 1 + Part 2 ---")
    
    # Load Part 1
    with open(final_result_path, 'r') as f:
        part1_results = json.load(f)
    print(f"Part 1 has {len(part1_results)} items")
    
    # Merge Part 1 + combined Part 2
    merged_results = part1_results + combined_part2
    
    # Backup original final file
    backup_path = final_result_path.replace('.json', '_backup.json')
    import shutil
    shutil.copy(final_result_path, backup_path)
    print(f"✅ Final file backed up to: {backup_path}")
    
    # Save merged results
    with open(final_result_path, 'w') as f:
        json.dump(merged_results, f, indent=4)
    
    print(f"✅ Final merge complete: {len(part1_results)} + {len(combined_part2)} = {len(merged_results)} items")
    print(f"✅ Final results saved to: {final_result_path}")
    
    # Token summary and metrics
    token_report = print_token_summary(final_result_path)
    
    if task == "title" and os.path.exists(final_result_path):
        print(f"\n--- Calculating ROUGE Metrics ---")
        report = calculate_title_metrics(final_result_path)
        if report:
            header = (
                f"\n" + "="*55 + "\n"
                f"ROUGE Performance Metrics for: {final_filename}\n"
                f"{'='*55}\n"
                f"{'Metric':<12} | {'F1 (%)':<10} | {'Prec (%)':<10} | {'Rec (%)':<10}\n"
                f"{'-' * 55}"
            )
            print(header)
            
            rows = ""
            for m in ['rouge1', 'rouge2', 'rougeL']:
                row = f"{m.upper():<12} | {report[m]['F1']:>10.2f} | {report[m]['Precision']:>10.2f} | {report[m]['Recall']:>10.2f}"
                print(row)
                rows += row + "\n"
            print("-" * 55)

            metrics_txt_path = final_result_path.replace(".json", ".txt")
            with open(metrics_txt_path, "w") as f:
                if token_report:
                    f.write(token_report + "\n\n")
                f.write(header + "\n")
                f.write(rows)
                f.write("-" * 55 + "\n")
            
            print(f"\n✅ Metrics saved to: {metrics_txt_path}")
        else:
            print("⚠️  Metric calculation failed.")
    
    print("\n" + "="*60)
    print("RESUME COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()

