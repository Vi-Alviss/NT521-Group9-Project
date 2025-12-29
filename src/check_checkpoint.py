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
    
    # CHECKPOINT SETTINGS
    start_from = 25441  # Resume from this item
    total_items = 33438  # Total dataset size
    
    print(f"\n[CHECKPOINT INFO]")
    print(f"  Task: {task}")
    print(f"  Dataset: {dataset}")
    print(f"  Method: {method}")
    print(f"  Resuming from item: {start_from}")
    print(f"  Remaining items: {total_items - start_from}")
    print(f"  Progress: {start_from}/{total_items} ({start_from/total_items*100:.1f}%)")
    
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
    
    # Create temporary file for new results
    temp_filename = f"{task}_{method}_{test_val}_part2"
    final_filename = f"{task}_{method}_{test_val}"
    
    temp_result_path = os.path.join(result_output_path, temp_filename + ".json")
    final_result_path = os.path.join(result_output_path, final_filename + ".json")
    
    print(f"\n--- Generating prompts for remaining items ---")
    generated_prompts = prompt.generate_prompt(
        root=root_data_path,
        task=task,
        dataset=dataset,
        method=method,
        TEST=test_val,
        testNum=total_items
    )
    
    # Extract only remaining items
    remaining_prompts = generated_prompts[start_from:]
    print(f"Remaining prompts to process: {len(remaining_prompts)}")
    
    print(f"\n--- Starting API Requests for remaining {len(remaining_prompts)} items ---")
    request_url = "https://api.openai.com/v1/chat/completions"
    
    asyncio.run(
        async_api_requests(
            max_requests_per_minute=rpm,
            max_tokens_per_minute=tpm,
            request_url=request_url,
            api_key=api_key,
            root_path=root_data_path,
            result_file_path=result_output_path,
            result_file_name=temp_filename,
            task=task,
            dataset=dataset,
            model=model,
            dataNum=0,  # Start from 0 in the sliced list
            testNum=len(remaining_prompts),
            method=method,
            data=remaining_prompts
        )
    )
    
    print(f"\n--- Merging results ---")
    # Load existing results
    with open(final_result_path, 'r') as f:
        existing_results = json.load(f)
    
    # Load new results
    with open(temp_result_path, 'r') as f:
        new_results = json.load(f)
    
    # Merge
    merged_results = existing_results + new_results
    
    # Save merged results
    with open(final_result_path, 'w') as f:
        json.dump(merged_results, f, indent=4)
    
    print(f"✅ Merged {len(existing_results)} + {len(new_results)} = {len(merged_results)} items")
    print(f"✅ Final results saved to: {final_result_path}")
    
    # Token summary and metrics
    token_report = print_token_summary(final_result_path)
    
    if task == "title" and os.path.exists(final_result_path):
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

if __name__ == "__main__":
    main()