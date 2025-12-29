import os
import json
from src.metrics import calculate_title_metrics

def print_token_summary(filepath):
    """Parses the result JSON to calculate total tokens and estimated cost."""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None

    total_prompt = 0
    total_completion = 0
    successful_requests = 0
    failed_requests = 0

    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("❌ Invalid JSON file")
            return None

    for item in data:
        res = item.get('response')
        if isinstance(res, dict) and 'usage' in res:
            usage = res['usage']
            total_prompt += usage.get('prompt_tokens', 0)
            total_completion += usage.get('completion_tokens', 0)
            successful_requests += 1
        else:
            failed_requests += 1

    # Cost Calculation for gpt-4o-mini
    input_cost = (total_prompt / 1_000_000) * 0.15
    output_cost = (total_completion / 1_000_000) * 0.60
    total_cost = input_cost + output_cost

    summary_output = (
        f"\n" + "="*60 + "\n"
        f"TOKEN USAGE & ESTIMATED COST\n"
        f"{'-' * 60}\n"
        f"Total Items:          {len(data):,}\n"
        f"Successful Requests:  {successful_requests:,}\n"
        f"Failed Requests:      {failed_requests:,}\n"
        f"Total Prompt Tokens:  {total_prompt:,}\n"
        f"Total Completion:     {total_completion:,}\n"
        f"Total Tokens Used:    {total_prompt + total_completion:,}\n"
        f"Estimated Cost:       ${total_cost:.6f} (USD)\n"
        f"{'='*60}"
    )
    print(summary_output)
    return summary_output


def main():
    print("="*60)
    print("CALCULATE METRICS FOR MAIN FILE")
    print("="*60)
    
    # File path
    result_file = 'results/title/title_few-shot_test.json'
    
    if not os.path.exists(result_file):
        print(f"\n❌ File not found: {result_file}")
        return
    
    # Load and analyze
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n📂 File: {result_file}")
    print(f"   Total items: {len(data):,}")
    
    # Get ID range
    ids = [int(item['id'].split('-')[1]) for item in data]
    print(f"   ID range: title-{min(ids)} → title-{max(ids)}")
    
    # Token summary
    print(f"\n{'='*60}")
    print("STEP 1: TOKEN USAGE ANALYSIS")
    print(f"{'='*60}")
    token_report = print_token_summary(result_file)
    
    # ROUGE metrics
    print(f"\n{'='*60}")
    print("STEP 2: ROUGE METRICS CALCULATION")
    print(f"{'='*60}")
    
    report = calculate_title_metrics(result_file)
    
    if report:
        header = (
            f"\n" + "="*60 + "\n"
            f"ROUGE PERFORMANCE METRICS\n"
            f"{'='*60}\n"
            f"{'Metric':<12} | {'F1 (%)':<10} | {'Prec (%)':<10} | {'Rec (%)':<10}\n"
            f"{'-' * 60}"
        )
        print(header)
        
        rows = ""
        for m in ['rouge1', 'rouge2', 'rougeL']:
            row = f"{m.upper():<12} | {report[m]['F1']:>10.2f} | {report[m]['Precision']:>10.2f} | {report[m]['Recall']:>10.2f}"
            print(row)
            rows += row + "\n"
        print("-" * 60)
        
        # Save to TXT file
        metrics_txt_path = result_file.replace(".json", "_metrics.txt")
        with open(metrics_txt_path, "w") as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION METRICS REPORT\n")
            f.write(f"File: {result_file}\n")
            f.write(f"Total items: {len(data):,}\n")
            f.write(f"ID range: title-{min(ids)} → title-{max(ids)}\n")
            f.write("="*60 + "\n\n")
            
            if token_report:
                f.write(token_report + "\n\n")
            
            f.write(header + "\n")
            f.write(rows)
            f.write("-" * 60 + "\n")
        
        print(f"\n✅ Metrics saved to: {metrics_txt_path}")
    else:
        print("❌ ROUGE metric calculation failed.")
    
    print("\n" + "="*60)
    print("METRICS CALCULATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()