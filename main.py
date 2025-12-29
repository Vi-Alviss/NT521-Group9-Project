import asyncio
import os
import json
from dotenv import load_dotenv  # New import
from src import prompt
from src.request import async_api_requests
from src.metrics import calculate_title_metrics

# Load variables from .env file
load_dotenv()

def get_user_input(prompt_text, options=None):
    while True:
        val = input(prompt_text).strip()
        if options and val not in options:
            print(f"Invalid input. Please choose from: {', '.join(options)}")
            continue
        return val

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
        # Check if the response is a dictionary (successful API call)
        res = item.get('response')
        if isinstance(res, dict) and 'usage' in res:
            usage = res['usage']
            total_prompt += usage.get('prompt_tokens', 0)
            total_completion += usage.get('completion_tokens', 0)
            successful_requests += 1

    # Cost Calculation for gpt-4o-mini
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
    print("="*50)
    print("ChatGPT Vulnerability Management - Interactive Runner")
    print("="*50)

    # 1. TASK MAPPING
    print("\n[1] Select Task:")
    print("1. title (dataset: title_itape)")
    print("2. SBRP  (dataset: Chromium)")
    print("3. cvss  (dataset: AV, AC, PR, or UI)")
    print("4. vulfix(dataset: vulfix_extractfix)")
    print("5. APCA  (dataset: APCA_quatrain, APCA_panther, or APCA_invalidator)")
    print("6. stable(dataset: stable_patchnet)")
    
    task_map = {
        "1": "title", "2": "SBRP", "3": "cvss", 
        "4": "vulfix", "5": "APCA", "6": "stable"
    }
    task_choice = get_user_input("Enter choice (1-6): ", options=task_map.keys())
    task = task_map[task_choice]

    # 2. DATASET INPUT
    print(f"\n[2] Enter Dataset name for {task}:")
    dataset = input("Dataset name: ").strip()

    # 3. METHOD MAPPING
    print("\n[3] Select Method:")
    print("1. base")
    print("2. one-shot")
    print("3. few-shot")
    print("4. info-manual (or manual-info)")
    print("5. prompt-eng")
    print("6. info-gpt")
    print("7. summary")
    
    method_map = {
        "1": "base", "2": "one-shot", "3": "few-shot", 
        "4": "info-manual", "5": "prompt-eng", "6": "info-gpt", "7": "summary"
    }
    method_choice = get_user_input("Enter choice (1-7): ", options=method_map.keys())
    method = method_map[method_choice]

    # 4. TEST MAPPING
    print("\n[4] Select Split (TEST):")
    print("1. vali (Loads -probe.json)")
    print("2. test (Loads -test.json)")
    print("3. remain")
    
    test_map = {"1": "vali", "2": "test", "3": "remain"}
    test_choice = get_user_input("Enter choice (1-3): ", options=test_map.keys())
    test_val = test_map[test_choice]

    # 5. CONFIGURATION
    print("\n[5] Configuration:")
    # Check if key exists in .env, otherwise ask user
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter OpenAI API Key: ").strip()
    else:
        print("API Key loaded successfully from .env")
        
    model = input("Model (default: gpt-4o-mini): ") or "gpt-4o-mini"
    test_num = int(input("Number of items to test (default 1): ") or 1)

    # --- DYNAMIC RATE LIMIT LOGIC ---
    if "gpt-4o-mini" in model:
        rpm = 500
        tpm = 200000
    elif "gpt-4" in model:
        rpm = 500
        tpm = 10000
    else:
        rpm = 3
        tpm = 40000
    print(f"Rate limits set for {model}: {rpm} RPM, {tpm} TPM")
    
    # 6. SETUP & EXECUTION
    root_data_path = os.path.join(os.getcwd(), 'data')
    result_output_path = os.path.join(os.getcwd(), 'results', task)
    dynamic_filename = f"{task}_{method}_{test_val}"
    evaluation_full_result_path = os.path.join(result_output_path, dynamic_filename + ".json")

    print(f"\n--- Generating prompts for {task}/{dataset} ---")
    generated_prompts = prompt.generate_prompt(
        root=root_data_path,
        task=task,
        dataset=dataset,
        method=method,
        TEST=test_val,
        testNum=test_num
    )

    print(f"--- Starting API Requests. Output: {dynamic_filename}.json ---")
    request_url = "https://api.openai.com/v1/chat/completions"
    
    asyncio.run(
        async_api_requests(
            max_requests_per_minute=rpm,
            max_tokens_per_minute=tpm,
            request_url=request_url,
            api_key=api_key,
            root_path=root_data_path,
            result_file_path=result_output_path,
            result_file_name=dynamic_filename,
            task=task,
            dataset=dataset,
            model=model,
            dataNum=0,
            testNum=test_num,
            method=method,
            data=generated_prompts
        )
    )

    # 7. AUTOMATED EVALUATION & TOKEN SUMMARY
    token_report = print_token_summary(evaluation_full_result_path)

    if task == "title" and os.path.exists(evaluation_full_result_path):
        report = calculate_title_metrics(evaluation_full_result_path)
        if report:
            # Prepare strings for both printing and saving
            header = (
                f"\n" + "="*55 + "\n"
                f"ROUGE Performance Metrics for: {dynamic_filename}\n"
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

            # SAVE TO TEXT FILE
            metrics_txt_path = evaluation_full_result_path.replace(".json", ".txt")
            with open(metrics_txt_path, "w") as f:
                if token_report:
                    f.write(token_report + "\n\n")
                f.write(header + "\n")
                f.write(rows)
                f.write("-" * 55 + "\n")
            
            print(f"\n[Success] Metrics saved to: {metrics_txt_path}")
        else:
            print("Metric calculation failed.")

if __name__ == "__main__":
    main()