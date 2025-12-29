import json
import numpy as np
import os
from rouge_score import rouge_scorer

def calculate_title_metrics(evaluation_full_result_path):
    """
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L (F1, Precision, Recall)
    for the title summarization task based on the generated result file.
    """
    if not os.path.exists(evaluation_full_result_path):
        print(f"Result file not found: {evaluation_full_result_path}")
        return None

    try:
        with open(evaluation_full_result_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error reading result file: {e}")
        return None

    # Initialize the scorer with standard academic settings
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Accumulators for metrics
    scores_acc = {
        'rouge1': {'f1': [], 'p': [], 'r': []},
        'rouge2': {'f1': [], 'p': [], 'r': []},
        'rougeL': {'f1': [], 'p': [], 'r': []}
    }

    for entry in results:
        # 1. Extract ground truth (Reference)
        ref = entry.get('ground_truth', '')
        
        # 2. Extract generated content from OpenAI response structure
        try:
            # Based on the structure: response['choices'][0]['message']['content']
            gen = entry['response']['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError, TypeError):
            continue

        if not ref or not gen:
            continue

        # 3. Calculate Scores
        scores = scorer.score(ref, gen)
        
        # 4. Store metrics
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            scores_acc[metric]['f1'].append(scores[metric].fmeasure)
            scores_acc[metric]['p'].append(scores[metric].precision)
            scores_acc[metric]['r'].append(scores[metric].recall)

    # Calculate averages and format as percentages for the report
    final_report = {}
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        final_report[metric] = {
            'F1': np.mean(scores_acc[metric]['f1']) * 100 if scores_acc[metric]['f1'] else 0,
            'Precision': np.mean(scores_acc[metric]['p']) * 100 if scores_acc[metric]['p'] else 0,
            'Recall': np.mean(scores_acc[metric]['r']) * 100 if scores_acc[metric]['r'] else 0
        }

    return final_report