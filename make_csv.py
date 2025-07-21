import json
import pandas as pd

def calculate_accuracies(group):
    total_questions = len(group)
    total_correct = group[group['score'] == "A"].shape[0]
    total_incorrect = group[group['score'] == "B"].shape[0]
    total_not_attempted = group[group['score'] == "C"].shape[0]
    
    total_correct_accuracy = total_correct / total_questions if total_questions > 0 else 0
    total_incorrect_accuracy = total_incorrect / total_questions if total_questions > 0 else 0
    total_not_attempted_accuracy = total_not_attempted / total_questions if total_questions > 0 else 0
    
    total_given_attempted_accuracy = total_correct / (total_correct + total_incorrect) if (total_correct + total_incorrect) > 0 else 0
    
    f1 = 2 * total_given_attempted_accuracy * total_correct_accuracy / (total_given_attempted_accuracy+ total_correct_accuracy) if (total_given_attempted_accuracy+ total_correct_accuracy) > 0 else 0

    return pd.Series({'correct': total_correct_accuracy, 'incorrect': total_incorrect_accuracy, 'not_attempted': total_not_attempted_accuracy, "given_attempted_accuracy": total_given_attempted_accuracy, "F1": f1})


new_file = "ChineseSimpleQA-master/evaluation/models_results/simpleqa_DeepSeek-R1-Distill-Qwen-7B_!t=0.4.jsonl"


with open(new_file, "r", encoding='utf-8') as fin:
    lines = fin.readlines()
    datas = [json.loads(line) for line in lines]
    df = pd.json_normalize(datas)
    accuracy_df = calculate_accuracies(df)

    overall_row = pd.DataFrame({
        'primary_category': ['Overall'],
        'correct': [accuracy_df['correct']],
        'incorrect': [accuracy_df['incorrect']],
         'not_attempted': [accuracy_df['not_attempted']],
        'given_attempted_accuracy': [accuracy_df['given_attempted_accuracy']],
        'F1': [accuracy_df['F1']],
    })

    accuracy_df = df.groupby('primary_category').apply(calculate_accuracies).reset_index()
    final_df = pd.concat([overall_row, accuracy_df], ignore_index=True)
    final_df.to_csv(new_file.replace(".jsonl", ".csv"), index=False)