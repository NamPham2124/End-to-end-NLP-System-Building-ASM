"""
This script is used to evaluate the performance of the generated answers in the RAG-based QA system.
For evaluation metrics, we use 3 metrics: answer recall, exact match, and F1 score frollowing the setting in 
the SQuAD paper (https://arxiv.org/pdf/1606.05250). 
"""

import re
import pandas as pd
import json
import string
from collections import Counter
import argparse

WHITESPACE_AND_PUNCTUATION = set(string.whitespace + string.punctuation)
ARTICLES = set(['the', 'a', 'an'])


def clean_answer(answer):
    answer = str(answer).lower()
    # Replace Unicode non-breaking space with a regular space
    answer = answer.replace(u'\u00a0', ' ')
    while len(answer) > 1 and answer[0] in WHITESPACE_AND_PUNCTUATION:
        answer = answer[1:]
    while len(answer) > 1 and answer[-1] in WHITESPACE_AND_PUNCTUATION:
        answer = answer[:-1]

    answer = answer.split()
    if len(answer) > 1 and answer[0] in ARTICLES:
        answer = answer[1:]
    answer = ' '.join(answer)

    return answer

def compute_exact_match_single(gold_answer_list, generated_answer):
    """Check if the generated answer exactly matches any of the gold answers."""
    for gold_answer in gold_answer_list:
        if clean_answer(gold_answer) == clean_answer(generated_answer):
            return 1
    return 0

def compute_exact_match(gold_answers, generated_answers):
    """Given two lists of gold and generated answers, compute the exact match score."""
    exact_match = 0
    for gold_answer_list, generated_answer in zip(gold_answers, generated_answers):
        exact_match += compute_exact_match_single(gold_answer_list, generated_answer)
        
    return 100 * exact_match / len(gold_answers)

def compute_recall_f1_single(gold_answer_list, generated_answer):
    """Compute F1 score between the generated answer and the gold answers."""
    
    def GetTokens(text):
            text = clean_answer(text)
            for delimeter in WHITESPACE_AND_PUNCTUATION:
                text = text.replace(delimeter, ' ')
            return text.split()
    
    max_f1 = 0
    max_recall = 0
    predicted_answer_tokens = Counter(GetTokens(generated_answer))
    num_predicted_answer_tokens = sum(predicted_answer_tokens.values())
    
    for answer in gold_answer_list:
        answer_tokens = Counter(GetTokens(answer))
        num_answer_tokens = sum(answer_tokens.values())
        num_same = sum((predicted_answer_tokens & answer_tokens).values())
        
        if num_same == 0:
            continue
        precision = 1.0 * num_same / num_predicted_answer_tokens
        recall = 1.0 * num_same / num_answer_tokens
        max_recall = max(recall, max_recall)
        max_f1 = max(2 * precision * recall / (precision + recall), max_f1)
        
    return max_recall, max_f1

def compute_recall_f1(gold_answers, generated_answers):
    """Given two lists of gold and generated answers, compute the F1 score."""
    total_f1 = 0
    total_recall = 0
    for gold_answer_list, generated_answer in zip(gold_answers, generated_answers):
        recall, f1 = compute_recall_f1_single(gold_answer_list, generated_answer)
        total_f1 += f1
        total_recall += recall
    
    avg_f1 = 100 * total_f1 / len(gold_answers)
    avg_recall = 100 * total_recall / len(gold_answers)
    return avg_recall, avg_f1

def evaluate(gold_answers, generated_answers):
    """Evaluate a list of generated answers against the reference (gold) answers."""
    
    exact_match = compute_exact_match(gold_answers, generated_answers)
    answer_recall, f1_score_avg = compute_recall_f1(gold_answers, generated_answers)
    
    return {
        "Exact Match": exact_match,
        "F1 Score": f1_score_avg,
        "Answer Recall": answer_recall
    }

if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(description='Evaluate the performance of the generated answers.')
    parser.add_argument('--combined_dir', type=str, help='Path to the directory containing the combined gold and generated answers.')
    parser.add_argument('--gold_answer_dir', type=str, help='Path to the directory containing the gold answers.')
    parser.add_argument('--generated_answer_dir', type=str, help='Path to the directory containing the generated answers.')
    parser.add_argument('--output_dir', type=str, help='Path to the directory to save the evaluation results.')
    
    args = parser.parse_args()
    
    if args.combined_dir:
        # read in as csv file
        
        generation_df = pd.read_csv(args.combined_dir)
        generated_answers = generation_df["Generated_Answer"].tolist()
        # each row is a list of gold answers
        # example gold answers: ["William Pitt", "William Pitt the Younger"]
        gold_answers = generation_df["Reference_Answers"].apply(lambda x: str(x).split("[SEP]")).tolist()
        print(gold_answers[:5])
    
    else:
        # read the gold answers, create a list of lists
        # each sublist contains one or more gold answers
        gold_answers = []
        with open(args.gold_answer_dir, 'r') as f:
            for line in f:
                gold_answers.append(line.strip().split(';'))
                
        # read the generated answers, each line contains one generated answer
        generated_answers = []
        with open(args.generated_answer_dir, 'r') as f:
            for line in f:
                generated_answers.append(line.strip())
    
    # # sample gold and generated answers for testing
    # gold_answers = [
    #     ["William Pitt"], 
    #     ["ICML", "International Conference on Machine Learning"], 
    #     ["Billie Eilish"]
    # ]

    # generated_answers = ["William Pitt, the pioneer", "COLM", "Billie"]

    # evaluate the generated answers
    results = evaluate(gold_answers, generated_answers)
    print(results)
    
    # save the evaluation results as json file
    with open(args.output_dir, 'w') as f:
        json.dump(results, f)