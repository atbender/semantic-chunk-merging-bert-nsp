import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForNextSentencePrediction
from typing import List, Dict, Union
import random

def get_next_sentence_score(sentence1: str, sentence2: str, model: BertForNextSentencePrediction, tokenizer: BertTokenizer) -> Dict[str, float]:
    """
    Calculates the Next Sentence Prediction (NSP) scores for two sentences.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.
        model (BertForNextSentencePrediction): Pre-trained BERT NSP model.
        tokenizer (BertTokenizer): Pre-trained BERT tokenizer.

    Returns:
        Dict[str, float]: A dictionary with 'is_next_score' and 'is_not_next_score'.
    """
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    softmax_probs = torch.nn.functional.softmax(logits, dim=1)
    is_next_score = softmax_probs[0][0].item()  # Probability of "IsNext"
    is_not_next_score = softmax_probs[0][1].item()  # Probability of "IsNotNext"

    return {
        "is_next_score": is_next_score,
        "is_not_next_score": is_not_next_score
    }

def analyze_list(data: List[Union[int, float]], list_name: str = "Data", output_path: str = None):
    """
    Analyzes a list of numerical data by computing mean, median, and plotting its distribution.

    Args:
        data (List[Union[int, float]]): The list of numerical data to analyze.
        list_name (str): Name of the data, used in plot title and labels.
        output_path (str, optional): Path to save the plot. If None, shows the plot.
    """
    if not data:
        print(f"No data provided for list: {list_name}. Skipping analysis.")
        return

    mean_val = np.mean(data)
    median_val = np.median(data)
    std_dev = np.std(data)

    print(f"--- Analysis for: {list_name} ---")
    print(f"Count: {len(data)}")
    print(f"Mean: {mean_val:.4f}")
    print(f"Median: {median_val:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {list_name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == '__main__':
    print("--- Testing Utility Functions ---")

    # Test analyze_list
    sample_numeric_data = [random.gauss(10, 2) for _ in range(100)]
    sample_numeric_data.extend([random.gauss(20, 3) for _ in range(50)])
    # analyze_list(sample_numeric_data, "Sample Gaussian Data") # Uncomment to show plot
    print("analyze_list test would run if uncommented.")

    # Test get_next_sentence_score (requires models)
    # print("\n--- Testing get_next_sentence_score (requires models) ---")
    # try:
    #     from model_loader import load_models_and_tokenizer # Assuming model_loader.py is in PYTHONPATH
    #     _, nsp_model, _, tokenizer = load_models_and_tokenizer(nsp_model_name='bert-base-uncased') # Use base for quicker test
        
    #     sentence_a = "The cat sat on the mat."
    #     sentence_b = "It was a fluffy Persian cat."
    #     sentence_c = "Dogs are known for their loyalty."

    #     scores_ab = get_next_sentence_score(sentence_a, sentence_b, nsp_model, tokenizer)
    #     print(f"Scores for A-B (related): {scores_ab}")

    #     scores_ac = get_next_sentence_score(sentence_a, sentence_c, nsp_model, tokenizer)
    #     print(f"Scores for A-C (unrelated): {scores_ac}")

    # except ImportError:
    #     print("Skipping get_next_sentence_score test: model_loader.py not found or PyTorch/Transformers not installed.")
    # except Exception as e:
    #     print(f"Skipping get_next_sentence_score test due to an error: {e}")
    print("get_next_sentence_score test (requires models) would run if uncommented and models are available.") 