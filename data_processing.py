from datasets import load_dataset
from sklearn.model_selection import KFold
import numpy as np
import random
from typing import List, Tuple, Optional, Any

def analyze_squad_v2(sample_size: Optional[int] = None) -> List[Tuple[str, str, str]]:
    dataset = load_dataset("squad_v2", split="validation")

    total_questions = len(dataset)
    answerable_count = 0
    unanswerable_count = 0

    for entry in dataset:
        if entry['answers']['text']:
            answerable_count += 1
        else:
            unanswerable_count += 1

    answerable_percentage = (answerable_count / total_questions) * 100 if total_questions > 0 else 0
    unanswerable_percentage = (unanswerable_count / total_questions) * 100 if total_questions > 0 else 0

    print(f"Dataset Name: SQuAD v2")
    print(f"Total Questions: {total_questions}")
    print(f"Answerable Questions: {answerable_count} ({answerable_percentage:.2f}%)")
    print(f"Unanswerable Questions: {unanswerable_count} ({unanswerable_percentage:.2f}%)")

    answerable_data = [
        (entry['context'], entry['question'], entry['answers']['text'][0])
        for entry in dataset if entry['answers']['text']
    ]

    if sample_size is None or sample_size > len(answerable_data):
        sample_size = len(answerable_data)

    sample_to_return = random.sample(answerable_data, sample_size) if sample_size > 0 else []

    print(f"\nSample size returned: {len(sample_to_return)} (only answerable questions included)")
    return sample_to_return

def kfold_splits(data: List[Any], n_splits: int = 5, seed: int = 42) -> List[Tuple[List[Any], List[Any]]]:
    if not data:
        print("Warning: Empty data provided to kfold_splits. Returning empty list.")
        return []
        
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []

    for fold, (train_index, test_index) in enumerate(kf.split(data), start=1):
        train_part = [data[i] for i in train_index]
        # test_part = [data[i] for i in test_index] # Typical test set for the fold

        # Sample from the train_part, 10% as in original script
        sample_size_for_fold = max(1, len(train_part) // 10) if len(train_part) > 0 else 0
        
        if sample_size_for_fold > 0:
            sample_from_train_part = random.sample(train_part, sample_size_for_fold)
        else:
            sample_from_train_part = []
            
        splits.append((sample_from_train_part, train_part))

        print(f"\nFold {fold}:")
        print(f"  Sample from train part size: {len(sample_from_train_part)}")
        print(f"  Train part size: {len(train_part)}")

    return splits

def load_data(
    dataset_name: str = "squad_v2", 
    split_type: str = 'validation', 
    sample_size: Optional[int] = None, 
    seed: int = 42
) -> List[Tuple[str, str, str]]:
    print(f"Loading dataset: {dataset_name}, split: {split_type}")
    dataset = load_dataset(dataset_name, split=split_type)
    
    cleaned_dataset = dataset.filter(lambda x: x['answers']['text'] and len(x['answers']['text']) > 0)
    print(f"Original size of {split_type} split: {len(dataset)}")
    print(f"Size after filtering for questions with answers: {len(cleaned_dataset)}")

    np.random.seed(seed)

    num_available_samples = len(cleaned_dataset)
    if sample_size is None:
        actual_sample_size = num_available_samples
    else:
        actual_sample_size = min(sample_size, num_available_samples)

    if actual_sample_size == 0:
        print("No data available after filtering and sampling. Returning empty list.")
        return []

    sampled_indices = np.random.choice(num_available_samples, actual_sample_size, replace=False)
    sampled_dataset = cleaned_dataset.select(sampled_indices)

    data = []
    for entry in sampled_dataset:
        context = entry['context']
        question = entry['question']
        ground_truth = entry['answers']['text'][0] if entry['answers']['text'] else ""
        data.append((context, question, ground_truth))

    print(f"Dataset size after cleaning and sampling: {len(data)}")
    return data

if __name__ == '__main__':
    print("--- Testing SQuAD v2 Analysis ---")
    squad_sample = analyze_squad_v2(sample_size=5)

    print("\n--- Testing K-Fold Splits ---")
    dummy_data = [(f"context_{i}", f"question_{i}", f"answer_{i}") for i in range(20)]
    if dummy_data:
        splits = kfold_splits(dummy_data, n_splits=3)
        if splits:
            print(f"\nFirst fold sample size: {len(splits[0][0])}")
            print(f"First fold train part size: {len(splits[0][1])}")
    else:
        print("Skipping kfold_splits test due to no dummy data.")

    print("\n--- Testing Data Loading ---")
    loaded_data_sample = load_data(sample_size=5, seed=123)
    print(f"Loaded {len(loaded_data_sample)} samples.") 