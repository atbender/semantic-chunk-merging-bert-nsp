import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from model_loader import load_models_and_tokenizer
from data_processing import load_data
from chunking import (
    chunk_text, 
    chunk_text_with_cosine_similarity, 
    chunk_text_with_next_sentence_prediction,
    hybrid_chunking
)
from evaluation import (
    create_embeddings,
    setup_faiss,
    retrieve,
    calculate_recall_k,
    calculate_precision_k_jaccard,
    calculate_semantic_precision_k,
    plot_chunk_sizes
)

def run_experiment_with_evaluation(
    dataset_name: str = "squad_v2",
    dataset_sample_size: Optional[int] = None,
    train_split_ratio: float = 0.3,
    k_retrieval: int = 3,
    seed: int = 42,
    plot_output_path: Optional[str] = "output/chunk_size_comparison.png"
):
    print("Starting experiment...")
    print(f"Seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("\n--- Loading Models ---")
    retriever_model, bert_nsp_model, bert_embedding_model, bert_tokenizer = load_models_and_tokenizer()

    print("\n--- Loading Data ---")
    all_data = load_data(dataset_name=dataset_name, sample_size=dataset_sample_size, seed=seed)
    if not all_data:
        print("No data loaded. Exiting experiment.")
        return

    np.random.shuffle(all_data)
    train_size = int(len(all_data) * train_split_ratio)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    if not train_data or not test_data:
        print("Not enough data for train/test split. Exiting experiment.")
        return

    print(f"Training set size (for thresholds): {len(train_data)}")
    print(f"Test set size (for evaluation): {len(test_data)}")

    print("\n--- Calculating Thresholds and Average Context Length from Training Data ---")
    avg_context_length_words = int(np.mean([len(context.split()) for context, _, _ in train_data]))
    avg_fixed_chunk_size = max(1, round(avg_context_length_words / 2))
    print(f"Average fixed chunk size to be used: {avg_fixed_chunk_size} words")

    cosine_similarities = []
    nsp_probabilities = []

    print("Processing training data for threshold calculation...")
    for context, _, _ in tqdm(train_data, desc="Training Data Processing"):
        sentences = [s.strip() for s in context.split(". ") if s.strip()]
        if len(sentences) < 2:
            continue

        sent_embeddings = retriever_model.encode(sentences)
        for i in range(len(sentences) - 1):
            emb1 = sent_embeddings[i]
            emb2 = sent_embeddings[i+1]
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 > 0 and norm2 > 0:
                cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
                cosine_similarities.append(cosine_sim)
            
            inputs = bert_tokenizer(sentences[i], sentences[i+1], return_tensors='pt', truncation='longest_first', max_length=512)
            with torch.no_grad():
                logits = bert_nsp_model(**inputs).logits
            nsp_prob = torch.nn.functional.softmax(logits, dim=-1)[0, 0].item()
            nsp_probabilities.append(nsp_prob)

    cosine_threshold = np.mean(cosine_similarities) if cosine_similarities else 0.85
    nsp_threshold = np.mean(nsp_probabilities) if nsp_probabilities else 0.99
    print(f"Calculated Cosine Similarity Threshold: {cosine_threshold:.4f}")
    print(f"Calculated NSP Threshold: {nsp_threshold:.4f}")

    methods_to_evaluate = ["Fixed Size", "Sentence", "Bert Cosine", "Bert NSP"]

    chunk_sizes_by_method: Dict[str, List[int]] = {method: [] for method in methods_to_evaluate}
    recall_by_method: Dict[str, float] = {method: 0.0 for method in methods_to_evaluate}
    precision_by_method: Dict[str, float] = {method: 0.0 for method in methods_to_evaluate}
    semantic_precision_by_method: Dict[str, float] = {method: 0.0 for method in methods_to_evaluate}
    f1_by_method: Dict[str, float] = {method: 0.0 for method in methods_to_evaluate}

    print("\n--- Running Evaluation on Test Data ---")
    for method_name in methods_to_evaluate:
        print(f"\nEvaluating Chunking Method: {method_name}")
        method_recall_sum = 0.0
        method_precision_sum = 0.0
        method_semantic_precision_sum = 0.0
        method_f1_sum = 0.0
        num_samples_processed = 0

        for context, question, ground_truth in tqdm(test_data, desc=f"Evaluating {method_name}"):
            chunks: List[str] = []
            if method_name == "Fixed Size":
                chunks = chunk_text(context, method="Fixed Size", chunk_size=avg_fixed_chunk_size)
            elif method_name == "Sentence":
                chunks = chunk_text(context, method="Sentence")
            elif method_name == "Bert Cosine":
                chunks = chunk_text_with_cosine_similarity(context, bert_tokenizer, bert_embedding_model, threshold=cosine_threshold)
            elif method_name == "Bert NSP":
                chunks = chunk_text_with_next_sentence_prediction(context, bert_tokenizer, bert_nsp_model, threshold=nsp_threshold)
            elif method_name == "Hybrid":
                 chunks = hybrid_chunking(context, retriever_model, bert_tokenizer, bert_nsp_model, cosine_threshold, nsp_threshold)
            
            if not chunks:
                continue

            chunk_sizes_by_method[method_name].extend([len(c.split()) for c in chunks])
            
            chunk_embeddings = create_embeddings(chunks, retriever_model, convert_to_tensor=False)
            if chunk_embeddings is None or chunk_embeddings.shape[0] == 0:
                continue

            try:
                faiss_index = setup_faiss(chunk_embeddings)
            except ValueError as e:
                continue

            retrieved_chunks = retrieve(question, faiss_index, chunks, retriever_model, top_k=k_retrieval)

            recall = calculate_recall_k(retrieved_chunks, ground_truth, k=k_retrieval)
            precision = calculate_precision_k_jaccard(retrieved_chunks, ground_truth, k=k_retrieval)
            semantic_precision = calculate_semantic_precision_k(retrieved_chunks, ground_truth, retriever_model, k=k_retrieval)
            
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            method_recall_sum += recall
            method_precision_sum += precision
            method_semantic_precision_sum += semantic_precision
            method_f1_sum += f1_score
            num_samples_processed += 1

        if num_samples_processed > 0:
            recall_by_method[method_name] = method_recall_sum / num_samples_processed
            precision_by_method[method_name] = method_precision_sum / num_samples_processed
            semantic_precision_by_method[method_name] = method_semantic_precision_sum / num_samples_processed
            f1_by_method[method_name] = method_f1_sum / num_samples_processed
        else:
            print(f"No samples successfully processed for method: {method_name}")

        print(f"  Results for {method_name.replace('_', ' ').title()}:")
        print(f"    Average Recall@{k_retrieval}: {recall_by_method[method_name]:.4f}")
        print(f"    Average Jaccard Precision@{k_retrieval}: {precision_by_method[method_name]:.4f}")
        print(f"    Average Semantic Precision@{k_retrieval}: {semantic_precision_by_method[method_name]:.4f}")
        print(f"    Average F1-Score@{k_retrieval}: {f1_by_method[method_name]:.4f}")
        print(f"    Total Samples Processed: {num_samples_processed}")

    print("\n--- Final Consolidated Results ---")
    for method_name in methods_to_evaluate:
        print(f"{method_name.replace('_', ' ').title()} Method:")
        print(f"  Average Recall@{k_retrieval}: {recall_by_method[method_name]:.4f}")
        print(f"  Average Jaccard Precision@{k_retrieval}: {precision_by_method[method_name]:.4f}")
        print(f"  Average Semantic Precision@{k_retrieval}: {semantic_precision_by_method[method_name]:.4f}")
        print(f"  Average F1-Score@{k_retrieval}: {f1_by_method[method_name]:.4f}")

    print("\n--- Plotting Chunk Sizes ---")
    plot_chunk_sizes(chunk_sizes_by_method, output_path=plot_output_path)

    print("\nExperiment finished.")

if __name__ == "__main__":
    run_experiment_with_evaluation(
        dataset_sample_size=100,
        plot_output_path="output/chunk_size_comparison_main_test.png"
    ) 