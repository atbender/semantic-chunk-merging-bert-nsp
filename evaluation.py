import faiss
import numpy as np
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Any

def create_embeddings(chunks: List[str], retriever_model: SentenceTransformer, convert_to_tensor: bool = True, truncation: bool = True) -> Union[torch.Tensor, np.ndarray]:
    if not chunks:
        # SentenceTransformer handles empty list input, typically returning empty ndarray.
        pass 
    return retriever_model.encode(chunks, convert_to_tensor=convert_to_tensor, truncation=truncation)

def setup_faiss(embeddings: Union[np.ndarray, torch.Tensor]) -> faiss.IndexFlatL2:
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings
    
    if embeddings_np.ndim == 1:
        embeddings_np = np.expand_dims(embeddings_np, axis=0)
    
    if embeddings_np.shape[0] == 0:
        raise ValueError("Cannot setup FAISS index with empty embeddings.")

    d = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    return index

def retrieve(query: str, index: faiss.IndexFlatL2, chunks: List[str], retriever_model: SentenceTransformer, top_k: int = 3) -> List[str]:
    if not query.strip() or index.ntotal == 0:
        return []
    query_embedding = retriever_model.encode([query], convert_to_tensor=False, truncation=True)
    
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
    distances, indices = index.search(query_embedding, min(top_k, index.ntotal))
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]

def calculate_recall_k(retrieved_chunks: List[str], ground_truth: str, k: int = 3) -> int:
    if not ground_truth.strip(): return 0
    for chunk in retrieved_chunks[:k]:
        if ground_truth in chunk:
            return 1
    return 0

def calculate_precision_k_jaccard(retrieved_chunks: List[str], ground_truth: str, k: int = 3) -> float:
    if not ground_truth.strip() or not retrieved_chunks:
        return 0.0
        
    precision_scores = []
    ground_truth_words = set(ground_truth.lower().split())
    if not ground_truth_words: return 0.0

    for chunk in retrieved_chunks[:k]:
        chunk_words = set(chunk.lower().split())
        if not chunk_words: 
            precision_scores.append(0.0)
            continue
            
        intersection = len(chunk_words.intersection(ground_truth_words))
        union = len(chunk_words.union(ground_truth_words))

        if union == 0:
            precision_scores.append(0.0)
        else:
            precision_scores.append(intersection / union)
    
    return np.mean(precision_scores) if precision_scores else 0.0

def calculate_semantic_precision_k(retrieved_chunks: List[str], ground_truth: str, retriever_model: SentenceTransformer, k: int = 3) -> float:
    if not ground_truth.strip() or not retrieved_chunks:
        return 0.0

    relevant_scores = []
    ground_truth_embedding = retriever_model.encode([ground_truth], convert_to_tensor=True, truncation=True)

    for chunk in retrieved_chunks[:k]:
        if not chunk.strip():
            relevant_scores.append(0.0)
            continue
        chunk_embedding = retriever_model.encode([chunk], convert_to_tensor=True, truncation=True)
        
        similarity_score = torch.nn.functional.cosine_similarity(chunk_embedding, ground_truth_embedding, dim=1)
        relevant_scores.append(similarity_score.item())

    return np.mean(relevant_scores) if relevant_scores else 0.0

def plot_chunk_sizes(chunk_sizes_by_method: Dict[str, List[int]], output_path: str = None):
    if not chunk_sizes_by_method:
        print("No chunk sizes to plot.")
        return

    num_methods = len(chunk_sizes_by_method)
    fig, axes = plt.subplots(2, num_methods, figsize=(5 * num_methods, 10), squeeze=False)

    for idx, (method, chunk_sizes) in enumerate(chunk_sizes_by_method.items()):
        if not chunk_sizes:
            print(f"No chunk sizes available for method: {method}. Skipping plot.")
            ax_raw = axes[0, idx]
            ax_raw.set_title(f'Raw Chunk Sizes for {method}\n(No data)')
            ax_raw.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax_raw.transAxes)
            ax_dist = axes[1, idx]
            ax_dist.set_title(f'Chunk Size Distribution for {method}\n(No data)')
            ax_dist.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax_dist.transAxes)
            continue

        ax_raw = axes[0, idx]
        ax_raw.bar(range(len(chunk_sizes)), chunk_sizes, color='skyblue', edgecolor='black')
        ax_raw.set_title(f'Raw Chunk Sizes for {method}')
        ax_raw.set_xlabel('Chunk Index')
        ax_raw.set_ylabel('Chunk Size (Number of Words)')

        ax_dist = axes[1, idx]
        ax_dist.hist(chunk_sizes, bins=30, alpha=0.75, color='lightgreen', edgecolor='black')
        ax_dist.set_title(f'Chunk Size Distribution for {method}')
        ax_dist.set_xlabel('Chunk Size (Number of Words)')
        ax_dist.set_ylabel('Frequency')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == '__main__':
    print("--- Testing Evaluation Utilities ---")

    sample_chunks = [
        "This is the first chunk of text.", 
        "Another chunk follows, with different content.", 
        "The ground truth might be in this one.",
        "Final chunk here."
    ]
    sample_ground_truth = "ground truth might be"
    
    class MockRetrieverModel:
        def encode(self, texts, convert_to_tensor=True, truncation=True):
            dim = 384 
            embs = [np.array([hash(text + str(i)) % 1000 / 1000.0 for i in range(dim)], dtype=np.float32) for text in texts]
            embs = np.array(embs)
            return torch.tensor(embs) if convert_to_tensor else embs

    mock_retriever = MockRetrieverModel()

    if sample_chunks:
        embeddings = create_embeddings(sample_chunks, mock_retriever, convert_to_tensor=False)
        print(f"Created embeddings shape: {embeddings.shape if embeddings is not None else 'None'}")

        if embeddings is not None and embeddings.shape[0] > 0:
            faiss_index = setup_faiss(embeddings)
            print(f"FAISS index created with {faiss_index.ntotal} embeddings.")

            retrieved = retrieve("search for truth", faiss_index, sample_chunks, mock_retriever, top_k=2)
            print(f"Retrieved chunks: {retrieved}")

            recall = calculate_recall_k(retrieved, sample_ground_truth, k=2)
            print(f"Recall@2: {recall}")

            precision_jaccard = calculate_precision_k_jaccard(retrieved, sample_ground_truth, k=2)
            print(f"Jaccard Precision@2: {precision_jaccard:.4f}")

            semantic_precision = calculate_semantic_precision_k(retrieved, sample_ground_truth, mock_retriever, k=2)
            print(f"Semantic Precision@2: {semantic_precision:.4f}")
        else:
            print("Skipping FAISS and retrieval tests due to no embeddings.")
    else:
        print("Skipping embedding and retrieval tests due to no sample chunks.")

    chunk_sizes_data = {
        "Method A": [10, 12, 15, 11, 13, 25, 20, 22],
        "Method B": [8, 9, 7, 10, 20, 18, 22, 25, 23],
        "NoDataMethod": []
    }
    print("Plotting test would run if uncommented.") 