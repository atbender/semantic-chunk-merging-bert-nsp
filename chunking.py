import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction
from sentence_transformers import SentenceTransformer
from typing import List, Union

def chunk_text(text: str, method: str = "Sentence", chunk_size: int = 30) -> List[str]:
    if not text.strip():
        return []
    if method == "Sentence":
        return [s.strip() for s in text.split(". ") if s.strip()]
    elif method == "Fixed Size":
        words = text.split()
        if not words:
            return []
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    else:
        raise ValueError(f"Unknown chunking method: {method}. Choose 'Sentence' or 'Fixed Size'.")

def chunk_text_with_cosine_similarity(
    text: str, 
    bert_tokenizer: BertTokenizer, 
    bert_embedding_model: BertModel, 
    threshold: float = 0.85, 
    max_length: int = 512
) -> List[str]:
    if not text.strip():
        return []
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if not sentences:
        return []

    sentence_embeddings = []
    for sentence in sentences:
        inputs = bert_tokenizer(sentence, return_tensors='pt', truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = bert_embedding_model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
            sentence_embeddings.append(embedding)
    
    if not sentence_embeddings:
        return []

    chunks = []
    current_chunk = sentences[0]
    current_embedding = np.array(sentence_embeddings[0]) 

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        sentence_embedding = np.array(sentence_embeddings[i])

        if current_embedding.ndim > 1:
            current_embedding = current_embedding.squeeze()
        if sentence_embedding.ndim > 1:
            sentence_embedding = sentence_embedding.squeeze()

        norm_current = np.linalg.norm(current_embedding)
        norm_sentence = np.linalg.norm(sentence_embedding)

        if norm_current == 0 or norm_sentence == 0:
            similarity = 0.0
        else:
            similarity = np.dot(current_embedding, sentence_embedding) / (norm_current * norm_sentence)

        if similarity >= threshold:
            current_chunk += ". " + sentence 
            inputs = bert_tokenizer(current_chunk, return_tensors='pt', truncation=True, max_length=max_length)
            with torch.no_grad():
                outputs = bert_embedding_model(**inputs, output_hidden_states=True)
                current_embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
            current_embedding = sentence_embedding

    chunks.append(current_chunk)
    return chunks

def chunk_text_with_next_sentence_prediction(
    text: str, 
    bert_tokenizer: BertTokenizer, 
    bert_nsp_model: BertForNextSentencePrediction, 
    threshold: float = 0.99,
    max_length: int = 512
) -> List[str]:
    if not text.strip():
        return []
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if not sentences:
        return []

    chunks = []
    current_chunk = sentences[0]

    for i in range(1, len(sentences)):
        sentence_1 = current_chunk
        sentence_2 = sentences[i]

        inputs = bert_tokenizer(sentence_1, sentence_2, return_tensors='pt', truncation='longest_first', max_length=max_length)

        with torch.no_grad():
            logits = bert_nsp_model(**inputs).logits
        
        prob_next_sentence = torch.nn.functional.softmax(logits, dim=-1)[0, 0].item()

        if prob_next_sentence >= threshold:
            current_chunk += ". " + sentence_2
        else:
            chunks.append(current_chunk)
            current_chunk = sentence_2

    chunks.append(current_chunk)
    return chunks

def hybrid_chunking(
    text: str, 
    retriever_model: SentenceTransformer, 
    bert_tokenizer: BertTokenizer, 
    bert_nsp_model: BertForNextSentencePrediction, 
    cosine_threshold: float = 0.85, 
    nsp_threshold: float = 0.99,
    max_nsp_length: int = 512
) -> List[str]:
    if not text.strip():
        return []
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if not sentences:
        return []

    sentence_embeddings = retriever_model.encode(sentences)

    chunks = []
    current_chunk = sentences[0]
    current_embedding = sentence_embeddings[0]

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        next_sentence_embedding = sentence_embeddings[i]

        norm_current = np.linalg.norm(current_embedding)
        norm_next = np.linalg.norm(next_sentence_embedding)
        if norm_current == 0 or norm_next == 0:
            similarity = 0.0
        else:
            similarity = np.dot(current_embedding, next_sentence_embedding) / (norm_current * norm_next)

        inputs = bert_tokenizer(current_chunk, sentence, return_tensors='pt', truncation='longest_first', max_length=max_nsp_length)
        with torch.no_grad():
            logits = bert_nsp_model(**inputs).logits
        prob_next_sentence = torch.nn.functional.softmax(logits, dim=-1)[0, 0].item()

        if prob_next_sentence >= nsp_threshold and similarity >= cosine_threshold:
            current_chunk += ". " + sentence
            current_embedding = retriever_model.encode([current_chunk])[0]
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
            current_embedding = next_sentence_embedding

    chunks.append(current_chunk)
    return chunks

if __name__ == '__main__':
    sample_text = "This is the first sentence. This is the second sentence, closely related to the first. This is a third sentence, somewhat related. A fourth sentence, which is quite different. And a fifth one, also different."

    print("--- Testing chunk_text ---")
    fixed_chunks = chunk_text(sample_text, method="Fixed Size", chunk_size=7)
    print(f"Fixed Size Chunks: {fixed_chunks}")
    sentence_chunks = chunk_text(sample_text, method="Sentence")
    print(f"Sentence Chunks: {sentence_chunks}")

    print("\n(To run full tests for chunking.py, uncomment model loading and specific chunker tests)") 