from transformers import BertTokenizer, BertForNextSentencePrediction, BertModel
from sentence_transformers import SentenceTransformer

def load_models_and_tokenizer(
    retriever_model_name: str = 'all-MiniLM-L6-v2',
    nsp_model_name: str = 'bert-large-uncased',
    embedding_model_name: str = 'bert-large-uncased'
):
    print(f"Loading retriever model: {retriever_model_name}")
    retriever_model = SentenceTransformer(retriever_model_name)
    
    print(f"Loading BERT NSP model: {nsp_model_name}")
    bert_nsp_model = BertForNextSentencePrediction.from_pretrained(nsp_model_name)
    
    print(f"Loading BERT embedding model: {embedding_model_name}")
    bert_embedding_model = BertModel.from_pretrained(embedding_model_name)
    
    print(f"Loading BERT tokenizer: {embedding_model_name}") # Tokenizer usually matches the BERT model
    bert_tokenizer = BertTokenizer.from_pretrained(embedding_model_name)
    
    return retriever_model, bert_nsp_model, bert_embedding_model, bert_tokenizer

if __name__ == '__main__':
    ret_model, nsp_model, emb_model, tokenizer = load_models_and_tokenizer()
    print("Models and tokenizer loaded successfully.")
    print(f"Retriever: {type(ret_model)}")
    print(f"NSP Model: {type(nsp_model)}")
    print(f"Embedding Model: {type(emb_model)}")
    print(f"Tokenizer: {type(tokenizer)}") 