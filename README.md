# Semantic Chunk Merging with BERT NSP

This project explores different strategies for semantic text chunking, primarily leveraging BERT's Next Sentence Prediction (NSP) capabilities and other embedding-based similarity measures. It evaluates these chunking methods on a question-answering retrieval task using the SQuAD v2 dataset.

## Project Structure

- `main.py`: Main script to run the chunking and evaluation experiments.
- `chunking.py`: Contains various text chunking algorithms (Fixed Size, Sentence, BERT Cosine Similarity, BERT NSP, Hybrid).
- `data_processing.py`: Handles loading, preprocessing, and splitting of the SQuAD v2 dataset.
- `evaluation.py`: Includes functions for generating embeddings (SentenceTransformer), FAISS indexing, retrieval, calculating metrics (Recall@k, Precision@k, F1, Semantic Precision), and plotting results.
- `model_loader.py`: Loads pre-trained models (SentenceTransformer, BERT for NSP, BERT for embeddings) and tokenizers from Hugging Face.
- `utils.py`: General utility functions.
- `requirements.txt`: Python package dependencies.
- `Dockerfile`: Defines the Docker image for the project.
- `docker-compose.yml`: Docker Compose configuration to build and run the application.
- `output/`: Directory where output plots are saved (e.g., `chunk_size_comparison_main_test.png`).

## Prerequisites

- Docker Engine
- Docker Compose

## How to Run

1.  **Clone the repository (if applicable) or ensure all project files are in the current directory.**

2.  **Build and run the Docker container using Docker Compose:**

    ```bash
    docker-compose up --build
    ```

    This command will:
    - Build the Docker image based on the `Dockerfile` (installing all dependencies from `requirements.txt`).
    - Run the `main.py` script inside the container.

3.  **View Output:**
    - The script will print progress, threshold calculations, and evaluation metrics for each chunking method to the console.
    - A plot comparing chunk size distributions (`chunk_size_comparison_main_test.png` or similar, depending on `main.py` parameters) will be saved to the `./output` directory on your host machine.

## Notes

- The current Docker setup uses `faiss-cpu` for broader compatibility. If you have an NVIDIA GPU and the NVIDIA Docker runtime configured, you can modify `requirements.txt` to use `faiss-gpu` and uncomment the `deploy` section in `docker-compose.yml` for GPU acceleration.
- The `main.py` script is configured by default to run with a small sample of the dataset (`dataset_sample_size=100`) for a quick test. To run on the full SQuAD v2 validation set, modify the `dataset_sample_size` parameter to `None` within `main.py` before building the Docker image or adjust the `run_experiment_with_evaluation` call.
