# Semantic Chunk Merging with BERT NSP

**Link to Paper:** [Next Sentence Prediction with BERT as a Dynamic Chunking Mechanism for Retrieval-Augmented Generation Systems](https://journals.flvc.org/FLAIRS/article/view/138940)

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

## How to Cite

If you use this project or find the research helpful, please consider citing the following paper:

```bibtex
@article{Bender_Almeida_Gomes_Brisolara_Corrêa_Matsumura_Araujo_2025,
  title={Next Sentence Prediction with BERT as a Dynamic Chunking Mechanism for Retrieval-Augmented Generation Systems},
  volume={38},
  url={https://journals.flvc.org/FLAIRS/article/view/138940},
  DOI={10.32473/flairs.38.1.138940},
  abstractNote={&lt;p&gt;Retrieval-Augmented Generation systems enhance the generative capabilities of large language models by grounding their responses in external knowledge bases, addressing some of their major limitations and improving their reliability for tasks requiring factual accuracy or domain-specific information. Chunking is a critical step in Retrieval-Augmented Generation pipelines, where text is divided into smaller segments to facilitate efficient retrieval and optimize the use of model context. This paper introduces a method that uses BERT&amp;#039;s Next Sentence Prediction to adaptively merge related sentences into context-aware chunks. We evaluate the approach on the SQuAD v2 dataset, comparing it to standard chunking methods using Recall@k, Precision@k, Contextual-Precision@k, and processing time as metrics. Results indicate that the proposed method achieves competitive retrieval performance while reducing computational time by roughly 60%, demonstrating its potential to improve Retrieval-Augmented Generation systems.&lt;/p&gt;},
  number={1},
  journal={The International FLAIRS Conference Proceedings},
  author={Bender, Alexandre Thurow and Almeida Gomes, Gabriel and Brisolara Corrêa, Ulisses and Matsumura Araujo, Ricardo},
  year={2025},
  month={May}
}
```
