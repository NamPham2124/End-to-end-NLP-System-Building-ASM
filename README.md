# End-to-End Document Intelligence & Quantized RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system utilizing **4-bit quantized LLaMA 3.2 (3B Instruct)**, dense vector search with **FAISS**, cross-encoder reranking, and systematic metric-driven evaluation for domain document understanding.

---

## 📌 Architecture & Pipeline

The pipeline transforms raw document corpuses into high-accuracy Question-Answering intelligence through a multi-stage workflow:

```
[Raw Documents / Web Text]
           │
           ▼
[Recursive Text Splitter] ── (Chunk Size: 1000, Overlap: 200)
           │
           ▼
[Embedding Model] ────────── (sentence-transformers/all-MiniLM-L6-v2)
           │
           ▼
[FAISS Vector Index] ─────── Top-K Dense Vector Retrieval
           │
           ▼
[Cross-Encoder Reranker] ── (ms-marco-MiniLM-L-12-v2)
           │
           ▼
[LLaMA 3.2 (4-bit NF4)] ──── Context-Augmented Response Generation
           │
           ▼
[Evaluation Harness] ─────── Exact Match (EM), F1 Score, Answer Recall
```

---

## 🛠️ Key Technical Features

- **4-Bit NF4 NormalFloat Quantization:** Utilizes `bitsandbytes` to load and run LLaMA 3.2 3B Instruct under tight VRAM constraints (< 4GB GPU memory) without catastrophic degradation of generative coherence.
- **Two-Stage Retrieval:** Combines sub-millisecond FAISS dense index retrieval with neural cross-encoder reranking to prioritize semantically precise passages.
- **Quantitative Evaluation Suite:** Integrated scoring module evaluating generated answers against ground-truth benchmarks on **Exact Match**, **Token F1**, and **Recall**.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/NamPham2124/End-to-end-NLP-System-Building-ASM.git
cd End-to-end-NLP-System-Building-ASM

pip install -r requirements.txt # or install via pip
pip install torch transformers sentence-transformers faiss-cpu langchain langchain-community bitsandbytes accelerate
```

### 2. Run Retrieval-Augmented Generation Pipeline

```bash
python pipeline/rag_pipeline_new.py   --model_name meta-llama/Llama-3.2-3B-Instruct   --dtype float16   --embedding_model_name sentence-transformers/all-MiniLM-L6-v2   --embedding_dim 384   --splitter_type recursive   --chunk_size 1000   --chunk_overlap 200   --text_files_path data/crawled_text_datas   --top_k_search 3   --retriever_type FAISS   --rerank_model_name ms-marco-MiniLM-L-12-v2   --output_file output/baseline_rag.csv
```

### 3. Quantitative Evaluation

```bash
python evaluation/evaluate.py   --combined_dir output/baseline_rag.csv   --output_dir results/baseline_rag.json
```

---

## 📂 Project Structure

```
├── data/                       # Raw input text data and cached embeddings
├── output/                     # Generated QA pairs and intermediate logs
├── pipeline/
│   └── rag_pipeline_new.py     # Main end-to-end RAG orchestrator
├── evaluation/
│   └── evaluate.py             # Evaluation harness (EM, F1, Recall)
├── results/                    # Serialized JSON evaluation benchmark reports
├── run&evaluate_rag.ipynb      # Interactive experimentation notebook
└── README.md
```

---

## 📜 License

Distributed under the MIT License.
