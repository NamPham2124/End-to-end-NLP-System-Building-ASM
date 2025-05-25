# End-to-End NLP System Building

This project implements an end-to-end NLP system that leverages large language models (LLMs) and retrieval-augmented generation (RAG) techniques to generate and evaluate question-answer pairs from text documents. The system uses state-of-the-art transformer models, embeddings, and vector search to perform document understanding and question answering.

---

## Features

- Clone and setup environment with necessary dependencies.
- Load and quantize LLaMA 3.2 3B Instruct model for efficient inference with 4-bit precision.
- Preprocess text data and build embeddings using Sentence Transformers.
- Implement a retrieval-augmented generation pipeline with FAISS vector store.
- Generate question-answer pairs automatically from text files.
- Evaluate generated answers with metrics including Exact Match, F1 Score, and Answer Recall.
- Save results and evaluation reports for further analysis.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/NamPham2124/End-to-end-NLP-System-Building-ASM.git
cd End-to-end-NLP-System-Building-ASM
```

2. Install required Python packages:

```bash
pip install pypdf transformers torch torchvision tqdm accelerate>=0.26.0 \
    python-dotenv sentence-transformers huggingface_hub pandas \
    selenium beautifulsoup4 langchain langchain-community \
    langchain-chroma langchain-openai faiss-cpu scikit-learn \
    langchain_experimental langchain_openai flashrank flashrank[listwise] bitsandbytes
```

3. (Optional) Login to Hugging Face Hub for access to models:

```python
from huggingface_hub import login
login()
```

---

## Usage

### Model Setup

The project uses the `meta-llama/Llama-3.2-3B-Instruct` model with 4-bit quantization for efficient GPU inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Llama-3.2-3B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
)
```

### Run Retrieval-Augmented Generation Pipeline

Run the RAG pipeline script with your desired parameters:

```bash
python pipeline/rag_pipeline_new.py \
--model_name meta-llama/Llama-3.2-3B-Instruct \
--dtype float16 \
--embedding_model_name sentence-transformers/all-MiniLM-L6-v2 \
--embedding_dim 384 \
--splitter_type recursive \
--chunk_size 1000 \
--chunk_overlap 200 \
--text_files_path data/crawled_text_datas \
--top_k_search 3 \
--retriever_type FAISS \
--rerank_model_name ms-marco-MiniLM-L-12-v2 \
--hypo False \
--output_file output/baseline_rag.csv
```

### Evaluate Results

Evaluate the generated QA pairs using the evaluation script:

```bash
python evaluation/evaluate.py --combined_dir output/baseline_rag.csv --output_dir results/baseline_rag.json
```

---

## Directory Structure

```
End-to-end-NLP-System-Building-ASM/
├── data/
│   ├── crawled_text_datas/         # Input text files for processing
│   └── embeddings/                  # Stored embeddings
├── output/                         # Output CSV files of generated QA pairs
├── pipeline/
│   └── rag_pipeline_new.py         # Main RAG pipeline script
├── evaluation/
│   └── evaluate.py                 # Evaluation script for QA results
├── results/                        # Evaluation reports
└── README.md                      # This file
```

---

## Environment Variables

Set your Hugging Face and LangChain API keys before running the scripts:

```python
import os
os.environ["HUGGINGFACE_TOKEN"] = "your_huggingface_token"
os.environ["LANGCHAIN_API_KEY"] = "your_langchain_api_key"
```

---

## Notes

- The system requires a CUDA-enabled GPU for efficient model inference.
- Quantization with bitsandbytes reduces memory usage but requires compatible hardware.
- The project uses LangChain for embeddings and retrieval components.
- Make sure to update the API keys and tokens with your own credentials.

---




