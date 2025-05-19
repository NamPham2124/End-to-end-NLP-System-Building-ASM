# This script is used to fine-tune different combinations of RAG strategies on the test set.
# example usage: bash pipeline/run_rag.sh

# strategy 0: default setting
python pipeline/rag_pipeline_new.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --embedding_model_name sentence-transformers/all-MiniLM-L6-v2 \
    --embedding_dim 384 \
    --splitter_type recursive \
    --chunk_size 1000 \
    --chunk_overlap 200 \
    --text_files_path data/crawled/crawled_text_data \
    --top_k_search 3 \
    --retriever_type CHROMA\
    --output_file output/llama3_recursive_chroma_top3.csv \

# strategy 1: recursive splitter, CHROMA retriever, tune chunk size
python pipeline/rag_pipeline_new.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --embedding_model_name sentence-transformers/all-MiniLM-L6-v2 \
    --embedding_dim 384 \
    --splitter_type recursive \
    --chunk_size 1500 \
    --chunk_overlap 200 \
    --text_files_path data/crawled/crawled_text_data \
    --top_k_search 3 \
    --retriever_type CHROMA\
    --output_file output/llama3_recursive_chunk1500_chroma_top3_sample100.csv \

# strategy 2: CHROMA retriever, tune splitter
python pipeline/rag_pipeline_new.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --embedding_model_name sentence-transformers/all-MiniLM-L6-v2 \
    --embedding_dim 384 \
    --splitter_type semantic \
    --chunk_size 1000 \
    --chunk_overlap 200 \
    --text_files_path data/crawled/crawled_text_data \
    --top_k_search 3 \
    --retriever_type CHROMA\
    --output_file output/llama3_semantic_chroma_top3_sample100.csv \

# strategy 3: change retriever: FAISS
python pipeline/rag_pipeline_new.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --embedding_model_name sentence-transformers/all-MiniLM-L6-v2 \
    --embedding_dim 384 \
    --splitter_type recursive \
    --chunk_size 1000 \
    --chunk_overlap 200 \
    --text_files_path data/crawled/crawled_text_data \
    --top_k_search 3 \
    --retriever_type FAISS \
    --output_file output/tests/llama3_faiss_test_reverseprompt.csv \


# strategy 4: tune reranking using faiss
python pipeline/rag_pipeline_new.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --embedding_model_name sentence-transformers/all-MiniLM-L6-v2 \
    --embedding_dim 384 \
    --splitter_type recursive \
    --chunk_size 1000 \
    --chunk_overlap 200 \
    --text_files_path data/crawled/crawled_text_data \
    --qes_file_path data/test/test_questions.csv \
    --top_k_search 10 \
    --top_k_rerank 3 \
    --retriever_type FAISS \
    --rerank True \
    --rerank_model_name ms-marco-MiniLM-L-12-v2 \
    --output_file output/submission/llama3_faiss_rerank.csv \


# strategy 5: add hqde for better retrieval
python pipeline/rag_pipeline_new.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --embedding_model_name sentence-transformers/all-MiniLM-L6-v2 \
    --embedding_dim 384 \
    --splitter_type recursive \
    --chunk_size 1000 \
    --chunk_overlap 200 \
    --text_files_path data/crawled/crawled_text_data \
    --top_k_search 3 \
    --retriever_type FAISS \
    --rerank_model_name ms-marco-MiniLM-L-12-v2 \
    --hypo True \
    --output_file output/llama3_faiss_test_hypo_promptENG3.csv \