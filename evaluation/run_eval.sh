# # This bash file is used to evaluate the performance of QA systems on the test set.
# # example usage: bash evaluation/run_eval.sh

# # baseline performance
python evaluation/evaluate.py --combined_dir output/llama3_baseline.csv --output_dir results/llama3_baseline.json
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chroma_top3.csv --output_dir results/llama3_recursive_chroma_top3.json

# # for hyperparameter tuning on chunk size
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chunk2000_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chunk2000_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chunk500_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chunk500_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chunk750_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chunk750_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chunk1500_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chunk1500_chroma_top3_sample100.json

# # for hyperparameter tuning on splitter
python evaluation/evaluate.py --combined_dir output/llama3_character_chroma_top3_sample100.csv --output_dir results/llama3_character_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_tokensplit_chroma_top3_sample100.csv --output_dir results/llama3_tokensplit_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_semantic_chroma_top3_sample100.csv --output_dir results/llama3_semantic_chroma_top3_sample100.json

# # for tuning reranking using faiss
python evaluation/evaluate.py --combined_dir output/llama3_faiss_test.csv --output_dir results/llama3_faiss_test.json
python evaluation/evaluate.py --combined_dir output/llama3_faiss_test_rerank.csv --output_dir results/llama3_faiss_test_rerank.json
python evaluation/evaluate.py --combined_dir output/llama3_faiss_test_rerank_t5.csv --output_dir results/llama3_faiss_test_rerank_t5.json
python evaluation/evaluate.py --combined_dir output/llama3_faiss_test_rerank_MiniLM.csv --output_dir results/llama3_faiss_test_rerank_MiniLM.json

# for tuning hypo_doc retrieval
python evaluation/evaluate.py --combined_dir output/llama3_faiss_test_hypo.csv --output_dir results/llama3_faiss_test_hypo.json
python evaluation/evaluate.py --combined_dir output/llama3_faiss_test_hypo_promptENG.csv --output_dir results/llama3_faiss_test_hypo_promptENG.json
python evaluation/evaluate.py --combined_dir output/llama3_faiss_test_hypo_promptENG2.csv --output_dir results/llama3_faiss_test_hypo_promptENG2.json
python evaluation/evaluate.py --combined_dir output/llama3_faiss_test_hypo_promptENG3.csv --output_dir results/llama3_faiss_test_hypo_promptENG3.json

# for running evaluation on the full 3900 test set
python evaluation/evaluate.py --combined_dir output/qa3000/llama3_faiss_rerank.csv --output_dir results/qa3000/llama3_faiss_rerank.json
python evaluation/evaluate.py --combined_dir output/qa3000/llama3_faiss_rerank_sublink.csv --output_dir results/qa3000/llama3_faiss_rerank_sublink.json
