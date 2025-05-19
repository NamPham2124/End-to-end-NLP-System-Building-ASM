import os
import torch
import pandas as pd
from tqdm import tqdm

import faiss
import numpy as np
import argparse
import pickle, random

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, 
    CharacterTextSplitter, 
    TokenTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    PromptTemplate
)
# from sentence_transformers import SentenceTransformer

from utility.rag_utility import (
    load_text_files,  
    answer_generation, 
    PROMPT_TEMPLATE
)

# ========================================
# Vars that can be set and read from another var file.
# ========================================

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Script for running RAG pipeline with FAISS or CHROMA.")

    # Add arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Name of the Hugging Face model to use.")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Precision type (float16 or bfloat16).")
    parser.add_argument("--embedding_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Name of the embedding model to use.")
    parser.add_argument("--embedding_dim", type=int, default=384, help="Dimension of the embeddings.")
    parser.add_argument("--splitter_type", type=str, choices=["recursive", "character", "token", "semantic"], default="recursive",
                        help="Type of text splitter to use.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of the text chunks.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between text chunks.")
    parser.add_argument("--text_files_path", type=str, default="data/crawled/crawled_text_data",
                        help="Path to the text files directory.")
    parser.add_argument("--sublink_files_path", type=str, default="data/crawled/crawled_all",
                        help="Path to the sublink files directory.")
    parser.add_argument("--sublink_files_nums", type=int, default=0,
                        help="number of sublink file to be loaded.")
    parser.add_argument("--retriever_type", type=str, choices=["FAISS", "CHROMA"], default="FAISS",
                        help="Type of retriever to use (FAISS or CHROMA).")
    parser.add_argument("--retriever_algorithm", type=str, choices=["similarity", "mmr"], default="similarity")
    parser.add_argument("--rerank", type=str2bool, default=False, help="Whether to rerank the documents.")
    parser.add_argument("--rerank_model_name", type=str, default="ms-marco-MultiBERT-L-12", help="Name of the rerank model to use.")
    parser.add_argument("--top_k_search", type=int, default=3, help="Top K documents to retrieve.")
    parser.add_argument("--top_k_rerank", type=int, default=3, help="Top K documents to rerank.")
    parser.add_argument("--hypo", type=str2bool, default=False, help="Whether to use hypothetical queries.")
    parser.add_argument("--qes_file_path", type=str, default="data/annotated/QA_pairs_1.csv",
                        help="Path to the QA file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--qa_nums", type=int, default=100)
    return parser.parse_args()

# ========================================
# Main Script Execution
# ========================================
if __name__ == "__main__":

    # Step 0: Load environment variables
    load_dotenv()

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "rag-template"
    os.environ["USER_AGENT"] = "LangChain/1.0 (+https://www.langchain.com)"

    login(token=os.getenv('HUGGINGFACE_TOKEN'))

    args = parse_args()

    # Set model name, precision, and other parameters based on passed args
    model_name = args.model_name
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    embedding_model_name = args.embedding_model_name
    embedding_dim = args.embedding_dim
    splitter_type = args.splitter_type
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    text_files_path = args.text_files_path
    sublink_files_path = args.sublink_files_path
    sublink_files_nums = args.sublink_files_nums
    qes_file_path = args.qes_file_path
    top_k_search = args.top_k_search
    retriever_type = args.retriever_type
    retriever_algorithm = args.retriever_algorithm
    rerank = args.rerank
    top_k_rerank = args.top_k_rerank
    rerank_model_name = args.rerank_model_name
    hypo = args.hypo
    output_file = args.output_file
    qa_nums = args.qa_nums
    random.seed(42)

    # check if rerank is set to True
    if rerank:
        print("Reranking is set to True.")

    # Step 1: Initialize the Hugging Face model as your LLM
    print("Initializing the Hugging Face model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer, 
        torch_dtype=dtype
    )
    print("Model initialized successfully!")

    # Step 2: Load the Sentence Transformers model for embeddings
    # embedding_model = SentenceTransformer(embedding_model_name, truncate_dim=embedding_dim)
    docs_length = f"main160_sublink{sublink_files_nums}"
    model_name_str = embedding_model_name.split('/')[-1]
    embeddings_file_path = f"data/embeddings/embeddings_{model_name_str}_{docs_length}_{splitter_type}_{chunk_size}_{chunk_overlap}.npy"
    splits_file_path = f"data/embeddings/splits_{model_name_str}_{docs_length}_{splitter_type}_{chunk_size}_{chunk_overlap}.pkl"
    embeddings = None
    splits = None
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print(f"Start loading qa from {qes_file_path}")
    qa_test_data_path = qes_file_path
    qa_df = pd.read_csv(qa_test_data_path)
    print(len(qa_df))
    if len(qa_df) != 574:
        qa_df = qa_df.sample(qa_nums, random_state=221)
    # print(f"End loading texts. Number of documents for retrieval: {len(docs)}. Number of QA pairs: {len(qa_df)}")
    print(f"Loaded {len(qa_df)} qas")
    if not os.path.exists(embeddings_file_path):
        # Step 3: load the text files for building the index and qa evaluation
        print(f"Start loading texts from {text_files_path}")
        docs = load_text_files(path=text_files_path)
        # Step 4: Split the documents into smaller chunks
        # Wrap text strings in Document objects
        documents = []
        for text in tqdm(docs, desc="wrapping text in Document objects"):
            documents.append(Document(page_content=text))
        del docs

        if sublink_files_nums != 0:
            all_sublink_docs = None
            sublink_file_store_path = "data/embeddings/sublink_docs.pkl"
            if os.path.exists(sublink_file_store_path):
                print(f"Start loading sublink files from {sublink_file_store_path}")
                with open(sublink_file_store_path, "rb") as f:
                    all_sublink_docs = pickle.load(f)
            else:
                print(f"Start reading all sublink files")
                all_sublink_docs = load_text_files(path=sublink_files_path)
                print(f"Finish loading {len(all_sublink_docs)} sublinks, now store it")
                with open(sublink_file_store_path, 'wb') as f:
                    pickle.dump(all_sublink_docs, f)
                print(f"Store all sublink file in {sublink_file_store_path}")

            sampled_sublink_docs = random.sample(all_sublink_docs, sublink_files_nums)
            for text in tqdm(sampled_sublink_docs, desc="wrapping text in Document objects"):
                documents.append(Document(page_content=text))
            del sampled_sublink_docs
            del all_sublink_docs

        if splitter_type == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif splitter_type == "character":
            # separator should be the ., !, or ? or " "
            # separation_pattern = r"\.|\?|\!| $"
            text_splitter = CharacterTextSplitter(
                separator=" ",
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif splitter_type == "token":
            # note that the chunking is done at the token level
            text_splitter = TokenTextSplitter(
                chunk_size=int(chunk_size / 4), 
                chunk_overlap=int(chunk_overlap / 4))
        elif splitter_type == "semantic":
            text_splitter = SemanticChunker(
                embeddings=embedding_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=80)
        else:
            print("Invalid splitter type. Please choose between recursive, character, token, or semantic.")
        
        splits = text_splitter.split_documents(documents)
        del documents
        print(f"End Spliting texts -- Number of splits: {len(splits)}")
        
        # Step 5: Create Chroma vectorstore with embeddings from Sentence Transformers
        embeddings = embedding_model.embed_documents(
            [doc.page_content for doc in tqdm(splits, desc="Embedding texts")])
        print(f"End Embedding texts")
        # Free GPU cache after generating embeddings
        torch.cuda.empty_cache()
        print(f"Start Saving embeddings and splits")
        np.save(embeddings_file_path, embeddings)
        with open(splits_file_path, 'wb') as f:
            pickle.dump(splits, f)
        doc_metadata = [doc.metadata for doc in splits]  # Save metadata for documents
        np.save(f"data/embeddings/metadata_{docs_length}_{splitter_type}_{chunk_size}_{chunk_overlap}.npy", doc_metadata)
        print(f"embeddings saved in {embeddings_file_path}, splits saved in {splits_file_path}")
    else:
        print("Embeddings already exists! loading embeddings")
        # Step 1: Load embeddings from the saved NumPy file
        embeddings = np.load(embeddings_file_path)
        with open(splits_file_path, 'rb') as f:
            splits = pickle.load(f)
        # Step 2: Load document metadata if needed
        # doc_metadata = np.load("doc_metadata.npy", allow_pickle=True)
        print("end loading")
    # Step 6: Create the RAG prompting pipeline
    prompt_template = PromptTemplate(
        input_variables=['context', 'question'],
        template=PROMPT_TEMPLATE
    )

    # Update the HumanMessagePromptTemplate with the new PromptTemplate
    human_message_template = HumanMessagePromptTemplate(prompt=prompt_template)

    # Update the ChatPromptTemplate with the modified message
    chat_prompt_template = ChatPromptTemplate(
        input_variables=['context', 'question'],
        messages=[human_message_template]
    )

    prompt = chat_prompt_template
    
    # Step 7: Generate answers for the questions
    print("Building the vectorstore...")
    if retriever_type == "CHROMA":
        print("Building the vectorstore Chroma...")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model, collection_name="collectionChroma")
        chroma_retriever = vectorstore.as_retriever(search_type=retriever_algorithm, search_kwargs={'k': top_k_search})
        retriever = chroma_retriever
    elif retriever_type == "FAISS":
        print("Building FAISS...")
        # embeddings_np = np.array(embeddings).astype("float32")
        faiss_retriever = FAISS.from_documents(splits, embedding_model).as_retriever(search_type=retriever_algorithm, search_kwargs={"k": top_k_search})
        retriever = faiss_retriever
    else:
        print("Invalid retriever type. Please choose between FAISS or CHROMA.")
    
    print("Retriever built successfully!")
    torch.cuda.empty_cache()
    del splits
    
    answer_generation(
        qa_df, output_file, retriever, 
        generation_pipe, prompt, rerank, rerank_model_name, hypo, top_k_rerank=top_k_rerank)
    
    print(f"QA evaluation completed! Results saved to {output_file}")
