import os
import torch
import pandas as pd
from tqdm import tqdm

import faiss
import numpy as np

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain import hub
from langchain_chroma import Chroma
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, 
    CharacterTextSplitter, 
    TokenTextSplitter
)
from langchain.docstore.document import Document
from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    PromptTemplate
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from sentence_transformers import SentenceTransformer

# ========================================
# Custom Class for FAISS
# ========================================
# class FAISSRetriever:
#     """A FAISS retriever to handle vector search and document retrieval."""
#     def __init__(self, embeddings, documents):
#         self.documents = documents  # Store documents separately since FAISS does not handle metadata
#         self.index = self._create_faiss_index(embeddings)

#     def _create_faiss_index(self, embeddings):
#         """
#         Create a FAISS index from the embeddings.
#         Args:
#         embeddings (list): List of document embeddings.

#         Returns:
#         faiss.IndexFlatL2: FAISS index built from the embeddings.
#         """
#         d = len(embeddings[0])  # Dimensionality of the embeddings
#         index = faiss.IndexFlatL2(d)  # Using L2 distance for similarity search
#         index.add(embeddings)
#         return index

#     def retrieve(self, query_embedding, k=3):
#         """
#         Retrieve the top-k documents closest to the query embedding.
#         Args:
#         query_embedding (ndarray): The embedding of the query.
#         k (int): The number of nearest neighbors to return.

#         Returns:
#         list: A list of the top-k documents.
#         """
#         distances, indices = self.index.search(query_embedding, k)
#         return [self.documents[i] for i in indices[0]]  # Return the documents corresponding to the nearest neighbors

# # ========================================
# # Custom Class for Sentence Transformer Embeddings
# # ========================================
# class SentenceTransformerEmbeddings:
#     """A wrapper class for Sentence Transformers model to provide the same interface as OpenAIEmbeddings."""
#     def __init__(self, model):
#         self.model = model

#     # Method to generate embeddings for a list of texts (documents)
#     def embed_documents(self, texts):
#         return [self.model.encode(text) for text in texts]

#     # Method to generate embeddings for a single query (optional but useful)
#     def embed_query(self, text):
#         return self.model.encode(text)


# ========================================
# Helper Functions
# ========================================
def load_text_files(path):
    """
    Load text files from the given path.
    
    Args:
    path (str): The path to the directory containing the text files.
    
    Returns:
    list: A list of text documents.
    """
    
    docs = []

    # Check if the path is a directory
    if os.path.isdir(path):
        # Iterate over files in the directory
        for file_name in os.listdir(path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(path, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    docs.append(file.read())
    elif os.path.isfile(path) and path.endswith(".txt"):
        # If the path is a file, directly read it
        with open(path, 'r', encoding='utf-8') as file:
            docs.append(file.read())

    return docs

# def query_retriever(query, retriever_type, retriever, embedding_model, k=3):
#     # Retrieve context using the retriever
#     if retriever_type == "CHROMA":
#         retrieved_docs = retriever.invoke(query)
#     elif retriever_type == "FAISS":
#         # Chroma's k is set when creating the retriever. Possibly can be changed dynamically with further research.
#         query_embedding = embedding_model.encode(query).reshape(1, -1).astype("float32")
#     retrieved_docs = retriever.invoke(query)
#     return retrieved_docs

def format_retreived_docs(docs):
    """
    Format the retrieved documents by the following format:
    Context 1: <Document 1>
    Context 2: <Document 2>
    ...
    """
    # reverse the order of the documents
    docs = reversed(docs)
    return "\n\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(docs)])


def rerank_docs(query, retriever, rerank_model_name, k=3):
    """
    Rerank the retrieved documents based on the query using Flashrank.
    """
    # DEFAULT_MODEL_NAME = "ms-marco-MultiBERT-L-12"
    compressor = FlashrankRerank(top_n=k, model=rerank_model_name)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever)
    
    compressed_docs = compression_retriever.invoke(query)
    
    return compressed_docs

def get_hypo_doc(query, generation_pipe):
    """
    Generate a hypothesis document for the given query using the language model.
    """
    template = """Imagine you are an expert providing a detailed and factual explanation in response to the query '{query}'. 
    Your response should include all key points that would be found in a top search result, without adding any personal opinions, commentary, or experiences. 
    Do not include any subjective phrases such as 'I think', 'I believe', or 'I am not sure'. Do not apologize, hedge, or express uncertainty. 
    The response should be structured as an objective, factual explanation only, without any conversational elements or chatting.
    If you are truly uncertain and cannot provide an accurate answer, simply respond with: 'Unavailable: {query}'.
    Otherwise, answer confidently with only the relevant information.
    """
    
    messages = [
        {"role": "user", "content": template.format(query=query)}
    ]
    
    with torch.no_grad():
        hypo_doc = generation_pipe(messages, max_new_tokens=100, return_full_text=False)[0]["generated_text"]
    
    print("Question:", query)
    print("Hypothesis Document:", hypo_doc)
    
    # check if the hypo_doc starts with "Unavailable"
    if hypo_doc.startswith("Unavailable"):
        print("Using the original query.")
        return query
    else:
        return hypo_doc


def answer_generation(
    qa_df, output_file, retriever, generation_pipe, 
    prompt, rerank, rerank_model_name, hypo, top_k_rerank=3):
    """
    Generate answers for the given questions using the retriever and the generation pipeline.
    
    Args:
    questions (list): A list of questions to answer.
    retriever (Chroma): A retriever object to retrieve documents.
    generation_pipe (pipeline): A pipeline object for text generation.
    prompt (ChatPromptTemplate): A ChatPromptTemplate object for generating prompts.
    
    Returns:
    list: A list of generated answers
    """

    print("Generating answers for the questions...")
    
    # check if the output file 
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f_out:
            f_out.write(",".join(list(qa_df.columns) + ["Generated_Answer"]) + "\n")
            start_idx = 0
    else:
        # calculate the number of rows in the output file
        with open(output_file, 'r') as f_out:
            num_rows = sum(1 for line in f_out)
            # the iteration will start from the next row
            start_idx = num_rows - 1

    # iterate over the dataframe
    with open(output_file, 'a') as f_out:
        for idx, row in tqdm(qa_df.iterrows(), total=len(qa_df)):
            # skip the rows that have been processed
            if idx < start_idx:
                continue
            query = row["Question"]
            
            if hypo:
                query = get_hypo_doc(query, generation_pipe) 
            
            # Retrieve documents based on the question
            if rerank:
                print("Reranking documents...")
                retrieved_docs = rerank_docs(query, retriever, rerank_model_name, k=top_k_rerank)
            else:
                retrieved_docs = retriever.invoke(query)
            
            # Format the documents
            context = format_retreived_docs(retrieved_docs)

            # Create the full prompt using the prompt template
            prompt_messages = prompt.format_messages(context=context, question=row["Question"])
            full_prompt = "\n".join(message.content for message in prompt_messages)
            
            messages = [
            {"role": "user", "content": full_prompt},
            ]
            with torch.no_grad():
                llm_output = generation_pipe(
                    messages, max_new_tokens=20, return_full_text=False)[0]["generated_text"]
            
            row["Generated_Answer"] = llm_output
            pd.DataFrame([row]).to_csv(f_out, header=False, index=False)
            # Clear cache after generation
            del retrieved_docs, context, prompt_messages, full_prompt, messages, llm_output
            torch.cuda.empty_cache()

# ========================================
# Constants and Configuration
# ========================================

PROMPT_TEMPLATE = """
You are an expert assistant answering factual questions about Pittsburgh or Carnegie Mellon University (CMU). 
Use the retrieved information to give a detailed and helpful answer. If the provided context does not contain the answer, leverage your pretraining knowledge to provide the correct answer. 
If you truly do not know, just say "I don't know."

Important Instructions:
- Answer concisely without repeating the question.
- Use the provided context if relevant; otherwise, rely on your pretraining knowledge.
- Do **not** use complete sentences. Provide only the word, name, date, or phrase that directly answers the question. For example, given the question "When was Carnegie Mellon University founded?", you should only answer "1900".

Examples:
Question: Who is Pittsburgh named after? 
Answer: William Pitt
Question: What famous machine learning venue had its first conference in Pittsburgh in 1980? 
Answer: ICML
Question: What musical artist is performing at PPG Arena on October 13? 
Answer: Billie Eilish

Context: \n\n {context} \n\n
Question: {question} \n\n
Answer:
"""