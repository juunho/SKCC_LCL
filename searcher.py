from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
#from FlagEmbedding import BGEM3FlagModel
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
#from langchain_openai import AzureChatOpenAI
import os


def exp_normalize(x):
    b = np.max(x)
    y = np.exp(x - b)
    return y / np.sum(y)

class DenseSearcher:
    def __init__(self, collection_name):
        rerank_model_path= 'Dongjin-kr/ko-reranker'
        
        self.collection_name = collection_name
        # Initialize encoder model
        self.emb_model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
        #self.emb_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device="cpu")
        # initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_path).eval()

    def exp_normalize(self, x):
        b = np.max(x)
        y = np.exp(x - b)
        return y / np.sum(y)

    def ask_llm(self, query: str):              
        vector = self.emb_model.encode(query).tolist()
        #vector = self.emb_model.encode(query, return_dense=True)['dense_vecs']
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            limit=5,
        )

        answers = [f"{hit.payload['T전화 Title']}: {hit.payload['T전화 답변']}" for hit in search_result]
        pairs = [[query, answer] for answer in answers]
        print("stage1:", [f"{hit.score} {hit.payload['T전화 Title']}: {hit.payload['T전화 답변']}" for hit in search_result])
        print("------------------------------------------------------------------------------------------")
        
        with torch.no_grad(): 
            inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())
            
            # Create a list of tuples (score, pair)
            scored_pairs = list(zip(scores, pairs))
            
            # Filter out pairs with scores lower than 0.1
            scored_pairs_filtered = [pair for pair in scored_pairs if pair[0] >= 0.1]
            
            # Sort the filtered list in descending order of scores
            scored_pairs_filtered.sort(reverse=True, key=lambda x: x[0])

            # Extract top 2 results from the filtered list
            top2_results = scored_pairs_filtered[:2]
            
            # Extract result[n] values from the top 2 pairs
            top2_values = [pair[1] for _, pair in top2_results]
            print("stage2:", top2_results)
            print("------------------------------------------------------------------------------------------")
        
                            
        template = """
        You are an AI assistant for Frequently Asked Questions tasks. Answer must be in Korean.
        Answer ONLY with the facts listed in the list of sources below. And the first content is more relelvant to user question. 
        If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below.
        
        Question: {question} 
        Sources: {context} 

        Answer:
        """

        
        promptTemplate = PromptTemplate.from_template(template)
        prompt = promptTemplate.format(question=query, context="\n".join(top2_values))
        
        
        llm = AzureChatOpenAI(
            deployment_name="skt-rag2",
            max_tokens = 512
        ) 

        llm_answer = llm.invoke(prompt)
        
        instruction = query
        response = llm_answer.content
        context = top2_values

        return instruction, response, context