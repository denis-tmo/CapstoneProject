import os
import openai
import pickle
import random

from openai import OpenAI
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.embeddings import OpenAIEmbeddings

from embed import sanitize_input

import pdb
# pdb.set_trace()

MAX_HISTORY = 5
conversation_history = []

# Load OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Read the text file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Generate embeddings
def generate_embeddings(texts, model="text-embedding-ada-002", batch_size=100):

    try:
        with open('qdr_emb.pkl', 'rb') as f:
            embeddings = pickle.load(f)
    except Exception as e:
        print(f'emb_list not found')
        embeddings = []

    if len(embeddings) != 0:
        return embeddings

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch_texts,
            model=model
        )
        batch_embeddings = [res.embedding for res in response.data]
        embeddings.extend(batch_embeddings)

    with open('qdr_emb.pkl', 'wb') as file:
        pickle.dump(embeddings, file)

    return embeddings

# Setup Qdrant
def setup_qdrant(embedding_dim):
    qdrant_client = QdrantClient(":memory:")
    collection_name = "rag_contexts"
    vector_params = models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
    
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
    
    qdrant_client.create_collection(collection_name, vectors_config=vector_params)
    return qdrant_client

# Store embeddings in Qdrant
def store_embeddings_in_qdrant(embeddings, texts, qdrant_client):
    points = [
        models.PointStruct(
            id=idx,
            vector=embedding,
            payload={"text": text}
        ) for idx, (embedding, text) in enumerate(zip(embeddings, texts))
    ]
    qdrant_client.upsert(collection_name="rag_contexts", points=points)

# Search Qdrant
def search_qdrant(query, qdrant_client, embedding_model, top_k=5):
    query_vector = embedding_model.embed_query(query)
    search_response = qdrant_client.search(
        collection_name='rag_contexts',
        query_vector=query_vector,
        limit=top_k
    )
    return [hit.payload["text"] for hit in search_response]

# Ask question using OpenAI
def ask_question(question, text, client):

    global conversation_history

    messages=[
        {"role": "system", "content": "You are a helpful assistant."},

        {"role": "assistant", "content": text},
        {"role": "user", "content": question},
    ]

    # Include the conversation history
    for history in conversation_history[-MAX_HISTORY:]:
        messages.append({"role": "user", "content": sanitize_input(history['question'])})
        messages.append({"role": "assistant", "content": sanitize_input(history['answer'])})

    chat_completion = client.chat.completions.create(

        model="gpt-3.5-turbo",
        messages=messages
    )
    answer = chat_completion.choices[0].message.content

    return answer

def init_qdrant( fpath ):

    # Randomly sample 1/4 of the lines from file_content
    file_content = read_file(fpath)
    sample_size = len(file_content) // 4
    file_content = random.sample(file_content, sample_size)

    embeddings = generate_embeddings(file_content)
    embedding_dim = len(embeddings[0])
    qdrant_client = setup_qdrant(embedding_dim)
    store_embeddings_in_qdrant(embeddings, file_content, qdrant_client)

    embedding_model = OpenAIEmbeddings(api_key=api_key)

    return file_content, qdrant_client, embedding_model

def get_qdrant_answer( question, file_content, qdrant_client, embedding_model ):

    relevant_texts = search_qdrant(question, qdrant_client, embedding_model)
    if len(relevant_texts) == 0:
        answer = ask_question(question, relevant_texts, client)
    else:
        answer = ask_question(question, file_content[0], client)

    print(f"question: {question} rev: {len(relevant_texts)} \nAnswer: {answer}")

    return answer

###################################################################################################################################
#
###################################################################################################################################

if __name__ == '__main__':

    # OpenAI API key
    load_dotenv()
    client = OpenAI(api_key=openai.api_key)

    fpath = 'bona.txt'
    file_content, qdrant_client, embedding_model = init_qdrant( fpath )
    
    qlist = [
        "When was Smolensk reached and what did the soldiers do?", 
        "Where was Napoleon born?", 
        "What happened after Austerlitz?"
    ]

    for question in qlist:
        relevant_texts = search_qdrant(question, qdrant_client, embedding_model)
        if len(relevant_texts) == 0:
            answer = ask_question(question, relevant_texts, client)
        else:
            answer = ask_question(question, file_content[0], client)

        print(f"question: {question} rev: {len(relevant_texts)} \nAnswer: {answer}")
