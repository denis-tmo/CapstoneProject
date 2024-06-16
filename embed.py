import os
import numpy as np
import re
import random
from sklearn.metrics.pairwise import cosine_similarity

import openai
from openai import OpenAI
from dotenv import load_dotenv

import pdb
# pdb.set_trace()

MAX_HISTORY = 5
EMBEDDING_MODEL = "text-embedding-ada-002"

question_list = [

    'What was the death toll of Bonaparte\'s campaigns?',
#        'What is the outcome of the Waterloo battle?',
#        'What is the Down Jones?',
#        'Where was Napoleon born?',
#        'When was Smolensk was reached and what did the soldiers do?'

]

global doc_texts

# Initialize conversation history
conversation_history = []

######################################################################################################################################################

def get_next_1000_words(text, num_words=1000):

    # Split the text into a list of words
    words = text.split()
    
    # Determine the number of words in the list
    total_words = len(words)
    
    # Ensure there are enough words to take the next 1000 words
    if total_words <= num_words:
        return ' '.join(words)
    
    # Randomly select a starting index
    start_index = random.randint(0, total_words - num_words)
    
    # Slice the list to get the next 1000 words
    selected_words = words[start_index:start_index + num_words]
    
    # Join the selected words back into a string
    result = ' '.join(selected_words)
    
    return result

def get_embeddings(client, text, model=EMBEDDING_MODEL):

    tlist = "\n".join(text) 
    tlist = get_next_1000_words( tlist, 1000)
    response = client.embeddings.create(input = tlist, model=model)
    qe = np.array(response.data[0].embedding)
    qe = qe.reshape(1, -1)

    return qe

def search_documents(client, question, doc_texts, doc_embeddings, top_k=3):

    question_embedding = get_embeddings(client, question)
    similarities = cosine_similarity(question_embedding, doc_embeddings)
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]

    return [doc_texts[i] for i in top_indices]

def sanitize_input(input_text):
    # Basic example: strip leading/trailing whitespace and limit length
    sanitized_text = re.sub(r'[<>]', '', input_text.strip()[:1000])
    return sanitized_text

def ask_question(client, question, relevant_docs):
    global conversation_history

    question = sanitize_input(question)
    context = "\n\n".join(relevant_docs)

    # Define the System Role
    messages = [
        {"role": "system", "content": f"Using the below contexts:\n\n{context}\n\n**Please answer the following question.**\n{question}"}
    ]
    # Include the conversation history
    for history in conversation_history[-MAX_HISTORY:]:
        messages.append({"role": "user", "content": sanitize_input(history['question'])})
        messages.append({"role": "assistant", "content": sanitize_input(history['answer'])})

    # Add the user's question to the messages as a User Role
    messages.append({"role": "user", "content": question})

    # Generate a completion using the user's question
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    answer = chat_completion.choices[0].message.content

    # Update the conversation history
    conversation_history.append({"question": question, "answer": answer})

    return answer

def init_embed( client, token ):

    global doc_texts

    # Read and process files
    fpath = 'bona.txt'
    try:
        with open(fpath, "r") as infile:
            lines = infile.readlines()
    except Exception as e:
        print('Please make sure the file {fpath} is in your current directory')
        exit(1)

    doc_texts = lines
    doc_embeddings = get_embeddings(client, doc_texts)

    return doc_embeddings

def get_embed_answer( client, doc_embeddings, question ):

    global doc_texts

    relevant_docs = search_documents(client, question, doc_texts, doc_embeddings)
    if relevant_docs == ['O']:
        return 'I did not find any relevant documents for your question.'

    answer = ask_question(client, question, relevant_docs)

    return answer

###################################################################################################################################
#
###################################################################################################################################

if __name__ == '__main__':

    # OpenAI API key
    load_dotenv()
    client = OpenAI(api_key=openai.api_key)
    doc_emb = init_embed( client, 'ss' )

    qlist = [
        "what was Murat biggest victory?",
        "When was Smolensk reached and what did the soldiers do?", 
        "Where was Napoleon born?", 
        "What happened after Austerlitz?"
    ]

    response = get_embed_answer(client, doc_emb, qlist[ 0 ])
