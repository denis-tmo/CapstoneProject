# CapstoneProject
Deploy a Retrieval-Augmented Generation (RAG) application in Flask.

Capstone project

Unfortunately I was not able to launch my application on fly.io so I am using Flask.
 ✖ Failed: error creating a new machine: failed to launch VM: Your organization has reached its machine limit. Please contact billing@fly.io

Also to keep the cost down I had to trunked the size of the data I am using. It will affect the results but the point
of this assignment is to learn how to use AI. 
I choose Napoleon Bonaparte biography as an example since it is easy to verify the accuracy of the repsonses.
The Flask application is a bit not user friendly but you can ask multiple questions with 1,2 and should have what you are looking for (except fly.io...)
This application might be a bit slow, so just be patient.

The application implements internally a token-based authentication for secure access.

in lama.py please replace YOUR-KEY
    # API access to llama-cloud
    os.environ["LLAMA_CLOUD_API_KEY"] = "YOUR-KEY"
    # Using OpenAI API for embeddings/llms
    os.environ["OPENAI_API_KEY"] = "YOUR-KEY"

The directory has all the files needed to run the application, just type:

1. python capstone.py
2. right click on "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)"
3. app should come up in your browser.

I implemented 5 different ways to answer a question based on your classes.

1. Processes the message as a 'chat message' in the OpenAI ChatML format
In capstone.py, gets a token and do a search with or without URL

2. Ask questions about Napoleon Bonaparte using a Qdrant search
In quadrant.py, keeps track of the question history, sanitize input, store embedding in Qdrant and ask question using OpenAI 

3. Ask questions about Napoleon Bonaparte using LLM (might be slow…)
In embed.py, keeps track of the question history, sanitize input,  uses cosine_similarity

4. Ask questions about Napoleon Bonaparte using Lama-Cloud, LlamaParse (might be slow…)
In lama.py, use raw_query_engine and recursive_quey engine

5. Ask questions about Napoleon Bonaparte using Langchain (might be slow…)
In rag.py use LangChainQueryEngine

You can run all 5 files capstone.py. qdrant.py, embed.py and rag.py separately from the command line.
Option 2, 3, 4, 5 assumes that the question is about Napoleon Bonaparte. The app downloads/reads the relevant files to get context about Napoleon Bonaparte.
Of course this is just an example and the application can handle any type of PDF. It is also interesting to compare the differents answers received by option 2,3,4,5.

For example the question 'What happened at Smolensk and what did the soldiers do?' 
    option 1 will answer about WW two since it does not have any context,
    option 2,3,4,5 will reuturn a slightly different answer with the correct context

I’ll insert a metrics soon.
I found it more interesting to spend time on how to get answers rather than deploying the application.

Thanks


