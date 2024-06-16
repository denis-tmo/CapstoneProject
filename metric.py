# Set the OpenAI API key
import os
import openai
from langchain.document_loaders import DirectoryLoader

from ragas.testset import TestsetGenerator
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from ragas.llms import LangchainLLM

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.environ["OPENAI_API_KEY"]

loader = DirectoryLoader("data", show_progress=True)
documents = loader.load()

# Add custom llms and embeddings
generator_llm = LangchainLLM(llm=ChatOpenAI(model="gpt-3.5-turbo"))
critic_llm = LangchainLLM(llm=ChatOpenAI(model="gpt-4"))
embeddings_model = OpenAIEmbeddings()

# Change resulting question type distribution
testset_distribution = {
    "simple": 0.25,
    "reasoning": 0.5,
    "multi_context": 0.0,
    "conditional": 0.25,
}

# percentage of conversational question
chat_qa = 0.2

test_generator = TestsetGenerator(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings_model=embeddings_model,
    testset_distribution=testset_distribution,
    chat_qa=chat_qa,
)
testset = test_generator.generate(documents, test_size=5)
testset_df = testset.to_pandas()

print(f'{testset_df.head()}')