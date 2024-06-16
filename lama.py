import os
# llama-parse is async-first, running the async code in a notebook requires the use of nest_asyncio
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from PyPDF2 import PdfReader, PdfWriter

from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)

node_parser = MarkdownElementNodeParser(
    llm=OpenAI(model="gpt-3.5-turbo-0125", max_tokens=2048, temperature=0.7), num_workers=8
)

def extract_pages(input_pdf, start_page, end_page, output_pdf):

    pdf_writer = PdfWriter()
    pdf_reader = PdfReader(input_pdf)
    
    for page in range(start_page, end_page):
        pdf_writer.add_page(pdf_reader.pages[page])
    
    with open(output_pdf, 'wb') as output:
        pdf_writer.write(output)

async def init_lama():

    nest_asyncio.apply()

    # API access to llama-cloud
    os.environ["LLAMA_CLOUD_API_KEY"] = "YOUR-KEY"

    # Using OpenAI API for embeddings/llms
    os.environ["OPENAI_API_KEY"] = "YOUR-KEY"

    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-3.5-turbo-0125", max_tokens=2048, temperature=0.7)

    Settings.llm = llm
    Settings.embed_model = embed_model

    input_pdf = "wheeler_napoleon.pdf"
    output_pdf = "wheeler_napoleon_part.pdf"

    # Get total number of pages
    pdf_reader = PdfReader(input_pdf)
    total_pages = len(pdf_reader.pages)

    # Extract 1/4 of the PDF
    start_page = 0
    end_page = total_pages // 10

    extract_pages(input_pdf, start_page, end_page, output_pdf)
    
    # Load and process the extracted part
    documents = LlamaParse(result_type="markdown").load_data(output_pdf)
    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    
    recursive_index = VectorStoreIndex(nodes=base_nodes + objects)
    raw_index = VectorStoreIndex.from_documents(documents)
    
#    return recursive_index, raw_index

    if False:
        documents = LlamaParse(result_type="markdown").load_data("wheeler_napoleon.pdf")

        nodes = node_parser.get_nodes_from_documents(documents)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

        recursive_index = VectorStoreIndex(nodes=base_nodes + objects)
        raw_index = VectorStoreIndex.from_documents(documents)

    reranker = FlagEmbeddingReranker(
        top_n=5,
        model="BAAI/bge-reranker-large",
    )

    recursive_query_engine = recursive_index.as_query_engine(
        similarity_top_k=15, node_postprocessors=[reranker], verbose=True
    )

    raw_query_engine = raw_index.as_query_engine(
        similarity_top_k=15, node_postprocessors=[reranker]
    )

    return recursive_query_engine, raw_query_engine

def get_lama_answer(message, raw_qengine, recursive_qengine):

    response_1 = raw_qengine.query(message)
    response_2 = recursive_qengine.query(message)

    response = 'Basic Query Engine: ' + str(response_1) + 'Recursive Retriever Query Engine: ' + str(response_2)

    return response

#######################################################################################################################################
#
#######################################################################################################################################

if __name__ == "__main__":

    recursive_query_engine, raw_query_engine = init_lama()

    query = "What is the change of free cash flow and what is the rate from the financial and operational highlights?"
    query = "How many wars did Napoleon fight?"
    query = ' who won the friedland battle?'

    response_1 = raw_query_engine.query(query)
    print("\n***********New LlamaParse+ Basic Query Engine***********")
    print(response_1)

    response_2 = recursive_query_engine.query(query)
    print("\n***********New LlamaParse+ Recursive Retriever Query Engine***********")
    print(response_2)
