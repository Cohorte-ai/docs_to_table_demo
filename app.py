
import os
import json
import pprint
import openai
import chromadb

from chromadb.utils import embedding_functions
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st


def process_json_file(input_filename):
    # Read the JSON file
    with open(input_filename, 'r') as file:
        data = json.load(file)

    # Iterate over the JSON data and extract required table elements
    extracted_elements = []
    for entry in data:
        if entry["type"] == "Table":
            extracted_elements.append(entry["metadata"]["text_as_html"])

    # Write the extracted elements to the output file
    with open("outputs/nvidia-yolox.txt", 'w') as output_file:
        for element in extracted_elements:
            output_file.write(element + "\n\n")  # Adding two newlines for separation

def extract_tables_to_docs(filename, output_dir, strategy="hi_res", model_name="yolox", chunk_size=1000, chunk_overlap=200):
    # Partition the PDF and extract table structure
    elements = partition_pdf(filename=filename, strategy=strategy, infer_table_structure=True, model_name=model_name)

    # Convert elements to JSON and process the JSON file
    elements_to_json(elements, filename=f"{filename}.json")
    process_json_file(f"{filename}.json")

    # Load the processed text file
    text_file = f"{output_dir}/nvidia-yolox.txt"
    loader = TextLoader(text_file)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    return docs, text_file

# Set up OpenAI API key and embeddings
os.environ['OPENAI_API_KEY'] = "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"
embeddings = OpenAIEmbeddings()

# # Define input and output paths
# filename = "reports/Q4FY24-CFO-Commentary.pdf"
# output_dir = "outputs"

# # Extract tables and convert to documents
# docs, text_file = extract_tables_to_docs(filename, output_dir)

# # Create a Chroma database from the documents and embeddings
# db = Chroma.from_documents(docs, embeddings)

# # Initialize the model and retriever
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

# # List of questions
# questions = [
#     "How much revenue did Nvidia make in Q3 FY24?",
#     "What was the operating income for Q2 FY24?",
#     "How much was the net income in Q1 FY24?",
#     "What was the Q/Q revenue growth between Q1 and Q2 of FY24?",
#     "How much revenue did Nvidia's Data Center produce in Q2 FY24?",
#     "How much revenue did Nvidia's Gaming sector produce in Q2 FY24?",
#     "What percent of the total revenue in Q2 FY24 did Data Centers represent?"
# ]

# # Store responses in output_list
# output_list = []

# for query in questions:
#     response = qa_chain({"query": query})
#     output_list.append(response)

# # Use pprint to pretty print the output list
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(output_list)

# # Display the HTML tables from the text file
# with open(text_file, "r") as file:
#     html_content = file.read()
# display(HTML(html_content))
import tempfile
def main():
    st.title("Question Answering App")
    
    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file is not None:
        # Load and process the PDF
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        # Define input and output paths
        filename = temp_file_path
        output_dir = "outputs"

        # Extract tables and convert to documents
        docs, text_file = extract_tables_to_docs(filename, output_dir)

        # Create a Chroma database from the documents and embeddings
        db = Chroma.from_documents(docs, embeddings)

        # Initialize the model and retriever
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

        # Get user question
        question = st.text_input("Ask a question about the document:")
        
        if question:
            # Query the question and display the answer
            result = qa_chain({"query": question})
            st.write("Answer:")
            st.write(result['result'])
            
            # Display the relevant source documents
            st.write("Source Documents:")
            for doc in result['source_documents']:
                st.write(doc.page_content)

if __name__ == "__main__":
    main()