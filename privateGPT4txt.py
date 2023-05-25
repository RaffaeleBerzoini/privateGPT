#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
question_file = '../q.txt'
answer_file = '../a.txt'

from constants import CHROMA_SETTINGS

def main():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] # [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    
    query = read_text_file(question_file)
    if query == "exit":
        return

    # Get the answer from the chain
    res = qa(query)
    answer, docs = res['result'], res['source_documents']

    # Print the result
    print("\n\n> Question:")
    print(query)
    print("\n> Answer:")
    print(answer)

    write_text_file(answer_file, answer)
    append_text_file(answer_file, "\nSOURCES:")

    # Print the relevant sources used for the answer
    i = 1
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
        append_text_file(answer_file, "\n\t-SOURCE " + str(i) + "\n")
        i += 1
        append_text_file(answer_file, document.page_content)

    remove_empty_lines(answer_file)

def read_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
            return contents
    except FileNotFoundError:
        print("File not found! Closing program...")
    except IOError:
        print("Error reading the file! Closing program...")
    return "exit"

def write_text_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        print("File written successfully.")
    except IOError:
        print("Error writing to the file!")

def append_text_file(file_path, content):
    try:
        with open(file_path, 'a') as file:
            file.write(content)
        print("Content appended successfully.")
    except IOError:
        print("Error appending content to the file!")

def remove_empty_lines(file_path):
    try:
        with open(file_path, 'r+') as file:
            lines = file.readlines()
            file.seek(0)
            file.truncate()
            for line in lines:
                if line.strip():
                    file.write(line)
        print("Empty lines removed successfully.")
    except FileNotFoundError:
        print("File not found!")
    except IOError:
        print("Error removing empty lines from the file!")




if __name__ == "__main__":
    main()
