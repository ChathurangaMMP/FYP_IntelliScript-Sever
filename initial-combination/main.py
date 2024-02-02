import chainlit as cl
from transformers import pipeline
import numpy as np
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain_community.document_loaders import PyPDFLoader
import PyPDF2
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
import pandas as pd

# DATA_PATH = "/kaggle/working/FYP_IntelliScript-Sever/scraped-data"
DB_FAISS_PATH = "vectorstores/"

# Load the summarization model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")


def summarize_text(text, max_length=1000):
    # Tokenize and summarize the input text
    inputs = tokenizer.encode(
        "summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs, max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def pdf_reader(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages


def show_vstore(store):
    vectore_df = store_to_df(store)
    display(vectore_df)


def store_to_df(store):
    v_dict = store.docstore._dict
    data_rows = []
    for k in v_dict.keys():
        doc_name = v_dict[k].metadata['source'].split('/')[-1]
        page_number = v_dict[k].metadata["page"]+1
        content = v_dict[k].page_content
        data_rows.append({"chunk_id": k, "dcoument": doc_name,
                         "page": page_number, "content": content})
    vector_df = pd.DataFrame(data_rows)
    return vector_df


def delete_from_knowleadgebase(knowleadgebase, document):
    vector_df = store_to_df(knowleadgebase)
    chunk_list = vector_df.loc[vector_df['dcoument']
                               == document]['chunk_id'].tolist()
    knowleadgebase.delete(chunk_list)


def add_knowleadgebase(knowleadgebase, path):
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs={'device': 'cuda'})
    extension = FAISS.from_documents(pdf_reader(path), embeddings)
    knowleadgebase.merge_from(extension)


def knowleadgebase_create(summar):
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs={'device': 'cuda'})

    print(len(summar))
    knowledgeBase = FAISS.from_documents(summar, embeddings)

    knowledgeBase.save_local(DB_FAISS_PATH)

    return knowledgeBase


# pdf_paths = []
# for root, dirs, files in os.walk(DATA_PATH):
#     for file in files:
#         if file.endswith(".pdf"):
#             # Print the full path of .txt files
#             pdf_paths.append(os.path.join(root, file))

# summar = []
# success = 0
# denide = 0
# for i in pdf_paths:
#     try:
#         documents = pdf_reader(i)
#         summar += documents
#         success += 1
#     except:
#         denide += 1
#         continue

# knowledgeBase = knowleadgebase_create(summar)

embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                      model_kwargs={'device': 'cuda'})
knowledgeBase = FAISS.load_local(DB_FAISS_PATH, embeddings)


def filter_data_from_kb(knowledgeBase, query, max_retriew=10):
    docs = knowledgeBase.similarity_search_with_score(query, k=max_retriew)

    return docs


model_name = "NousResearch/Llama-2-7b-chat-hf"

llama2 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    # device_map="auto",
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    ),
    # revision='834565c23f9b28b96ccbeabe614dd906b6db551a'
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def generate_qa_prompt(context, question):
    text = 'INSTRUCTION: Answer the following question based on the given context, providing a concise and fact-based response. Look for a exact answer in the context. Should generate a complete answer.\
      If you can not find the answer from the text, say "No Answer".'

    # text += '''Consider below examples to understand the task.

    # Example Context: Microsoft ranked No. 14 in the 2022 Fortune 500 rankings of the largest United States corporations by total revenue;[3] it was the world's largest software maker by revenue as of 2022. It is considered one of the Big Five American information technology companies, alongside Alphabet (parent company of Google), Amazon, Apple, and Meta (parent company of Facebook).
    # QUESTION: What rank did Microsoft hold in the 2022 Fortune 500 rankings?
    # OUTPUT: Microsoft held the No. 14 rank in the 2022 Fortune 500 rankings.

    # Example Context: Bears are carnivoran mammals of the family Ursidae. They are classified as caniforms, or doglike carnivorans. Although only eight species of bears are extant, they are widespread, appearing in a wide variety of habitats throughout most of the Northern Hemisphere and partially in the Southern Hemisphere.
    # QUESTION: What rank did Microsoft hold in the 2022 Fortune 500 rankings?
    # OUTPUT: No answer\n
    # '''

    text += f"CONTEXT: {context}\n\n"
    text += f"QUESTION: {question}\n\n"
    text += "OUTPUT: "
    return {'text': text}


def generate_qg_prompt(context):
    text = 'INSTRUCTION: Generate straightforward, factual, reasoning-based and open-ended questions using the data available in the given context to cover all the details on it. To cover the same content, generate questions in different formats. The answers for the generated questions need to be in the context. Generate complete answers for the generated questions and give the output as JSON with question and answer pairs on it.\n\n'

    text += f"CONTEXT: {context}\n\n"
    text += "OUTPUT: "
    return {'text': text}


question_generation_kwargs = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "do_sample": True,
    # "no_repeat_ngram_size":4,
    # "repetition_penalty" : 1.5
}

llama2_QA_pipeline = pipeline(
    "text-generation",
    model=llama2,
    tokenizer=tokenizer,
    **question_generation_kwargs
)


def response_generation(query):
    filtered_text = filter_data_from_kb(knowledgeBase, query, 10)

    answers = []
    for i in range(5):
        prompt = generate_qa_prompt(
            filtered_text[i][0].page_content, query)['text']
        output_response = llama2_QA_pipeline(prompt)[0]['generated_text']
        output_text = output_response.split('\nOUTPUT:')[1]

        if "No Answer" not in output_text:
            answers.append(output_text)
            break

    if answers:
        print(output_text)
        print(filtered_text[i][0].metadata)
    else:
        print("Apologies, but it seems that I couldn't find a specific solution or answer for your query. Feel free to ask another question, and I'll do my best to assist you!")


# Chainlit
@cl.on_chat_start
async def start():
    msg = cl.Message(content="Starting the bot.....")
    await msg.send()
    msg.content = "Hi, Welcome to the IntelliScript Bot. What is your query?"
    await msg.update()


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    res = response_generation(message)
    answer = res
    # sources = res["source_documents"]

    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += f"\nNo Sources Found"

    await cl.Message(content=answer).send()
