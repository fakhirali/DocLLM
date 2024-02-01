#!/usr/bin/env python
# coding: utf-8


from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import faiss
import pickle as pkl
import gradio as gr
from utils import get_embeddings
from llm import LLM

name = 'atomic_habits'

@torch.no_grad()
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
embeddings_model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
embeddings_model.to(device)
embeddings_model.eval()
pass


chatbot = LLM()
index = faiss.read_index(f'embeddings/{name}.index')
text_info = pkl.load(open(f'embeddings/{name}_text_info.pkl', 'rb'))


def get_answer(query, k):
    query = query
    query_embedding = get_embeddings([query], tokenizer, embeddings_model)
    scores, text_idx = index.search(query_embedding, k)
    text_idx = text_idx.flatten()
    info = '\n--------\n'.join(np.array(text_info)[text_idx])
    info = info.replace('passage: ', '').strip()
    ans = chatbot.get_response(query, info)
    return ans + '\n-------------Information: \n' + info + '\n---------\n'


demo = gr.Interface(
    fn=get_answer,
    inputs=[gr.Textbox(lines=3, placeholder="User Query"),
            gr.Slider(1, 10, step=1, value=3)],
    outputs="text",
)
demo.launch()

'''
print('Enter a query to get a response. Enter "exit" to exit.')
while True:
    print()
    query = input()
    print()
    if query == 'exit':
        break
    query = 'query: ' + query

    query_embedding = get_embeddings([query])

    k = 3
    scores, text_idx = index.search(query_embedding,k)
    text_idx = text_idx.flatten()


    info = '.'.join(np.array(text_info)[text_idx][:,1])
    info = info.replace('passage: ', '').strip()
    tokens = gen_tokenizer('Human: What is your name?\n Assistant: ', return_tensors='pt')


    gen_text = f'### Instruction:\n{query.replace("query: ", "")}\n\nInput:\n{info}'

    agent.generate_response_greedy(gen_text,
                                verbose=True, temp=0.2,name='### Response:',max_length=1024)

    print(f'\nInformation: \n\n {info}')

'''
