#!/usr/bin/env python
# coding: utf-8


from pypdf import PdfReader
import os
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from Agent import Agent
import re
import faiss
import pickle as pkl
import gradio as gr

path = 'ai_papers/'
@torch.no_grad()
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
embeddings_model = AutoModel.from_pretrained('intfloat/e5-base-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
embeddings_model.to(device)
embeddings_model.eval()
pass

# for 'intfloat/e5-base-v2'
# Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.




model_name = 'databricks/dolly-v2-3b'
gen_model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                            low_cpu_mem_usage=True,
                                                 trust_remote_code=True)
gen_tokenizer = AutoTokenizer.from_pretrained(model_name)
gen_model.to(device)
pass

@torch.no_grad()
def get_embeddings(texts):
    # Tokenize the input texts
    batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)

    outputs = embeddings_model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).to('cpu')

    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings)
    return embeddings.numpy()




prompt = f'''
Below is a user query that describes a question. Write a response that appropriately answers the query using the 
information given in the input. The information is extracted from a research paper.



'''


agent = Agent(gen_model, gen_tokenizer, prompt,
              break_words=['### End'], device=device)


index = faiss.read_index(f'embeddings/{path.replace("/","_").strip("_")}.index')
text_info = pkl.load(open(f'embeddings/{path.replace("/", "_").strip("_")}_text_info.pkl', 'rb'))

def get_answer(query, use_information, k, temp):

    query = 'query: ' + query
    if use_information:
        query_embedding = get_embeddings([query])
        scores, text_idx = index.search(query_embedding,k)
        text_idx = text_idx.flatten()
        info = '\n'.join(np.array(text_info)[text_idx][:,1])
        info = info.replace('passage: ', '').strip()
        gen_text = f'### Instruction:\n{query.replace("query: ", "")}\n\nInput:\n{info}'
    else:
        info = ''
        gen_text = f'### Instruction:\n{query.replace("query: ", "")}'
        
    ans = agent.generate_response_greedy(gen_text,
                                verbose=True, temp=temp,name='### Response:',max_length=512)
    print('\nInformation: \n\n', info)
    return ans + '\n\nInformation: \n\n' + info + '\n\n'

demo = gr.Interface(
    fn=get_answer,
    inputs=[gr.Textbox(lines=3, placeholder="User Query"),
            gr.Checkbox(value=True),gr.Slider(1, 10, step=1, default=3), gr.Slider(0,1,step=0.1, default=0.3)],
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





