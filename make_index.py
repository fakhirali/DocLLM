#!/usr/bin/env python
# coding: utf-8


from PyPDF2 import PdfReader
import os
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torch.nn.functional as F
import re
import faiss
from tqdm import tqdm
import pickle as pkl
from utils import get_embeddings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

path = 'embeddings'
chunk_size = 2
overlap = 0
split_char = '.'


@torch.no_grad()
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
embeddings_model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
embeddings_model.to(device)
embeddings_model.eval()




class Chuncker:
    def __init__(self, chunck_size=3, chunk_overlap=1, split_char='.'):
        self.chunck_size = chunck_size
        self.chunk_overlap = chunk_overlap
        self.split_char = split_char

    def get_chunks(self, text):
        all_words = text.split(self.split_char)
        chunks = []
        for i in range(0, len(all_words), self.chunck_size - self.chunk_overlap):
            chunks.append('passage: ' + ' '.join(all_words[i:i + self.chunck_size]))
        return chunks


chunker = Chuncker(chunck_size=chunk_size, chunk_overlap=overlap, split_char=split_char)

index = faiss.IndexFlatL2(embeddings_model.config.hidden_size)


text_info = []


text = open('atomic_habits.txt', 'r').read()

text_chunks = chunker.get_chunks(text)
text_info.extend([x for x in text_chunks])
# The following is probably slow, should batch this
text_embeddings = np.array([get_embeddings([x], tokenizer, embeddings_model) for x in tqdm(text_chunks)]).reshape(len(text_chunks), -1)
index.add(text_embeddings)
name = "atomic_habits"
faiss.write_index(index, f'embeddings/{name}.index')

pkl.dump(text_info, open(f'embeddings/{name}_text_info.pkl', 'wb'))
