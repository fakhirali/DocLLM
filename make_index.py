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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

path = 'ai_papers/'
chunk_size = 5
overlap = 2
split_char = '\n\n'


@torch.no_grad()
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
embeddings_model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
embeddings_model.to(device)
embeddings_model.eval()


@torch.no_grad()
def get_embeddings(texts):
    # Tokenize the input texts
    batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)

    outputs = embeddings_model(**batch_dict)
    embeddings = outputs[0][:, 0]
    # embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).to('cpu')

    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


class Chuncker:
    def __init__(self, chunck_size=3, chunk_overlap=1, split_char='.'):
        # chunk_size is in words (after .split())
        self.chunck_size = chunck_size
        self.chunk_overlap = chunk_overlap

    def get_chunks(self, text):
        all_words = text.split('.')
        chunks = []
        for i in range(0, len(all_words), self.chunck_size - self.chunk_overlap):
            chunks.append('passage: ' + ' '.join(all_words[i:i + self.chunck_size]))
        return chunks


chunker = Chuncker(chunck_size=chunk_size, chunk_overlap=overlap, split_char=split_char)

index = faiss.IndexFlatL2(embeddings_model.config.hidden_size)

pdf_files = [x for x in os.listdir(path) if 'pdf' in x]

text_info = []

for file in tqdm(pdf_files):
    reader = PdfReader(path + file)
    text = ""
    for page in reader.pages:
        p_text = page.extract_text()
        # Merge hyphenated words
        p_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", p_text)
        # Fix newlines in the middle of sentences
        p_text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", p_text.strip())
        # Remove multiple newlines
        p_text = re.sub(r"\n\s*\n", "\n\n", p_text)
        text += p_text + '\n'

    text_chunks = chunker.get_chunks(text)
    text_info.extend([(file, x) for x in text_chunks])
    # The following is probably slow, should batch this
    text_embeddings = np.array([get_embeddings([x]) for x in text_chunks]).reshape(len(text_chunks), -1)
    index.add(text_embeddings)

faiss.write_index(index, f'embeddings/{path.replace("/", "_").strip("_")}.index')

pkl.dump(text_info, open(f'embeddings/{path.replace("/", "_").strip("_")}_text_info.pkl', 'wb'))
