# PDF Assistant

Using llms to chat with pdfs.


### Approach
- [x] Parse the pdfs (Using [pypdf2](https://github.com/py-pdf/pypdf))
- [ ] Improve parsing (Remove headers, footers, page numbers etc)
- [x] Create embeddings of the text chunks using a good embeddings model (Using [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2))
- [x] Store the embeddings in a Vector Store (Using [faiss](https://github.com/facebookresearch/faiss))
- [ ] Use the chat history and user question to create a proper question
    - The question must contain all the information needed to answer the question
- [ ] Reflexion, CoT etc
- [x] Embed the proper question
- [x] Compute top-k text chunks from the Vector store
- [x] Use these chunks in the pre prompt of the llm to answer the question (Using [databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b))
- [x] front-end ([Implemented](/chat.py) in [gradio](https://gradio.app/))
- [ ] Show from where the answer was extracted
- [ ] Faster inference (ggml, quantization)
- [ ] Better models


### Applications
- Research Assistant
    - Semantic search
    - Aggregated answers
    - Explaining text
    - What other papers have similar piece of text?
    - Summarize Sections
- Financial Analyst
    - Applied on Financial Documents like Quarterly Reports, Earnings Call Transcripts etc
    - Augmented with a database
- Auto Journalist
    - News articles
    - Discovery



Notes:  
In total two models will be used. One for the embeddings and one for text generation.
Can a single Transformer be used?
A transformer does have an encoder that "encodes" the text



