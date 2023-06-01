# PDF Assistant

Using llms to chat with pdfs.

### Approach
- [x] Parse the pdfs
- [x] Create embeddings of the text chunks using a good embeddings model
- [ ] Store the embeddings in a Vector Store
- [ ] Use the chat history and user question to create a proper question
    - The question must contain all the information (May be Reflection at this step)
- [ ] Embed the proper question
- [ ] Compute top-k text chunks from the Vector store
- [x] Use these chunks in the pre prompt of the llm to answer the question
    - Pre-prompt = "{Information} {Question}"
- [ ] front-end (streamlit is too ez, maybe react)


### Applications
- Research Assistant
    - Reading Research papers
    - Semantic search
    - Aggregated answers
    - Explaining text
    - What other papers have similar piece of text?
    - Summarize Sections


Notes:  
In total two models will be used. One for the embeddings and one for text generation.

Can a single Transformer be used?
A transformer does have an encoder that "encodes" the text



