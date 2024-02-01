import torch
import torch.nn.functional as F



@torch.no_grad()
def get_embeddings(texts, tokenizer, embeddings_model, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Tokenize the input texts
    batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)

    outputs = embeddings_model(**batch_dict)
    embeddings = outputs[0][:, 0]
    # embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).to('cpu')

    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

