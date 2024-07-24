from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B')
model = AutoModel.from_pretrained('Qwen/Qwen1.5-0.5B')

def generate_embeddings(txt, tokenizer, model):
    input = tokenizer(txt, return_tensors = 'pt', padding = True, truncation = True)
    with torch.no_grad():
        output = model(**input)
    embeddings= output.last_hidden_state.mean(dim=1).numpy()
    return embeddings
