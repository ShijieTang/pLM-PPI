import os
import pandas as pd
import torch
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import re
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, "with", torch.cuda.device_count(), "GPUs")

# Load ESM2 model
model = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
model.full() if device=='cpu' else model.half()

if torch.cuda.device_count() > 1:
    print("Using DataParallel with", torch.cuda.device_count(), "GPUs")
    model = torch.nn.DataParallel(model)

model = model.to(device)

def batch_compute_embeddings(seq_list, batch_size=2):
    embeddings = []
    total = len(seq_list)
    for i in tqdm(range(0, total, batch_size), desc="Batch Embedding"):
        batch_seqs = seq_list[i:i+batch_size]
        data = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in batch_seqs]
        data = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s # this expects 3Di sequences to be already lower-case
                      for s in data]
        ids = tokenizer.batch_encode_plus(data,
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt').to(device)
        with torch.no_grad():
            results = model(
              ids.input_ids, 
              attention_mask=ids.attention_mask
              )
        for j in range(len(batch_seqs)):
            lenth = len(batch_seqs[j])
            emb = results.last_hidden_state[j,0:lenth+1]
            embeddings.append(emb.cpu().numpy().tolist())

        del results, ids
        torch.cuda.empty_cache()
        gc.collect()
    return embeddings

# raed data
print("Loading data...")
df = pd.read_csv("./data/PPI_prediction_dataset_whole.csv")

# 根据已有长度信息，去除padding提取真实序列
seqs1_real = [seq[:l] for seq, l in zip(df['seq1'], df['seq1_len'])]
seqs2_real = [seq[:l] for seq, l in zip(df['seq2'], df['seq2_len'])]

# extract embedding
print("Computing embeddings for seq1 (real)...")
embeddings_seq1 = batch_compute_embeddings(seqs1_real, batch_size=8)
print("Computing embeddings for seq2 (real)...")
embeddings_seq2 = batch_compute_embeddings(seqs2_real, batch_size=8)

# mergeing embedding
concatenated_embeddings = [emb1 + emb2 for emb1, emb2 in zip(embeddings_seq1, embeddings_seq2)]

# add embeddings into DataFrame
df['embedding_concat'] = concatenated_embeddings

# save the result as json
df.to_json("./data/PPI_prediction_dataset_whole_with_prostT5.json", orient='records', force_ascii=False, indent=2)

print("Concatenated embeddings based on real sequences generated and saved!")
