import os
import pandas as pd
import torch
import esm
from tqdm import tqdm
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, "with", torch.cuda.device_count(), "GPUs")

# Load ESM2 model
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()

if torch.cuda.device_count() > 1:
    print("Using DataParallel with", torch.cuda.device_count(), "GPUs")
    model = torch.nn.DataParallel(model)

model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()

def batch_compute_embeddings(seq_list, batch_size=8):
    embeddings = []
    total = len(seq_list)
    for i in tqdm(range(0, total, batch_size), desc="Batch Embedding"):
        batch_seqs = seq_list[i:i+batch_size]
        data = [(f"protein_{i+j}", seq) for j, seq in enumerate(batch_seqs)]
        labels, strs, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens=tokens,
                            repr_layers=[model.module.num_layers if isinstance(model, torch.nn.DataParallel) 
                                         else model.num_layers],
                            return_contacts=False)
        layer = model.module.num_layers if isinstance(model, torch.nn.DataParallel) else model.num_layers
        token_representations = results["representations"][layer]
        for j in range(len(batch_seqs)):
            emb = token_representations[j, 1: tokens.size(1)-1].mean(0)
            embeddings.append(emb.cpu().numpy().tolist())
        del tokens, results, token_representations
        torch.cuda.empty_cache()
        gc.collect()
    return embeddings

# 读取数据
df = pd.read_csv("./data/PPI_prediction_dataset_whole.csv")

# 根据已有长度信息，去除padding提取真实序列
seqs1_real = [seq[:l] for seq, l in zip(df['seq1'], df['seq1_len'])]
seqs2_real = [seq[:l] for seq, l in zip(df['seq2'], df['seq2_len'])]

# 提取embedding
print("Computing embeddings for seq1 (real)...")
embeddings_seq1 = batch_compute_embeddings(seqs1_real, batch_size=8)
print("Computing embeddings for seq2 (real)...")
embeddings_seq2 = batch_compute_embeddings(seqs2_real, batch_size=8)

# 合并embedding
concatenated_embeddings = [emb1 + emb2 for emb1, emb2 in zip(embeddings_seq1, embeddings_seq2)]

# 将合并后的embedding加入DataFrame
df['embedding_concat'] = concatenated_embeddings

# 保存结果到json
df.to_json("./data/PPI_prediction_dataset_whole_with_esm2_35.json", orient='records', force_ascii=False, indent=2)

print("Concatenated embeddings based on real sequences generated and saved!")
