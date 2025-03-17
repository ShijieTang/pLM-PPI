import os
import pandas as pd
import torch
import esm
from tqdm import tqdm
import gc

# Optional: restrict usage to specific GPUs if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Set the GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, "with", torch.cuda.device_count(), "GPUs")

# Load the ESM2 model (e.g., esm2_t33_650M_UR50D)
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()

# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

# If multiple GPUs are available, wrap the model with DataParallel
if torch.cuda.device_count() > 1:
    print("Using DataParallel with", torch.cuda.device_count(), "GPUs")
    model = torch.nn.DataParallel(model)
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()  # Set model to evaluation mode

def batch_compute_embeddings(seq_list, batch_size=8):
    """
    Compute embeddings for a list of sequences in batches.
    
    Parameters:
      seq_list: A list of protein sequences (strings).
      batch_size: Number of sequences to process per batch.
    
    Returns:
      A list of embeddings corresponding to the input sequences.
    """
    embeddings = []
    total = len(seq_list)
    for i in tqdm(range(0, total, batch_size), desc="Batch Embedding"):
        batch_seqs = seq_list[i:i+batch_size]
        # Prepare data in the required format: list of (name, sequence)
        data = [(f"protein_{i+j}", seq) for j, seq in enumerate(batch_seqs)]
        labels, strs, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens=tokens,
                            repr_layers=[model.module.num_layers if isinstance(model, torch.nn.DataParallel) 
                                           else model.num_layers],
                            return_contacts=False)
        # Determine which layer to use for representations
        if isinstance(model, torch.nn.DataParallel):
            layer = model.module.num_layers
        else:
            layer = model.num_layers
        token_representations = results["representations"][layer]
        # For each sequence, exclude the start and end tokens and compute the mean
        for j in range(len(batch_seqs)):
            emb = token_representations[j, 1: tokens.size(1)-1].mean(0)
            embeddings.append(emb.cpu().numpy().tolist())
        # Release GPU memory for this batch
        del tokens, results, token_representations
        torch.cuda.empty_cache()
        gc.collect()
    return embeddings

# Read the CSV file (make sure the file contains 'seq1' and 'seq2' columns)
df = pd.read_csv("./data/PPI_prediction_dataset_2000_with_seq.csv")

# Get the list of sequences from each column
seqs1 = df['seq1'].tolist()
seqs2 = df['seq2'].tolist()

# Compute embeddings in batches for seq1 and seq2
print("Computing embeddings for seq1...")
embeddings_seq1 = batch_compute_embeddings(seqs1, batch_size=8)
print("Computing embeddings for seq2...")
embeddings_seq2 = batch_compute_embeddings(seqs2, batch_size=8)

# Concatenate the two embeddings for each row (assuming both are lists)
concatenated_embeddings = [emb1 + emb2 for emb1, emb2 in zip(embeddings_seq1, embeddings_seq2)]

# Add the concatenated embedding as a new column to the DataFrame
df['embedding_concat'] = concatenated_embeddings

# Optionally, you can drop the individual embedding columns if not needed
df.drop(columns=['embedding1', 'embedding2'], inplace=True)

# Save the resulting DataFrame as a JSON file
df.to_json("./data/PPI_prediction_dataset_with_esm2_embeddings.json", orient='records', force_ascii=False, indent=2)

print("Concatenated embeddings generated and saved!")