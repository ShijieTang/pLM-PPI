import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import gc
import torch.distributed as dist
import numpy as np

def setup(local_rank):
    """
    Initialize the distributed process group.
    Assumes that the script is launched using torchrun or torch.distributed.launch.
    """
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"Process initialized, local rank: {local_rank}, device: {device}")
    return local_rank, device

def cleanup():
    dist.destroy_process_group()

import numpy as np

def onehot_embedding(seq):
    aa_list = "ACDEFGHIKLMNPQRSTVWYX"
    aa_to_index = {aa: i for i, aa in enumerate(aa_list)}
    seq_length = len(seq)
    aa_number = len(aa_list)

    onehot_matrix = np.zeros((seq_length, aa_number), dtype=np.float32)
    
    for i, char in enumerate(seq):
        if char in aa_to_index:
            onehot_matrix[i, aa_to_index[char]] = 1.0

    df = pd.DataFrame(onehot_matrix, columns=list(aa_list))
    return df.values


def positional_encoding(seq_len, d_model):
    """
    Compute the sinusoidal positional encoding for a sequence of length seq_len.
    Returns an array of shape (seq_len, d_model).
    """
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    # Apply sin to even indices and cos to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads

def positional_embedding(seq, d_model=128):
    """
    Compute a fixed-length positional embedding for a protein sequence.
    This implementation computes the sinusoidal encoding for each position and averages them.
    """
    seq_len = len(seq)
    pos_enc = positional_encoding(seq_len, d_model)
    avg_enc = pos_enc.mean(axis=0)
    return avg_enc.tolist()

def compute_combined_embeddings(seq_list, embedding_types, pos_dim):
    """
    Compute embeddings for a list of sequences based on the selected types.
    Supported types (case-insensitive): "onehot", "positional".
    When multiple types are specified, their computed embeddings are concatenated.
    """
    onehot_embeds = None
    positional_embeds = None
    
    if "onehot" in embedding_types:
        onehot_embeds = [onehot_embedding(seq) for seq in seq_list]
    if "positional" in embedding_types:
        positional_embeds = [positional_embedding(seq, d_model=pos_dim) for seq in seq_list]
    
    combined_embeddings = []
    for i in range(len(seq_list)):
        emb = []
        if onehot_embeds is not None:
            emb.extend(onehot_embeds[i])
        if positional_embeds is not None:
            emb.extend(positional_embeds[i])
        combined_embeddings.append(emb)
    return combined_embeddings

def main():
    parser = argparse.ArgumentParser()
    # Add local rank argument to accept the parameter passed by the distributed launcher.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0,
                        help="Local rank provided by distributed launcher")
    parser.add_argument("--embedding_type", type=str, default="onehot,positional",
                        help="Comma-separated embedding types to use. Options: onehot, positional")
    parser.add_argument("--pos_dim", type=int, default=128, help="Dimension for positional embedding if used")
    args, unknown = parser.parse_known_args()  # Allow unknown arguments if any
    
    # Normalize embedding type names to lower case and split by comma
    embedding_types = [x.strip().lower() for x in args.embedding_type.split(",")]
    
    # Set up the distributed environment using the provided local rank
    local_rank, device = setup(args.local_rank)
    
    # Read the CSV file (each process reads the entire file)
    df = pd.read_csv("./data/PPI_prediction_dataset_whole.csv")
    seqs1 = df['seq1'].tolist()
    seqs2 = df['seq2'].tolist()
    total_rows = len(df)
    
    # Split the work among processes using round-robin distribution
    indices = list(range(total_rows))
    local_indices = indices[args.local_rank::dist.get_world_size()]
    
    # Get local sequences based on assigned indices
    local_seqs1 = [seqs1[i] for i in local_indices]
    local_seqs2 = [seqs2[i] for i in local_indices]
    
    print(f"Rank {args.local_rank} processing {len(local_indices)} rows.")
    
    # Compute combined embeddings for local seq1 and seq2 based on the chosen embedding types
    print(f"Rank {args.local_rank}: Computing embeddings for seq1...")
    embeddings_seq1 = compute_combined_embeddings(local_seqs1, embedding_types, args.pos_dim)
    print(f"Rank {args.local_rank}: Computing embeddings for seq2...")
    embeddings_seq2 = compute_combined_embeddings(local_seqs2, embedding_types, args.pos_dim)
    
    # Concatenate the two embeddings for each row and keep track of the original row index
    local_embeddings = [(idx, emb1 + emb2) for idx, emb1, emb2 in zip(local_indices, embeddings_seq1, embeddings_seq2)]
    
    # Gather embeddings from all processes using all_gather_object
    gathered_embeddings = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_embeddings, local_embeddings)
    
    # Only rank 0 aggregates, reorders, and saves the results
    if args.local_rank == 0:
        # Flatten the list and sort by the original index
        all_embeddings = [item for sublist in gathered_embeddings for item in sublist]
        all_embeddings.sort(key=lambda x: x[0])
        concatenated_embeddings = [emb for idx, emb in all_embeddings]
        # Add the concatenated embedding as a new column to the DataFrame
        df['embedding_concat'] = concatenated_embeddings
        # Save the resulting DataFrame as a JSON file
        df.to_json("./data/PPI_prediction_dataset_whole_onehot.json",
                   orient='records', force_ascii=False, indent=2)
        print("Concatenated embeddings generated and saved!")
    
    cleanup()

if __name__ == "__main__":
    main()