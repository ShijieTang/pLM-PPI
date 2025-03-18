import os
import argparse
import pandas as pd
import torch
import esm
from tqdm import tqdm
import gc
import torch.distributed as dist

def setup():
    """
    Initialize the distributed process group.
    Assumes that the script is launched using torchrun or torch.distributed.launch.
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"Process initialized, local rank: {local_rank}, device: {device}")
    return local_rank, device

def cleanup():
    dist.destroy_process_group()

def load_model(device):
    """
    Load the ESM2 model and wrap it in DistributedDataParallel.
    """
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    # Wrap model with DDP. Note that each process should use a single GPU.
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Set model to evaluation mode
    return model, batch_converter

def batch_compute_embeddings(seq_list, model, batch_converter, device, batch_size=8):
    """
    Compute embeddings for a list of sequences in batches.

    Parameters:
      seq_list: A list of protein sequences (strings).
      model: The loaded model.
      batch_converter: The alphabet batch converter.
      device: The device to run inference on.
      batch_size: Number of sequences to process per batch.

    Returns:
      A list of embeddings corresponding to the input sequences.
    """
    embeddings = []
    total = len(seq_list)
    for i in tqdm(range(0, total, batch_size), desc="Batch Embedding", disable=False):
        batch_seqs = seq_list[i:i+batch_size]
        # Prepare data in the required format: list of (name, sequence)
        data = [(f"protein_{i+j}", seq) for j, seq in enumerate(batch_seqs)]
        labels, strs, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens=tokens,
                            repr_layers=[model.module.num_layers if hasattr(model, 'module') else model.num_layers],
                            return_contacts=False)
        # Determine the layer to use for representations
        if hasattr(model, 'module'):
            layer = model.module.num_layers
        else:
            layer = model.num_layers
        token_representations = results["representations"][layer]
        # For each sequence, exclude the start and end tokens and compute the mean
        for j in range(len(batch_seqs)):
            emb = token_representations[j, 1: tokens.size(1)-1].mean(0)
            embeddings.append(emb.cpu().numpy().tolist())
        # Clean up GPU memory for this batch
        del tokens, results, token_representations
        torch.cuda.empty_cache()
        gc.collect()
    return embeddings

def main():
    # Set up the distributed environment
    local_rank, device = setup()
    
    # Load model and batch converter
    model, batch_converter = load_model(device)
    
    # Read the CSV file (each process reads the entire file)
    df = pd.read_csv("./data/PPI_prediction_dataset_whole.csv")
    seqs1 = df['seq1'].tolist()
    seqs2 = df['seq2'].tolist()
    total_rows = len(df)
    
    # Split the work among processes using round-robin distribution
    indices = list(range(total_rows))
    local_indices = indices[local_rank::dist.get_world_size()]
    
    # Get local sequences based on assigned indices
    local_seqs1 = [seqs1[i] for i in local_indices]
    local_seqs2 = [seqs2[i] for i in local_indices]
    
    print(f"Rank {local_rank} processing {len(local_indices)} rows.")
    
    # Compute embeddings in batches for local seq1 and seq2
    print(f"Rank {local_rank}: Computing embeddings for seq1...")
    embeddings_seq1 = batch_compute_embeddings(local_seqs1, model, batch_converter, device, batch_size=32)
    print(f"Rank {local_rank}: Computing embeddings for seq2...")
    embeddings_seq2 = batch_compute_embeddings(local_seqs2, model, batch_converter, device, batch_size=32)
    
    # Concatenate the two embeddings for each row and keep track of the original row index
    local_embeddings = [(idx, emb1 + emb2) for idx, emb1, emb2 in zip(local_indices, embeddings_seq1, embeddings_seq2)]
    
    # Gather embeddings from all processes using all_gather_object
    gathered_embeddings = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_embeddings, local_embeddings)
    
    # Only rank 0 aggregates, reorders, and saves the results
    if local_rank == 0:
        # Flatten the list and sort by the original index
        all_embeddings = [item for sublist in gathered_embeddings for item in sublist]
        all_embeddings.sort(key=lambda x: x[0])
        concatenated_embeddings = [emb for idx, emb in all_embeddings]
        # Add the concatenated embedding as a new column to the DataFrame
        df['embedding_concat'] = concatenated_embeddings
        # Save the resulting DataFrame as a JSON file
        df.to_json("./data/PPI_prediction_dataset_whole_with_esm2_embeddings.json",
                   orient='records', force_ascii=False, indent=2)
        print("Concatenated embeddings generated and saved!")
    
    cleanup()

if __name__ == "__main__":
    main()