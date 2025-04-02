import argparse
import pandas as pd
import numpy as np

RES2ID = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6,
          'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13,
          'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, '-': 20}


def onehot_emb_fetch(seq):
    df = pd.DataFrame(np.zeros((len(seq), len(RES2ID))), columns=RES2ID.keys())
    for i, aa in enumerate(seq):
        aa = aa if aa in RES2ID else '-'
        df.iloc[i, RES2ID[aa]] = 1
    return df.values


def positional_emb_fetch(seq, pos_dim=6500):
    arr = [RES2ID.get(aa, 20) for aa in seq]
    return np.array(arr)


def compute_combined_embeddings(seq_list, embedding_types, pos_dim):
    combined_embeddings = []
    for seq in seq_list:
        emb = []
        if "onehot" in embedding_types:
            emb.extend(onehot_emb_fetch(seq))
        if "positional" in embedding_types:
            emb.extend(positional_emb_fetch(seq, pos_dim))
        combined_embeddings.append(emb)
    return combined_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_type", type=str, default="onehot,positional",
                        help="Comma-separated embedding types to use. Options: onehot, positional")
    parser.add_argument("--pos_dim", type=int, default=480, help="Dimension for positional embedding")
    parser.add_argument("-o", type=str, default="./output.json", help="Output path")
    args = parser.parse_args()

    embedding_types = [x.strip().lower() for x in args.embedding_type.split(",")]

    df = pd.read_csv("./data/PPI_prediction_dataset_whole.csv")
    seqs1, seqs2 = df['seq1'].tolist(), df['seq2'].tolist()

    embeddings_seq1 = compute_combined_embeddings(seqs1, embedding_types, args.pos_dim)
    embeddings_seq2 = compute_combined_embeddings(seqs2, embedding_types, args.pos_dim)
    
    for type in embedding_types:
        if type == "positional":
            concatenated_embeddings = [emb1 + emb2 for emb1, emb2 in zip(embeddings_seq1, embeddings_seq2)]
        if type == "onehot":
            concatenated_embeddings = [np.concatenate([emb1, emb2], axis=0) for emb1, emb2 in zip(embeddings_seq1, embeddings_seq2)]

    df['embedding_concat'] = concatenated_embeddings
    df.to_json(args.o, orient='records', force_ascii=False, indent=2)
    print("Concatenated embeddings generated and saved!")


if __name__ == "__main__":
    main()
