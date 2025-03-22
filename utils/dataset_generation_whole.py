import pandas as pd
import random
import requests

# Parse FASTA to build dictionary
def parse_fasta_sequences(fasta_file):
    seq_dict = {}
    current_id = None
    current_seq = []
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id:
                    seq_dict[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            seq_dict[current_id] = "".join(current_seq)
    return seq_dict

# Fetch sequence from UniProt
def fetch_sequence_from_uniprot(protein_id):
    print(f"Fetching {protein_id} from UniProt...")
    url = f"https://www.uniprot.org/uniprot/{protein_id}.fasta"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            fasta_text = response.text
            seq = "".join(line.strip() for line in fasta_text.splitlines() if not line.startswith('>'))
            return seq
        else:
            return ""
    except Exception as e:
        print(f"Error fetching {protein_id}: {e}")
        return ""

# Retrieve sequence locally or from UniProt
def get_sequence(protein_id, seq_dict):
    return seq_dict.get(protein_id, fetch_sequence_from_uniprot(protein_id))

# Process sequences and record original length
def process_sequence(seq, target_len=2048):
    original_len = len(seq)
    if original_len > target_len:
        return None, original_len
    return seq.ljust(target_len, 'X'), original_len

# Get valid protein pairs
def get_valid_pairs(candidate_pairs, seq_dict, target_count):
    valid = []
    candidate_list = list(candidate_pairs)
    random.shuffle(candidate_list)
    for protA, protB in candidate_list:
        seq1, len1 = process_sequence(get_sequence(protA, seq_dict))
        seq2, len2 = process_sequence(get_sequence(protB, seq_dict))
        if seq1 and seq2:
            valid.append((protA, protB, seq1, seq2, len1, len2))
            if len(valid) >= target_count:
                break
    return valid

# Generate negative pairs
def get_valid_negative_pairs(protein_list, positive_pairs_set, seq_dict, target_count):
    valid = []
    attempts = 0
    while len(valid) < target_count and attempts < 100000:
        protA, protB = tuple(sorted(random.sample(protein_list, 2)))
        if (protA, protB) in positive_pairs_set:
            attempts += 1
            continue
        seq1, len1 = process_sequence(get_sequence(protA, seq_dict))
        seq2, len2 = process_sequence(get_sequence(protB, seq_dict))
        if seq1 and seq2:
            valid.append((protA, protB, seq1, seq2, len1, len2))
        attempts += 1
    return valid

# File paths
biogrid_file = './BIOGRID-MV-Physical-4.4.242.tab3.txt'
fasta_file = './cafa-5-protein-function-prediction/Train/train_sequences.fasta'

# Parse CAFA5 FASTA
seq_dict = parse_fasta_sequences(fasta_file)
cafa_proteins = list(seq_dict.keys())

# Read and prepare BIOGRID data
biogrid_df = pd.read_csv(biogrid_file, sep='\t')
biogrid_df = biogrid_df[['SWISS-PROT Accessions Interactor A', 'SWISS-PROT Accessions Interactor B']]
biogrid_df = biogrid_df[~biogrid_df.isin(['-']).any(axis=1)].drop_duplicates()
positive_candidates = {tuple(sorted([a.split('|')[0], b.split('|')[0]])) for a, b in biogrid_df.values}

# Sample size
target_size = 5000

# Generate valid pairs
valid_positive = get_valid_pairs(positive_candidates, seq_dict, target_size)
valid_negative = get_valid_negative_pairs(cafa_proteins, positive_candidates, seq_dict, target_size)

# Construct DataFrames
pos_df = pd.DataFrame(valid_positive, columns=['Protein_A', 'Protein_B', 'seq1', 'seq2', 'seq1_len', 'seq2_len'])
pos_df['Label'] = 1
neg_df = pd.DataFrame(valid_negative, columns=['Protein_A', 'Protein_B', 'seq1', 'seq2', 'seq1_len', 'seq2_len'])
neg_df['Label'] = 0

# Final dataset
ppi_dataset = pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
ppi_dataset.to_csv('./pLM-PPI/data/PPI_prediction_dataset_whole.csv', index=False)

print("Final dataset shape:", ppi_dataset.shape)
print("Dataset saved with original sequence lengths.")
