import pandas as pd
import random
import requests

# 1. Parse the FASTA file to build a dictionary mapping protein ID to its sequence.
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
                if current_id is not None:
                    seq_dict[current_id] = "".join(current_seq)
                # Assume the first word after '>' is the protein ID.
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            seq_dict[current_id] = "".join(current_seq)
    return seq_dict

# 2. Fetch sequence from UniProt using protein ID if not available locally.
def fetch_sequence_from_uniprot(protein_id):
    print(f"Fetching {protein_id} from UniProt...")
    url = f"https://www.uniprot.org/uniprot/{protein_id}.fasta"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            fasta_text = response.text
            # Concatenate all lines except the description line.
            seq = "".join(line.strip() for line in fasta_text.splitlines() if not line.startswith('>'))
            return seq
        else:
            return ""
    except Exception as e:
        print(f"Error fetching {protein_id}: {e}")
        return ""

# 3. Retrieve the sequence from the local dictionary; if not present, fetch online.
def get_sequence(protein_id, seq_dict):
    if protein_id in seq_dict and seq_dict[protein_id]:
        return seq_dict[protein_id]
    else:
        return fetch_sequence_from_uniprot(protein_id)

# 4. Process the sequence:
#    - If the sequence length is greater than target_len (2048), return None (i.e. filter it out).
#    - If the sequence length is less than target_len, pad it with "X" to reach target_len.
#    - Otherwise, return the sequence as is.
def process_sequence(seq, target_len=2048):
    if len(seq) > target_len:
        # Instead of truncating, we filter out sequences that exceed the target length.
        return None
    elif len(seq) < target_len:
        return seq + "X" * (target_len - len(seq))
    else:
        return seq

# Set file paths.
biogrid_file = './BIOGRID-MV-Physical-4.4.242.tab3.txt'
fasta_file = './cafa-5-protein-function-prediction/Train/train_sequences.fasta'

# 5. Parse the CAFA5 FASTA file to obtain the protein sequence dictionary.
seq_dict = parse_fasta_sequences(fasta_file)
cafa_proteins = list(seq_dict.keys())
print("The number of proteins in CAFA5:", len(cafa_proteins))

# 6. Read the BIOGRID file (tab-separated, with header).
biogrid_df = pd.read_csv(biogrid_file, sep='\t', header=0)

# 7. Select the columns needed for positive samples (using "SWISS-PROT Accessions Interactor A/B").
biogrid_df = biogrid_df[['SWISS-PROT Accessions Interactor A', 'SWISS-PROT Accessions Interactor B']]
biogrid_df = biogrid_df[(biogrid_df['SWISS-PROT Accessions Interactor A'] != '-') & 
                        (biogrid_df['SWISS-PROT Accessions Interactor B'] != '-')]
biogrid_df = biogrid_df.drop_duplicates()

# 8. Build the candidate positive pairs, ensuring uniqueness by sorting the pair (to avoid A-B vs. B-A duplicates).
positive_candidates = set()
for idx, row in biogrid_df.iterrows():
    protA = row['SWISS-PROT Accessions Interactor A'].split('|')[0]
    protB = row['SWISS-PROT Accessions Interactor B'].split('|')[0]
    pair = tuple(sorted([protA, protB]))
    positive_candidates.add(pair)
print("Candidate positive pairs in BIOGRID:", len(positive_candidates))

# Set target sample count for both positive and negative pairs.
target_size = 5000

# 9. Function to filter valid pairs from a candidate set.
def get_valid_pairs(candidate_pairs, seq_dict, target_count):
    valid = []
    candidate_list = list(candidate_pairs)
    random.shuffle(candidate_list)
    for pair in candidate_list:
        protA, protB = pair
        seq1 = process_sequence(get_sequence(protA, seq_dict))
        seq2 = process_sequence(get_sequence(protB, seq_dict))
        if seq1 is not None and seq2 is not None:
            valid.append((protA, protB, seq1, seq2))
            if len(valid) >= target_count:
                break
    return valid

# 10. Obtain valid positive pairs.
valid_positive = get_valid_pairs(positive_candidates, seq_dict, target_size)
if len(valid_positive) < target_size:
    raise ValueError(f"Not enough valid positive pairs. Required {target_size}, got {len(valid_positive)}")
print("Valid positive pairs count:", len(valid_positive))

# 11. Function to generate valid negative pairs.
def get_valid_negative_pairs(protein_list, positive_pairs_set, seq_dict, target_count):
    valid = []
    candidate_set = set()
    # Loop until we have enough valid negative pairs.
    while len(valid) < target_count:
        # Generate a batch of candidate negative pairs.
        new_candidates = set()
        for _ in range(target_count):
            pair = tuple(sorted(random.sample(protein_list, 2)))
            # Exclude pairs that are in the positive set or already generated.
            if pair not in positive_pairs_set and pair not in candidate_set:
                new_candidates.add(pair)
        candidate_set.update(new_candidates)
        candidate_list = list(new_candidates)
        random.shuffle(candidate_list)
        for pair in candidate_list:
            protA, protB = pair
            seq1 = process_sequence(get_sequence(protA, seq_dict))
            seq2 = process_sequence(get_sequence(protB, seq_dict))
            if seq1 is not None and seq2 is not None:
                valid.append((protA, protB, seq1, seq2))
                if len(valid) >= target_count:
                    break
        # If candidate_set grows too large without enough valid pairs, break to avoid infinite loop.
        if len(candidate_set) > 100000 and len(valid) < target_count:
            break
    return valid

# For negative pairs, use the entire CAFA5 protein list as candidate pool.
valid_negative = get_valid_negative_pairs(cafa_proteins, positive_candidates, seq_dict, target_size)
if len(valid_negative) < target_size:
    raise ValueError(f"Not enough valid negative pairs. Required {target_size}, got {len(valid_negative)}")
print("Valid negative pairs count:", len(valid_negative))

# 12. Construct DataFrames for positive and negative samples.
pos_df = pd.DataFrame(valid_positive, columns=['Protein_A', 'Protein_B', 'seq1', 'seq2'])
pos_df['Label'] = 1
neg_df = pd.DataFrame(valid_negative, columns=['Protein_A', 'Protein_B', 'seq1', 'seq2'])
neg_df['Label'] = 0

# Combine and shuffle the DataFrames.
ppi_dataset = pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
print("Final dataset shape:", ppi_dataset.shape)

# 13. Save the final dataset to a CSV file.
ppi_dataset.to_csv('./pLM-PPI/data/PPI_prediction_dataset_whole.csv', index=False)
print("Dataset saved.")