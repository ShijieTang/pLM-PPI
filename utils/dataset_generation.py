import pandas as pd
import random
import requests

# 1. 解析FASTA文件，构建蛋白ID到序列的字典
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
                # 假设ID为">"后第一个单词
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            seq_dict[current_id] = "".join(current_seq)
    return seq_dict

# 2. 根据蛋白ID从 UniProt 在线获取FASTA序列
def fetch_sequence_from_uniprot(protein_id):
    print(f"Fetching {protein_id} from UniProt...")
    url = f"https://www.uniprot.org/uniprot/{protein_id}.fasta"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            fasta_text = response.text
            # 拼接除描述行外的所有序列行
            seq = "".join(line.strip() for line in fasta_text.splitlines() if not line.startswith('>'))
            return seq
        else:
            return ""
    except Exception as e:
        print(f"Error fetching {protein_id}: {e}")
        return ""

# 3. 获取蛋白序列，先从本地字典查找，若不存在则尝试在线获取
def get_sequence(protein_id, seq_dict):
    if protein_id in seq_dict and seq_dict[protein_id]:
        return seq_dict[protein_id]
    else:
        return fetch_sequence_from_uniprot(protein_id)

# 新增：对蛋白序列进行截断或填充，确保长度为2048
def process_sequence(seq, target_len=2048):
    if len(seq) > target_len:
        # 取中间的target_len个氨基酸
        start = (len(seq) - target_len) // 2
        return seq[start:start+target_len]
    elif len(seq) < target_len:
        # 填充"X"到target_len
        return seq + "X" * (target_len - len(seq))
    else:
        return seq

# 设置文件路径
biogrid_file = './BIOGRID-MV-Physical-4.4.242.tab3.txt'
fasta_file = './cafa-5-protein-function-prediction/Train/train_sequences.fasta'

# 4. 解析CAFA5的FASTA文件，得到蛋白序列字典和蛋白ID列表
seq_dict = parse_fasta_sequences(fasta_file)
cafa_proteins = list(seq_dict.keys())
print("The number of proteins in CAFA5:", len(cafa_proteins))

# 5. 读取BIOGRID文件（制表符分隔，跳过以'#'开头的注释行）
biogrid_df = pd.read_csv(biogrid_file, sep='\t', header=0)

# 6. 选择构建正样本所需的列，这里假设使用“SWISS-PROT Accessions Interactor A/B”
biogrid_df = biogrid_df[['SWISS-PROT Accessions Interactor A', 'SWISS-PROT Accessions Interactor B']]
biogrid_df = biogrid_df[(biogrid_df['SWISS-PROT Accessions Interactor A'] != '-') & 
                        (biogrid_df['SWISS-PROT Accessions Interactor B'] != '-')]
biogrid_df = biogrid_df.drop_duplicates()

# 7. 构建正样本集合，确保蛋白对排序后唯一（避免A-B与B-A重复）
positive_pairs = set()
for idx, row in biogrid_df.iterrows():
    protA = row['SWISS-PROT Accessions Interactor A'].split('|')[0]
    protB = row['SWISS-PROT Accessions Interactor B'].split('|')[0]
    pair = tuple(sorted([protA, protB]))
    positive_pairs.add(pair)
print("Positive sample number in BIOGRID:", len(positive_pairs))

# 8. 设置目标样本数（正负各5000个）
target_size = 5000
if len(positive_pairs) >= target_size:
    positive_pairs_sample = random.sample(list(positive_pairs), target_size)
else:
    raise ValueError(f"正样本数量不足 {target_size}，请检查数据！")

# 9. 定义生成负样本的函数，从CAFA5蛋白列表中随机采样蛋白对，保证不在正样本中
def generate_negative_samples(protein_list, positive_pairs_set, sample_size):
    negative_samples = set()
    while len(negative_samples) < sample_size:
        pair = tuple(sorted(random.sample(protein_list, 2)))
        if pair not in positive_pairs_set and pair not in negative_samples:
            negative_samples.add(pair)
    return list(negative_samples)

negative_pairs = generate_negative_samples(cafa_proteins, set(positive_pairs_sample), target_size)
print("Negative sample number:", len(negative_pairs))

# 10. 构建DataFrame，正样本Label=1，负样本Label=0
positive_df = pd.DataFrame(positive_pairs_sample, columns=['Protein_A', 'Protein_B'])
positive_df['Label'] = 1
negative_df = pd.DataFrame(negative_pairs, columns=['Protein_A', 'Protein_B'])
negative_df['Label'] = 0
ppi_dataset = pd.concat([positive_df, negative_df]).sample(frac=1).reset_index(drop=True)

# 11. 添加序列信息：若在本地未找到，则尝试在线获取，并对序列进行处理（截断/填充至2048）
ppi_dataset['seq1'] = ppi_dataset['Protein_A'].apply(lambda x: process_sequence(get_sequence(x, seq_dict)))
ppi_dataset['seq2'] = ppi_dataset['Protein_B'].apply(lambda x: process_sequence(get_sequence(x, seq_dict)))

print(ppi_dataset.head())

# 12. 保存最终数据集到CSV文件
ppi_dataset.to_csv('./pLM-PPI/data/PPI_prediction_dataset_2000_with_seq.csv', index=False)