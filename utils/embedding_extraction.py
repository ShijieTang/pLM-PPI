import pandas as pd
import torch
import esm
from tqdm import tqdm
import gc

# 设置GPU设备（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载ESM2模型（例如 esm2_t33_650M_UR50D）并移动到GPU
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()  # 切换为评估模式

def compute_batch_embeddings(batch_data, batch_size=32):
    """
    批量计算ESM2 embedding，每个batch计算完后及时将GPU内存释放。
    参数：
      batch_data: [(name, sequence), ...] 列表，其中name用于标识序列；
      batch_size: 每批处理的序列数。
    返回：
      一个字典，将序列字符串映射到其embedding（列表格式）。
    """
    embedding_dict = {}
    for i in tqdm(range(0, len(batch_data), batch_size), desc="Batch Embedding"):
        batch = batch_data[i:i+batch_size]
        labels, strs, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
        token_representations = results["representations"][model.num_layers]
        # 对于每个序列，排除首尾特殊token后对token表示求均值作为全局embedding
        for j, (name, seq) in enumerate(batch):
            rep = token_representations[j, 1: tokens.size(1)-1].mean(0)
            # 将结果移回CPU并保存
            embedding_dict[seq] = rep.cpu().numpy().tolist()
        # 释放当前batch使用的变量，清空GPU内存
        del tokens, results, token_representations
        torch.cuda.empty_cache()
        gc.collect()
    return embedding_dict

# 1. 读取CSV文件
df = pd.read_csv("/content/PPI_prediction_dataset_2000_with_seq.csv")

# 2. 收集所有唯一的蛋白序列（seq1 和 seq2），过滤空字符串
all_seqs = set(df['seq1'].dropna().tolist() + df['seq2'].dropna().tolist())
all_seqs = [s for s in all_seqs if s.strip() != ""]

# 3. 构建批处理数据，使用序列本身作为名称（确保名称唯一即可）
batch_data = [(seq, seq) for seq in all_seqs]

# 4. 批量计算embedding，注意可以根据GPU内存情况调整batch_size
embeddings = compute_batch_embeddings(batch_data, batch_size=1)

# 5. 根据生成的embedding字典，为每行数据添加embedding列
def get_embedding(seq):
    return embeddings.get(seq, None)

df['embedding1'] = df['seq1'].apply(get_embedding)
df['embedding2'] = df['seq2'].apply(get_embedding)

# 6. 保存结果为JSON文件
df.to_json("/content/PPI_prediction_dataset_with_esm2_embeddings.json", orient='records', force_ascii=False, indent=2)

print("Embedding generation completed!")