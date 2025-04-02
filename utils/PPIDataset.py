import torch
import json
from torch.utils.data import Dataset

class PPIDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        seq1 = item['seq1']
        seq2 = item['seq2']
        full_seq = seq1 + seq2
        L = len(full_seq)
        # 生成 mask：非 'X' 的位置记为 1，padding 的位置记为 0
        mask = [1 if aa != 'X' else 0 for aa in full_seq]
        mask = torch.tensor(mask, dtype=torch.float32)  # [L]
        
        embedding = torch.tensor(item['embedding_concat'], dtype=torch.float32)
        
        label = torch.tensor(item['Label'], dtype=torch.float32)
        return embedding, mask, label