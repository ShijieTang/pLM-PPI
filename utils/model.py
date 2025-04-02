import torch.nn as nn
import torch
import math

class MLP(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        num_hidden = cfg.hidden
        num_output = cfg.label_num
        num_input = cfg.num_features

        self.fc_input = nn.Linear(num_input, num_hidden)

        self.norm = nn.LayerNorm(num_hidden)
        self.dropout = nn.Dropout(cfg.dropout)

        self.predict = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden, num_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden // 2, num_output)
        )

    def forward(self, x):
        x = self.fc_input(x)
        x = self.norm(x)
        x = self.dropout(x)

        output = self.predict(x)

        return output



class OnehotMLP(torch.nn.Module):
  def __init__(self, cfg):
      super().__init__()

      num_input = cfg.num_features
      num_hidden = cfg.hidden
      num_output = cfg.label_num

      self.hidden = nn.Linear(num_input, num_hidden)
      self.predict = nn.Sequential(
          nn.Dropout(cfg.dropout),
          # nn.LeakyReLU(inplace=True),
          nn.ReLU(inplace=True),
          nn.Linear(num_hidden, int(num_hidden/2)),
          # nn.Linear(int(num_hidden/2), num_output)
      )
      self.avg = nn.AdaptiveAvgPool1d(1)

      self.cls = nn.Linear(int(num_hidden/2), num_output)

      # self.coor = nn.Linear(int(num_hidden/2), 2)

      self.softmax = nn.Softmax(dim=1)
      
  def forward(self, x, mask):
      x = self.hidden(x)
      x = self.predict(x)

      # last_hidden_state = x.mean(dim=1)
      x = self.cls(x)

      # apply mask
      x = x * mask.unsqueeze(-1)  # 确保掩码的形状与x相匹配

      # 汇聚操作，这里使用平均汇聚
      sum_embeddings = torch.sum(x, dim=1)
      # 使用掩码计算每个序列的有效长度
      lengths = mask.sum(dim=1, keepdim=True)
      cls = sum_embeddings / lengths

      # return cls, last_hidden_state
      return cls

class PositionalEncoding1D(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
      super().__init__()
      self.dropout = nn.Dropout(p=dropout)

      position = torch.arange(max_len).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
      pos_enc = torch.zeros(1, max_len, d_model)
      pos_enc[0, :, 0::2] = torch.sin(position * div_term)
      pos_enc[0, :, 1::2] = torch.cos(position * div_term)

      self.register_buffer('pos_enc', pos_enc)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      """
      Args:
          x: Tensor, shape [seq_len, batch_size, embedding_dim]
      """
      # added permute to make batch first
      # x = x + self.pos_enc.permute(1, 0, 2)[:1, :x.size(1)]
      x = x + self.pos_enc[:1, :x.size(1)]
      return self.dropout(x)
    
class PositionalMLP(torch.nn.Module):
  def __init__(self, cfg):
      super().__init__()

      num_input = cfg.num_features
      num_hidden = cfg.hidden
      num_output = cfg.label_num

      self.embed = nn.Embedding(21, 256)
      self.fc1 = nn.Linear(num_input, num_hidden)
      self.pe = PositionalEncoding1D(256)
      self.fc2 = nn.Linear(256, num_hidden)

      # Multi-Head Attention Layer
      self.mha = nn.MultiheadAttention(embed_dim=num_hidden,
                                        num_heads=cfg.attn_head,
                                        dropout=cfg.dropout,
                                        batch_first=True)

      self.norm = nn.LayerNorm(num_input)
      self.dropout = nn.Dropout(cfg.dropout)
      self.predict = nn.Sequential(
          nn.Dropout(cfg.dropout),
          # nn.LeakyReLU(inplace=True),
          nn.ReLU(inplace=True),
          nn.Linear(num_hidden, int(num_hidden/2)),
          # nn.Linear(int(num_hidden/2), num_output)
      )


      self.avg = nn.AdaptiveAvgPool1d(1)
      self.fc = nn.Linear(num_hidden, num_hidden)
      self.cls = nn.Linear(int(num_hidden/2), num_output)
  def forward(self, x, mask):
      # x = self.fc1(x)
      x = self.embed(x.long())
      x= self.pe(x)
      x = self.fc2(x)
      x = x.squeeze(0)
      # Permute tensor dimensions for Multi-Head Attention
      attn_output, _ = self.mha(x, x, x)  # Query, Key, Value
      x = self.fc(attn_output)
      x = self.predict(x)
      x = self.cls(x)

      x = x * mask.unsqueeze(-1)
      sum_embeddings = torch.sum(x, dim=1)
      lengths = mask.sum(dim=1, keepdim=True)
      cls = sum_embeddings / lengths

      return cls