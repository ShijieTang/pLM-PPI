# pLM-PPI

#### data
- **whole sequences**: sequences with overall length less than 2048 and padded to 2048 length.

- **truncated sequences**: randomly selected sequences and truncated or padded to 2048 length.

#### utils

for onehot embedding generation:
```
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 utils/baseline_embedding.py --embedding_type "onehot"
```

for positional embedding generation:
```
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 utils/baseline_embedding.py --embedding_type "positional"
```



