# pLM-PPI

### Task
This repotary is a project for 02-710 Computational Genomics at Carnegie Mellon University aiming to evaluate the effectiveness of protein language models (pLMs) on protein-protein interaction (PPI) task. Regular PPI prediction is a regression task. Here, to simplify, it is designed as a binary classification task (0 for no known physical interations, 1 for having known physical detection).

### Data

The entire dataset was generated by Shijie Tang using BIOGRID and CAFA5 competition. Sequence length was limited with lower than 2048 to save computational resources.

- **whole sequences**: sequences with overall length less than 2048 and padded to 2048 length.

- **truncated sequences**: randomly selected sequences and truncated or padded to 2048 length.

### Utils

For onehot embedding generation:
```
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 utils/baseline_embedding.py --embedding_type "onehot"
```

For positional embedding generation:
```
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 utils/baseline_embedding.py --embedding_type "positional"
```

For ESM2/ProstT5 embedding generation (modify the model and saving path):
```
python utils/embedding_extraction.py
```

### Model training
The `main.py` contains all training required functions.
```
python utils/embedding_extraction.py
```
`config.yaml` file records all parameters for both model training and training recording. You can directly modify the parameters by using text editor or by this way:
```
python main.py model.model_choice=esm35 model.num_features=960
```

All training result was saved in results/model directory.
