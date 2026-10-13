# Released weights

| File | Contents | Size | SHA-256 |
| --- | --- | ---: | --- |
| `spacompass_pretrained.pt` | released model weights | 1.9 GB | `381a6faecfa880d825793460abc3b92665280daa4f124ff8771f7f2368539058` |

## Contents

`torch.load(path)["model"]` is a state dict with two prefixes:

```python
checkpoint["model"]["first_stage_model.*"]   # stage 1, gene attention (421 M parameters)
checkpoint["model"]["second_stage_model.*"]  # stage 2, cell attention (78 M parameters)
```

`spacompass.model.builder` rebuilds the architecture from
`configs/spacompass.json` and loads the checkpoint, so fairseq is not needed:

```python
from spacompass.model.builder import load_gene_model, read_config
from spacompass.model.two_stage import load_two_stage

config = read_config("configs/spacompass.json")
stage1, source = load_gene_model(config, "weights/spacompass_pretrained.pt")  # zero-shot embeddings
two_stage = load_two_stage(config, "weights/spacompass_pretrained.pt")        # patch-level inference
```

## Distribution

`weights/*.pt` is listed in `.gitignore`: publish the file through a model hub
(Hugging Face, Zenodo, …) or Git LFS rather than committing it, and ship this
README with the download.
