# GPT-2 (124M) — From First Principles

This repository provides:

- A minimal PyTorch re-implementation of GPT-2 (124M) in `gpt2_pytorch.py` for learning and experimentation.
- A small inference helper `hf_inference.py` that uses Hugging Face's pretrained `gpt2` model for immediate results.

Quick start

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the HF-backed demo (downloads pretrained weights automatically):

```
python hf_inference.py --prompt "Hello world" --max-length 50
```

Notes

- The PyTorch reimplementation is intended for study and small experiments. To use pretrained weights, prefer the Hugging Face model (demo above) or run a weight-conversion routine (not included).
