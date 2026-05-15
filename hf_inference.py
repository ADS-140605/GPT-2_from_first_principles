"""Hugging Face-backed inference helper for GPT-2 (124M).

This script uses `transformers` to download and run the pretrained `gpt2` model.
"""
import argparse
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch


def generate(prompt: str, max_length: int = 50, temperature: float = 1.0):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            max_length=inputs["input_ids"].shape[1] + max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    out = generate(args.prompt, max_length=args.max_length, temperature=args.temperature)
    print(out)


if __name__ == "__main__":
    main()
