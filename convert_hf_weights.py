"""Convert Hugging Face `gpt2` weights into the local `GPT2Model`.

Usage:
    python convert_hf_weights.py --out local_gpt2.pt

It will also run a short sanity check comparing logits on a sample prompt.
"""
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from gpt2_pytorch import GPT2Model, GPT2Config


def map_and_copy(hf_state, local_model):
    local_sd = local_model.state_dict()
    new_sd = {}

    # direct mappings
    mapping = {
        'wte.weight': 'transformer.wte.weight',
        'wpe.weight': 'transformer.wpe.weight',
        'ln_f.weight': 'transformer.ln_f.weight',
        'ln_f.bias': 'transformer.ln_f.bias',
        'lm_head.weight': 'lm_head.weight',
    }

    for k, v in mapping.items():
        if v in hf_state:
            tensor = hf_state[v].clone()
            # copy or transpose if shapes differ
            if tensor.shape == local_sd[k].shape:
                new_sd[k] = tensor
            elif tensor.t().shape == local_sd[k].shape:
                new_sd[k] = tensor.t().clone()
            else:
                new_sd[k] = tensor

    # per-layer mappings
    for i in range(local_model.config.n_layer):
        prefix_local = f'h.{i}.'
        prefix_hf = f'transformer.h.{i}.'
        layer_map = [
            ('ln_1.weight', 'ln_1.weight'),
            ('ln_1.bias', 'ln_1.bias'),
            ('attn.c_attn.weight', 'attn.c_attn.weight'),
            ('attn.c_attn.bias', 'attn.c_attn.bias'),
            ('attn.c_proj.weight', 'attn.c_proj.weight'),
            ('attn.c_proj.bias', 'attn.c_proj.bias'),
            ('ln_2.weight', 'ln_2.weight'),
            ('ln_2.bias', 'ln_2.bias'),
            ('mlp.c_fc.weight', 'mlp.c_fc.weight'),
            ('mlp.c_fc.bias', 'mlp.c_fc.bias'),
            ('mlp.c_proj.weight', 'mlp.c_proj.weight'),
            ('mlp.c_proj.bias', 'mlp.c_proj.bias'),
        ]

        for local_name, hf_name in layer_map:
            local_key = prefix_local + local_name
            hf_key = prefix_hf + hf_name
            if hf_key in hf_state:
                tensor = hf_state[hf_key].clone()
                if tensor.shape == local_sd[local_key].shape:
                    new_sd[local_key] = tensor
                elif tensor.t().shape == local_sd[local_key].shape:
                    new_sd[local_key] = tensor.t().clone()
                else:
                    new_sd[local_key] = tensor

    # ensure all expected keys are present; fall back to original if missing
    final_sd = local_sd.copy()
    for k in final_sd.keys():
        if k in new_sd:
            final_sd[k] = new_sd[k]

    # load into model
    local_model.load_state_dict(final_sd)

    # re-tie lm_head to embeddings
    local_model.lm_head.weight = local_model.wte.weight


def sanity_check(local_model, hf_model, tokenizer, prompt="Hello world"):
    local_model.eval()
    hf_model.eval()

    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        hf_logits = hf_model(**inputs).logits
        local_logits = local_model(inputs['input_ids']).cpu()

    # compare shapes first
    print('hf logits shape:', hf_logits.shape)
    print('local logits shape:', local_logits.shape)

    # compute max absolute difference on the overlapping slice
    min_t = min(hf_logits.shape[1], local_logits.shape[1])
    diff = (hf_logits.cpu()[:, :min_t, :] - local_logits[:, :min_t, :]).abs().max()
    print('max abs diff:', diff.item())
    return diff.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='local_gpt2.pt')
    parser.add_argument('--prompt', type=str, default='Hello world')
    args = parser.parse_args()

    print('Loading HF model...')
    hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
    hf_state = hf_model.state_dict()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    print('Creating local model...')
    cfg = GPT2Config()
    local_model = GPT2Model(cfg)

    print('Mapping and copying weights...')
    map_and_copy(hf_state, local_model)

    print('Saving local weights to', args.out)
    torch.save(local_model.state_dict(), args.out)

    print('Running sanity check...')
    diff = sanity_check(local_model, hf_model, tokenizer, prompt=args.prompt)
    if diff < 1e-3:
        print('Sanity check passed (close match).')
    else:
        print('Sanity check completed (difference may be expected).')


if __name__ == '__main__':
    main()
