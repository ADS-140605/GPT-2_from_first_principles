# Graph Report - GPT-2_from_first_principles  (2026-05-15)

## Corpus Check
- Corpus is ~663 words - fits in a single context window. You may not need a graph.

## Summary
- 27 nodes · 33 edges · 5 communities (3 shown, 2 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Transformer Components|Transformer Components]]
- [[_COMMUNITY_Model Architecture|Model Architecture]]
- [[_COMMUNITY_Semantic Links|Semantic Links]]
- [[_COMMUNITY_Inference Scripts|Inference Scripts]]
- [[_COMMUNITY_Configuration|Configuration]]

## God Nodes (most connected - your core abstractions)
1. `MLP` - 5 edges
2. `Block` - 5 edges
3. `CausalSelfAttention` - 4 edges
4. `GPT2Model` - 3 edges
5. `Block` - 3 edges
6. `GPT2Model` - 3 edges
7. `generate()` - 2 edges
8. `main()` - 2 edges
9. `generate` - 2 edges
10. `GPT-2 — From First Principles` - 2 edges

## Surprising Connections (you probably didn't know these)
- `GPT2Model` --semantically_similar_to--> `GPT2LMHeadModel`  [INFERRED] [semantically similar]
  D:/MyGPT2/GPT-2_from_first_principles/gpt2_pytorch.py → D:/MyGPT2/GPT-2_from_first_principles/hf_inference.py
- `GPT-2 — From First Principles` --references--> `GPT2Model`  [EXTRACTED]
  D:/MyGPT2/GPT-2_from_first_principles/README.md → D:/MyGPT2/GPT-2_from_first_principles/gpt2_pytorch.py
- `GPT-2 — From First Principles` --references--> `generate`  [EXTRACTED]
  D:/MyGPT2/GPT-2_from_first_principles/README.md → D:/MyGPT2/GPT-2_from_first_principles/hf_inference.py

## Hyperedges (group relationships)
- **GPT-2 Transformer Stack** — gpt2_pytorch_gpt2model, gpt2_pytorch_block, gpt2_pytorch_causalselfattention, gpt2_pytorch_mlp [EXTRACTED 0.95]

## Communities (5 total, 2 thin omitted)

### Community 1 - "Model Architecture"
Cohesion: 0.33
Nodes (4): Block, GPT2Config, GPT2Model, Minimal GPT-2 (124M) architecture in PyTorch.  This is a pedagogical implement

### Community 2 - "Semantic Links"
Cohesion: 0.33
Nodes (7): Block, CausalSelfAttention, GPT2Model, MLP, generate, GPT-2 — From First Principles, GPT2LMHeadModel

### Community 3 - "Inference Scripts"
Cohesion: 0.67
Nodes (3): generate(), main(), Hugging Face-backed inference helper for GPT-2 (124M).  This script uses `tran

## Knowledge Gaps
- **6 isolated node(s):** `GPT2Config`, `Minimal GPT-2 (124M) architecture in PyTorch.  This is a pedagogical implement`, `Hugging Face-backed inference helper for GPT-2 (124M).  This script uses `tran`, `GPT2Config`, `CausalSelfAttention` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MLP` connect `Transformer Components` to `Model Architecture`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Why does `CausalSelfAttention` connect `Transformer Components` to `Model Architecture`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Why does `Block` connect `Model Architecture` to `Transformer Components`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **What connects `GPT2Config`, `Minimal GPT-2 (124M) architecture in PyTorch.  This is a pedagogical implement`, `Hugging Face-backed inference helper for GPT-2 (124M).  This script uses `tran` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._