# Arquitectura

```mermaid
flowchart LR
  cfgD[configs/data.yaml] --> prep[scripts/prepare_data.py]
  prep --> dataset[(data/processed/*.jsonl + manifest.json)]
  prep --> card[data/DATASET_CARD.md]
  dataset --> dry[scripts/train.py --dry-run<br/>CPU, tokenizer real + modelo tiny]
  dataset --> train[scripts/train.py<br/>GPU: notebook QLoRA NF4]
  train --> runs[(runs/train-*/<br/>config, environment, metrics, adapter)]
  dataset --> ev[scripts/evaluate.py<br/>rules / zero_shot / few_shot / lora]
  runs --> ev
  ev --> evruns[(runs/eval-*/<br/>metrics.json, predictions.jsonl)]
  evruns --> report[reports/evaluation.md]
  evruns --> stop{critério de paragem<br/>adaptlab/evaluation/stopping.py}
  stop -->|aprovado| export[scripts/export_adapter.py]
  stop -->|bloqueado / pendente| hold[não publicar]
  runs --> export
  export --> card2[reports/model_card.md]
  export --> api[api/main.py<br/>/predict /model-card /health /metrics]
  api --> ui[ui/app.py Streamlit]
```

## Camadas

| Camada | Módulo | Responsabilidade |
|---|---|---|
| Schema | `adaptlab/schema.py` | Label, parsing defensivo, serialização canónica |
| Dados | `adaptlab/data/` | Catálogo, gerador, validação, deduplicação, splits, fugas, revisão, card |
| Formato | `adaptlab/training/format.py` | System prompt, prompt/completion |
| Treino | `adaptlab/training/` | Config validada, dry-run CPU, treino GPU, tracking, guarda CUDA |
| Avaliação | `adaptlab/evaluation/` | Métricas, bootstrap, baselines, critério, relatório |
| Serviço | `adaptlab/serving/`, `api/` | Predictores real e demo, endpoints, métricas |
| Publicação | `adaptlab/publication/` | Portão de licenças e critério, model card |

## Fronteiras e invariantes

- O **teste** só é lido por `evaluate.py --final`, com hash verificado contra o manifesto.
- O **modo demo** nunca chama um modelo; as respostas trazem `mode: precomputed_demo`.
- O **label** é o mesmo objecto em dados, treino, avaliação e API (`Label`).
- O **base** é sempre o commit fixado em `configs/train.yaml`.
- **Nenhum peso** entra no git (`.gitignore`, `.gitattributes`).
