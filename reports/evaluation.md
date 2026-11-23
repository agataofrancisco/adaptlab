# Relatório de avaliação AdaptLab

Gerado por `scripts/evaluate.py --report` a partir de `runs/eval-*/metrics.json`.
Só contém valores medidos; o resto está `TODO: medir`.

## Dataset

- Hash SHA-256 do teste (manifesto): `8850e7c0a283d4ed81c576a67def9874e96f3936f86918943d6ec6dc5ebd219d`

## Uso do conjunto de teste

- Leituras finais registadas: **0**
- Ainda não houve leitura final do teste.

## Critério de paragem

- Estado: **PENDENTE**
- Motivo: LoRA ainda não medido no dev
- Margem configurada: 0.02 (macro-F1 no dev)

## Resultados no DEV (seleccionar decisões aqui)

| Abordagem | Macro-F1 intent | IC 95% | F1 entidades | JSON válido | Recall alto risco | Latência média (s) | Tokens de saída (média) | Seeds |
|---|---|---|---|---|---|---|---|---|
| Regras (palavras-chave e regex) | 0.8351 | [0.7728, 0.8809] | 0.9696 | 1.0000 | 0.6275 | 0.0005 | 0.0 | 1 |
| Base zero-shot | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir |
| Base few-shot (k=5) | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir |
| LoRA (QLoRA NF4) | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir |

## Resultados no TESTE (uma vez, com `--final`)

| Abordagem | Macro-F1 intent | IC 95% | F1 entidades | JSON válido | Recall alto risco | Latência média (s) | Tokens de saída (média) | Seeds |
|---|---|---|---|---|---|---|---|---|
| Regras (palavras-chave e regex) | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir |
| Base zero-shot | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir |
| Base few-shot (k=5) | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir |
| LoRA (QLoRA NF4) | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir | TODO: medir |

## Matriz de confusão (regras, DEV)

| gold \ pred | recarga | avaria_rede | faturacao | portabilidade | cancelamento | pacote_dados | outro | __invalid__ |
|---|---|---|---|---|---|---|---|---|
| recarga | 31 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| avaria_rede | 0 | 8 | 0 | 0 | 0 | 0 | 5 | 0 |
| faturacao | 0 | 0 | 22 | 0 | 0 | 0 | 1 | 0 |
| portabilidade | 0 | 0 | 0 | 31 | 0 | 0 | 1 | 0 |
| cancelamento | 0 | 0 | 0 | 0 | 12 | 0 | 17 | 0 |
| pacote_dados | 0 | 0 | 0 | 0 | 0 | 30 | 2 | 0 |
| outro | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 0 |
| __invalid__ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Treino (LoRA)

- Adapter, tamanho, VRAM máxima, tempo e curvas de loss: TODO: medir (run GPU do notebook ainda não feita).
- Quantização prevista: NF4 (configs/train_gpu.yaml).

## Custo de inferência

- Custo por pedido em USD: TODO: medir (não há modelo de preços configurado).
- Tokens de saída por pedido: ver a coluna acima (medidos).

## Limitações

- Dev e teste têm poucos exemplos com escalonamento (ver `data/DATASET_CARD.md`); o IC do recall de alto risco é largo.
- A baseline de regras usa as mesmas localidades e expressões temporais do catálogo sintético.
- Com uma só seed, o relatório não apresenta diferenças como robustas.
