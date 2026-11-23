# AdaptLab

Fine-tuning LoRA/QLoRA de um modelo pequeno (0,5B) para classificar mensagens de apoio ao cliente
de uma operadora **fictícia** (Mukua Telecom), com dados sintéticos, comparação justa com
baselines e critério de paragem se o LoRA não ganhar.

Estado actual: pipeline de dados, dry-run de treino, baseline de regras, API demo e UI estão
feitos e verificados. O treino real em GPU e a avaliação com modelo estão **pendentes** (não há
GPU local): ver `reports/evaluation.md` e `docs/learn/00-visao-geral.md`.

## Instalação (Windows, CPU)

```
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install --no-compile -e ".[train,eval,serve,dev]"
```

## Comandos

```
.venv\Scripts\python scripts\prepare_data.py --check        # gerar e validar dados (não escreve)
.venv\Scripts\python scripts\prepare_data.py                # escreve data/processed e o dataset card
.venv\Scripts\python scripts\train.py --dry-run             # treino em CPU: tokenização, máscara, 2 passos
.venv\Scripts\python scripts\evaluate.py --approach rules --split dev
.venv\Scripts\python scripts\evaluate.py --report           # reports/evaluation.md
.venv\Scripts\python scripts\build_demo_fixtures.py --run runs\eval-rules-dev-XXXX
.venv\Scripts\python -m pytest tests                        # testes
.venv\Scripts\ruff check adaptlab scripts tests api         # lint
```

Treino real (QLoRA NF4), só no notebook com GPU: `notebooks/train_colab.ipynb`
(`scripts/train.py --config configs/train_gpu.yaml`). Sem CUDA o script falha com mensagem clara.

API e UI (modo demo, sem modelo):

```
set ADAPTLAB_MODE=demo
.venv\Scripts\python -m uvicorn api.main:app --port 8000
.venv\Scripts\streamlit run ui\app.py
```

Modo real: `ADAPTLAB_MODE=real` e `ADAPTLAB_ADAPTER_DIR=<pasta do adapter>`. Sem GPU não é
possível. No modo demo, cada resposta vem marcada `precomputed_demo`.

## Estrutura

- `adaptlab/` código: `data/` (dataset, splits, fugas, revisão), `training/` (formato,
  dry-run, treino, tracking), `evaluation/` (métricas, regras, abordagens, paragem, relatório),
  `serving/` (predictores), `publication/` (portão e model card), `schema.py` (label).
- `api/` FastAPI, `ui/` Streamlit, `scripts/` pontos de entrada, `configs/` YAML.
- `data/` dataset gerado e card, `reports/` relatórios, `runs/` metadados de runs (adapters ignorados).
- `docs/learn/` documentação de aprendizagem (00 a 10). `openspec/` planeamento.

## Dados e licenças

- Dados sintéticos gerados por `scripts/prepare_data.py`. Sem dados pessoais.
- Modelo base: `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0), commit fixado em `configs/train.yaml`.
- Licença do dataset: CC-BY-4.0 (proposta, a confirmar).
- Publicação: bloqueada até o critério de paragem ser aprovado e haver confirmação manual
  (`scripts/export_adapter.py --check-publication --confirm`). Nada é enviado automaticamente.

## Documentação

`docs/learn/` (explicações e perguntas de entrevista), `docs/architecture.md`,
`docs/decisions.md`, `docs/case-study.md`, `data/DATASET_CARD.md`, `reports/model_card.md`.
