# 00 · Visão geral do AdaptLab

## O problema

A Mukua Telecom (fictícia) recebe mensagens de clientes em português de Angola, com
gíria e mistura com inglês. Precisamos de um classificador que devolva sempre um JSON
com `intent`, `priority`, `entities` e `requires_human`. O AdaptLab testa se um modelo
pequeno com LoRA bate as alternativas mais simples, e para se não bater.

## O que entra e o que sai

Entra: o catálogo sintético de cenários (`adaptlab/data/catalog.py`) e a configuração
(`configs/data.yaml`, `configs/train.yaml`).

Sai, neste momento, medido:

- Dataset: 688 exemplos após deduplicação, splits por cenário, zero fugas
  (`data/processed/manifest.json`, ver `02`).
- Dry-run de treino em CPU: `status: ok`, máscara verificada (ver `04`).
- 43 testes a passar (`tests/`), ruff sem erros.

Implementação de API e UI em modo demo está disponível. Permanecem por fazer a revisão humana
do dataset, a comparação final dos modelos, o treino e a avaliação em GPU e a validação de
deployment (ver "Estado" abaixo).

## Como funciona por dentro

```
configs/data.yaml
      │
      ▼
scripts/prepare_data.py ──► adaptlab/data/*  (gerar, validar, deduplicar, splits, fugas, revisão)
      │                            │
      │                            ▼
      │                    data/processed/{train,dev,test}.jsonl + manifest.json + DATASET_CARD.md
      ▼
scripts/train.py --dry-run ──► adaptlab/training/dryrun.py   (CPU: tokenizer real + modelo tiny)
scripts/train.py           ──► adaptlab/training/trainer.py  (GPU: QLoRA NF4, notebooks/train_colab.ipynb)
      │
      ▼
runs/<id>/  (config.json, environment.json, metrics.json, adapter/)   ← tracking
      │
      ▼
scripts/evaluate.py  (baselines + zero/few-shot + LoRA)
      │
      ▼
api/ (FastAPI /predict, /model-card, /health, /metrics), ui/            ← implementados
```

Código comum: `adaptlab/schema.py` (label e parsing defensivo) e
`adaptlab/training/format.py` (system prompt e formato de treino).

## Porquê esta escolha

- **Um único schema** (`schema.py`) para dados, treino, avaliação e serviço. Se o formato
  mudar, muda num só sítio.
- **Dry-run que usa o caminho real do TRL** (não uma reimplementação): o que se valida é o
  que o treino executa.
- **Pipeline de dados que falha** em vez de avisar: um dataset com fugas não chega ao
  treino.

## Instalação local (sem GPU)

```
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install --no-compile -e ".[train,eval,serve,dev]"
```

Versões em `pyproject.toml`, todas confirmadas no índice do PyPI. Torch vem do índice CPU
oficial (`download.pytorch.org/whl/cpu`) para não puxar CUDA. Verificação:
`torch.cuda.is_available()` é `False` neste PC, como esperado.

Observação real: a instalação completa demorou várias dezenas de minutos no Windows
(descarga lenta e compilação de bytecode). `--no-compile` acelera a instalação mas a
primeira importação de `transformers`/`trl` é lenta.

## Estado

| Área | Estado |
|---|---|
| Dados e splits | feito, verificado |
| Revisão PT-AO | **pendente** (pessoa revisora; ver `data/review/`) |
| Dry-run CPU | feito, verificado |
| Selecção do modelo base | amostra exploratória n=20 para Qwen 0.5B; comparação ≥2 modelos pendente |
| Treino QLoRA real | **pendente** (GPU; notebook pronto, não executado) |
| Avaliação e relatório | regras medidas; zero-shot exploratório; few-shot e LoRA pendentes |
| API e UI | implementadas em modo demo |
| Publicação | condicionada a treino, avaliação e revisão humana |

## Como falha

- Versões de bibliotecas: fixadas no `pyproject.toml` e registadas em
  `runs/*/environment.json`.
- Ambiente errado: QLoRA sem CUDA falha com mensagem clara (`04`/`06`).
- Teste de fuga: se algum template aparecer em dois splits, o pipeline falha.

## Perguntas de entrevista

**1. Porque é que o projecto tem tantos scripts pequenos em vez de um só?**
Cada etapa tem um critério de falha próprio: dados (fugas), treino (ambiente), avaliação
(baselines). Separar permite correr só a etapa necessária e ver onde falhou.

**2. O que é reprodutível neste projecto e o que não é?**
Reprodutível: os dados (seed), os splits, o hash do teste, a revisão do commit do modelo,
as versões. Não reprodutível por completo: o treino em GPU, por não-determinismo de kernels
CUDA; por isso o relatório mostra variância entre seeds.

**3. Porque é que o torch vem de um índice diferente?**
O índice padrão do PyPI traz a build com CUDA, pesada e inútil num PC sem GPU. O índice
CPU traz uma build pequena para o mesmo número de versão.

**4. Porque é que não há Docker?**
Não há Docker neste PC. O Dockerfile da API (tarefa 6.1) fica escrito para correr em modo
demo, mas a sua validade fica por verificar até haver Docker.
