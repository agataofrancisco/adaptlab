# Proposal

## Why

Fine-tuning é muitas vezes a primeira ideia e a pior escolha. Serve mal para ensinar factos que mudam, é caro e é fácil de "provar" com uma avaliação contaminada. Mas para tarefas de **formato e classificação num domínio e numa variante linguística específicos** (português de Angola, mistura PT/EN, gíria de apoio ao cliente), um modelo pequeno adaptado com LoRA pode bater um modelo maior com prompt, com menos custo e latência. O AdaptLab mostra um pipeline reproduzível e cientificamente honesto. Inclui dataset com licença clara, splits sem fuga, comparação obrigatória com baselines zero-shot e few-shot, medição de VRAM, tempo e variância, e **paragem explícita se o LoRA não melhorar**.

Domínio fictício: o apoio ao cliente da **Mukua Telecom**, uma operadora móvel angolana fictícia. A tarefa é ler a mensagem do cliente e devolver JSON com `intent` (por exemplo `recarga`, `avaria_rede`, `faturacao`, `portabilidade`, `cancelamento`, `pacote_dados`), `priority`, `entities` (número, valor, localidade, data) e `requires_human`. Todos os dados são sintéticos ou cedidos com licença.

## What Changes

- Projecto novo em `projects/adaptlab` com os scripts `prepare_data.py`, `train.py`, `evaluate.py`, `export_adapter.py` e `inference.py`, e um notebook Colab/Kaggle com versões fixadas.
- Gerador de ≥500 exemplos sintéticos com labels controlados, ruído, mistura PT/EN, mensagens incompletas, entidades ausentes e casos de escalonamento. Revisão humana de uma amostra para o léxico PT-AO e um dataset card.
- Splits por cenário/template (sem contaminação), com conjunto de teste intocado.
- Selecção justificada de um modelo base de 0,5B a 1,5B parâmetros (licença, chat template, memória e benchmark).
- Treino LoRA/QLoRA com TRL + PEFT (bitsandbytes só em CUDA). `--dry-run` em CPU para validar os dados e a tokenização sem treino.
- Comparação obrigatória: base zero-shot, base few-shot, LoRA e regras determinísticas (opcional). Métricas macro-F1, F1 de entidades, validade JSON, recall de alto risco, latência e custo, com seeds repetidas quando o orçamento o permite.
- API FastAPI `/predict`, `/model-card`, `/health` e `/metrics`, com o modo DEMO sem GPU claramente identificado como pré-calculado.
- Model card e publicação no Hugging Face apenas se as licenças o permitirem.
- Documentação de aprendizagem.

## Capabilities

### New Capabilities
- `training-dataset`: geração sintética, revisão, schema, splits sem fuga e dataset card.
- `adapter-training`: selecção do modelo base, configuração LoRA/QLoRA, treino reproduzível, dry-run e tracking.
- `model-evaluation`: baselines, métricas, variância, gráficos, critério de paragem e relatório.
- `inference-service`: API de predição, modo demo honesto, comparação base/adaptado e UI.
- `model-publication`: model card, licenças, exportação do adapter e publicação condicionada.

### Modified Capabilities
<!-- Nenhuma: projecto novo. -->

## Impact

- **Código novo**: `adaptlab/` (pacote), `scripts/`, `notebooks/`, `data/` (gerado, com o card), `api/`, `ui/`, `reports/` e `docs/`.
- **Dependências** (confirmar versões): PyTorch, Transformers, Datasets, TRL, PEFT, bitsandbytes (só CUDA), MLflow ou tracking JSON, scikit-learn (métricas), FastAPI e pytest.
- **Hardware**: não há GPU local. O treino real corre num notebook Colab/Kaggle com GPU. Localmente só correm `--dry-run`, testes, avaliação de baselines pequenas em CPU (lenta) e a API em modo demo.
- **Fora de âmbito**: ensinar factos (preços de pacotes) por fine-tuning, RLHF/DPO, modelos acima de 1,5B e treino distribuído.
