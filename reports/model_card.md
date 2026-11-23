# Model card: AdaptLab (classificador de intents, Mukua Telecom sintética)

## Modelo base

- Nome: `Qwen/Qwen2.5-0.5B-Instruct`
- Revisão (commit): `7ae557604adf67be50417f59c2c2f167def9a775`
- Licença: apache-2.0

## Dataset

- Sintético, gerado por `scripts/prepare_data.py` (ver `data/DATASET_CARD.md`).
- Licença: CC-BY-4.0
- Hash SHA-256 do teste: `8850e7c0a283d4ed81c576a67def9874e96f3936f86918943d6ec6dc5ebd219d`

## Hiperparâmetros

```
{'lora': {'r': 16, 'alpha': 32, 'dropout': 0.05, 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], 'bias': 'none'}, 'training': {'learning_rate': 0.0002, 'num_train_epochs': 3, 'per_device_train_batch_size': 8, 'gradient_accumulation_steps': 2, 'warmup_ratio': 0.03, 'lr_scheduler_type': 'cosine', 'logging_steps': 10, 'eval_strategy': 'epoch', 'save_strategy': 'no', 'seed': 42, 'completion_only_loss': True}, 'quantization': 'none'}
```

## Métricas (com run de origem)

Macro-F1 no dev por abordagem: rules=0.8351, zero_shot=TODO: medir, few_shot=TODO: medir, lora=TODO: medir

- LoRA no dev: TODO: medir (run TODO: medir)
- LoRA no teste: TODO: medir (run TODO: medir)

## Critério de paragem

- Estado: **pendente**. LoRA ainda não medido no dev

## Utilização prevista

Classificar mensagens de apoio ao cliente de uma operadora (fictícia), para encaminhar pedidos
e marcar escalonamentos. Apoio à decisão com revisão humana.

## Utilização não prevista

- Decisões automáticas sem revisão humana, sobretudo quando `requires_human` é verdadeiro.
- Dados reais de clientes, outras operadoras, outros idiomas.
- Factos que mudam (preços, cobertura): não são ensinados por fine-tuning.

## Limitações

- Dados sintéticos; o modelo pode aprender os templates e não a fala real.
- Português de Angola ainda sem revisão humana (ver dataset card).
- Teste pequeno e escalonamentos escassos: IC largo.
- Uma só seed, salvo indicação em contrário nas métricas.

## Riscos de enviesamento

- Localidades e padrões de número escolhidos pelos autores.
- Escalonamento definido pelos autores, não por política da operadora.

## Reprodução

```
.venv\Scripts\python scripts\prepare_data.py --check
.venv\Scripts\python scripts\train.py --config configs\train_gpu.yaml   (GPU)
.venv\Scripts\python scripts\evaluate.py --approach lora --run-dir runs\train-XXXX --split dev
.venv\Scripts\python scripts\evaluate.py --report
```
