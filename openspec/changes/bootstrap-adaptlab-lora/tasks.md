# Tasks

## 1. Fundações

- [x] 1.1 Criar `adaptlab/{data,training,evaluation,serving}`, `scripts/`, `notebooks/`, `api/`, `ui/`, `reports/`, `runs/` e `docs/learn/`; verificar com `git status`
- [x] 1.2 Confirmar e fixar versões (torch CPU local, transformers, datasets, trl, peft, scikit-learn, fastapi, pytest, Ruff; bitsandbytes só no notebook); verificar a instalação local sem CUDA
- [x] 1.3 Escrever `docs/learn/00-visao-geral.md` e `01-quando-fazer-fine-tuning.md`; verificar que o diagrama do fluxo corresponde aos scripts

## 2. Dataset

- [x] 2.1 Definir o schema de saída (Pydantic) e os cenários/templates com o léxico PT-AO, ruído, PT/EN, incompletos e escalonamento; verificar a validação com exemplos manuais
- [x] 2.2 Implementar `prepare_data.py` (gerador ≥500 com seed, estatísticas, validação de entidades); verificar a reprodutibilidade e a cobertura mínima
- [x] 2.3 Implementar os splits por template, a detecção de duplicados e quase-duplicados e o hash do teste; verificar com teste de fuga artificial
- [ ] 2.4 Fazer a revisão humana de uma amostra e registar as decisões; escrever o dataset card; verificar a validação do card
- [x] 2.5 Escrever `docs/learn/02-dataset-sintetico-e-fugas.md`; verificar com as estatísticas reais

## 3. Treino

- [x] 3.1 Implementar o formato conversacional, o chat template e a loss completion-only; verificar com teste que imprime e confirma os tokens mascarados
- [x] 3.2 Implementar `train.py` com config, `--dry-run` em CPU, erro claro sem CUDA no QLoRA e tracking JSON; verificar o dry-run localmente
- [ ] 3.3 Fazer a selecção do modelo base: zero-shot no dev em ≥2 candidatos (CPU/amostra ou notebook) com tabela de licenças e memória; verificar a tabela em `reports/`
- [ ] 3.4 Criar o notebook GPU com versões fixadas (treino, avaliação, exportação); verificar a execução no Colab/Kaggle, ou registar a pendência
- [x] 3.5 Escrever `docs/learn/03-tokenizacao-e-chat-templates.md`, `04-loss-mascarada-completion-only.md`, `05-lora-por-dentro.md`, `06-quantizacao-e-qlora.md` e `07-hiperparametros-e-dev-set.md`; verificar os exemplos de tokens com o dry-run

## 4. Avaliação

- [x] 4.1 Implementar `evaluate.py` com o parsing comum, as baselines zero/few-shot/regras e as métricas (macro-F1, F1 entidades, JSON válido, recall alto risco, latência, custo, confusão); verificar as métricas com testes de exemplos à mão
- [x] 4.2 Implementar o relatório (curvas, confusão, adapter, VRAM, tempo, seeds, `TODO: medir`) e o critério de paragem; verificar com teste em que o LoRA não melhora → bloqueio
- [ ] 4.3 Correr o treino e a avaliação no notebook (≥1 seed, 3 se possível) e gravar os resultados; verificar que cada número tem uma run de origem
- [ ] 4.4 Escrever `docs/learn/08-avaliacao-e-baselines.md` com os resultados obtidos; verificar a consistência

## 5. Serviço e publicação

- [ ] 5.1 Implementar a API (`/predict`, `/model-card`, `/health`, `/metrics`) com o modo real (PEFT) e o demo `precomputed_demo`; verificar com testes de saída inválida, mensagem nova em demo e health em CPU
- [ ] 5.2 Implementar a UI lado a lado base vs adaptado; verificar no browser
- [ ] 5.3 Implementar `export_adapter.py` e `inference.py`, o model card e a publicação condicionada (licenças, critério de paragem, confirmação manual); verificar com teste de licença proibida → recusa
- [ ] 5.4 Escrever `docs/learn/09-servir-adapters.md`; verificar com a API a correr

## 6. Entrega

- [ ] 6.1 Dockerfile da API (modo demo), CI (Ruff, pytest, dry-run, validação de dados e fugas; avaliação completa só manual); verificar a validade
- [ ] 6.2 `README.md`, `docs/architecture.md`, `docs/decisions.md` e `docs/case-study.md`; verificar os comandos
- [ ] 6.3 `docs/learn/10-perguntas-de-entrevista.md` (≥25); verificar a consistência
- [ ] 6.4 Revisão independente (fugas de dados, licenças, honestidade das métricas, modo demo); verificar que os achados estão resolvidos ou documentados

## Workflow follow-up

- Publicar o adapter no Hugging Face só se aprovado.
- Arquivar após revisão.
