# PROMPT 08 — AdaptLab: fine-tuning de um modelo pequeno com LoRA/QLoRA

## Objectivo e caso de uso
Cria um pipeline reproduzível para adaptar um pequeno modelo de linguagem (0,5B a 1,5B parâmetros, licença explícita e permissiva quando possível) a um domínio de **classificação e extração estruturada de pedidos de apoio ao cliente em português angolano**, usando apenas dados sintéticos/cedidos com licença. O alvo é devolver JSON com `intent`, `priority`, `entities`, `requires_human`. Não ensinar factos mutáveis por fine-tuning nem inventar melhorias.

## Stack e arquitectura
Python, PyTorch, Transformers, Hugging Face Datasets, TRL, PEFT LoRA/QLoRA, bitsandbytes só em CUDA/GPU suportada; trackings de runs via MLflow ou ficheiros JSON reproducíveis. Scripts `prepare_data.py`, `train.py`, `evaluate.py`, `export_adapter.py`, `inference.py`; notebook GPU opcional Colab/Kaggle com version pins. Zero obrigação de treinar no PC local sem GPU: disponibiliza `--dry-run` de CPU para validar dataset/tokenização sem treino pesado. A selecção do modelo deve ser justificada por licença, suporte de chat templates, memória GPU e benchmark.

## Dataset
Cria gerador de >=500 exemplos sintéticos com labels controlados e sotaque/léxico PT-AO quando validado por revisões humanas; incluir ruído, mistura PT/EN, mensagens incompletas, entidades ausentes e casos que exigem escalar. Esquema de train/dev/test SEM duplicados/contaminação por templates, split por cenário/template e dataset card com origem, licença, classes, limitações e risco de enviesamento. Reserva amostra de teste intacta e não ajusta hiperparâmetros com ela.

## Comparação científica obrigatória
Benchmark (1) modelo base zero-shot com prompt fixo, (2) modelo base few-shot, (3) LoRA adaptado, (4) opcional regras determinísticas. Métricas macro-F1 de intent, F1 de entidades, JSON validity, recall de pedidos high-risk, latência e custo de inferência. Executar repetidas seeds quando orçamento suportar e reportar variância. Gráficos train/validation loss, matriz de confusão, erros por tipo, tamanho do adapter, VRAM máxima medida, tempo de treino e quantização. Parar se LoRA não melhorar.

## Exposição e deploy
API FastAPI `/predict`, `/model-card`, `/health`, `GET /metrics`; pode servir modelo quantizado em GPU externa e um endpoint DEMO sem GPU com outputs de casos pré-calculados claramente identificados; não simular inferência real como se fosse produção. UI pequena com input, predicted JSON, confiança calibrada quando possível e comparação base/adaptado. Publicar model card e adapter no Hugging Face apenas se os termos/licenças permitirem; documentar versão base/checkpoint/hiperparâmetros e como reproduzir.

## Testes
Schema, validação dataset, split leakage, tokenização, checkpoints e erro sem CUDA; evaluate completo opcional via CI manual. Nunca publicar dados privados nem pesos base sem direitos de redistribuição. Usa PEFT/TRL docs actuais.

## Regras de execução (obrigatórias)
És um agente sénior de software engineering e AI engineering. A tua missão é IMPLEMENTAR código funcional neste repositório, não apenas escrever um plano ou pseudocódigo. Primeiro inspecciona o repositório; depois propõe em `IMPLEMENTATION_PLAN.md` um plano curto e segue-o por fases até teres uma versão executável. Não pares após criar ficheiros vazios. Usa documentação oficial actual das bibliotecas e fixa versões compatíveis; não inventes APIs nem dependências. Caso não tenhas acesso a rede/credenciais, implementa adaptadores, mocks determinísticos e testes locais, mas distingue explicitamente o que foi testado de verdade do que foi apenas preparado.

Normas comuns: Python 3.11+ (salvo indicação explícita), tipagem, Pydantic v2, configuração por `.env.example`, FastAPI com OpenAPI, Ruff, pytest, logging estruturado sem secrets/PII, Dockerfile, `docker-compose.yml` quando fizer sentido, healthcheck e README claro (arranque, decisões de arquitectura, ameaças, limitações, custos, credenciais e deploy). Não coloques chaves no cliente ou no Git; valida entradas, implementa timeouts/retries limitados e autenticacao para operações administrativas. Inclui modo DEMO com dados sintéticos e sem pagamento; implementa testes unitários e testes de integração determinísticos em CI, mais uma suite de integração real opcional. Escreve `docs/architecture.md`, `docs/decisions.md`, `docs/case-study.md` com métricas REAIS ou campos `TODO: medir` (nunca inventes resultados). Entrega exactamente os comandos de execução e testes e lista limitações.

Para deployment, prepara backend persistente numa VM/serviço de containers e frontend estático quando possível; não assumes que Vercel Functions suporta Redis, workers long-running, volumes persistentes ou treino GPU. Se houver persistência, usa serviço gerido/volume externo e documenta credenciais, migrações e backups. Antes de concluir, executa testes, corrige falhas, regista o output e documenta o que não foi possível executar.
