# Design

## Context

O repositório está vazio. Motivação: ver `proposal.md`. Requisitos: ver `specs/`. Não há GPU local: o treino real acontece num notebook Colab/Kaggle, e localmente correm os dados, o dry-run, os testes, a API demo e a avaliação lenta em CPU de amostras. Pedagogicamente, este é o projecto que mais ensina sobre "como funciona um LLM por dentro": tokenização, chat templates, máscaras de loss, quantização e adapters.

## Goals / Non-Goals

**Goals:**
- Reprodutibilidade total: a seed, as versões, o commit do modelo base e o hash dos dados estão registados em cada run.
- Uma comparação justa: os mesmos dados de teste, o mesmo formato de saída e um parsing igual para todas as abordagens.

**Non-Goals:**
- O estado da arte em classificação de intents.
- Servir em produção com alta carga (vLLM e batching ficam citados como evolução).

## Decisions

### D1 — Candidatos a modelo base
Shortlist a confirmar no momento: modelos instruct de 0,5B–1,5B com licença permissiva e chat template (por exemplo, famílias Qwen2.5 0.5B/1.5B Instruct, com licença Apache-2.0 a verificar na model card de cada variante; SmolLM2 1.7B está acima do limite e fica fora). A decisão final vem da tabela da spec (zero-shot no dev, licença e memória).

### D2 — Formato de treino
Formato conversacional: `system` (instruções e schema), `user` (mensagem) e `assistant` (JSON). Aplica-se o chat template do modelo. Loss **só nos tokens do assistant** (completion-only), através da opção correspondente do TRL na versão fixada; confirmar o nome do parâmetro na documentação. O dry-run verifica a máscara a imprimir os tokens com e sem label.
- *Porquê:* treinar na mensagem do utilizador gasta capacidade a aprender a gerar mensagens de clientes.

### D3 — LoRA/QLoRA
LoRA nas projecções de atenção e MLP (`q,k,v,o,gate,up,down`). Rank 16, alpha 32 e dropout 0.05 como ponto de partida, ajustados só no dev. QLoRA (4-bit NF4, compute bf16/fp16) quando a VRAM o exigir. Explicar no learn: o que é a decomposição de baixo rank, porque é que o alpha/rank escala a actualização e o que a quantização faz aos pesos congelados.

### D4 — Splits por template
Cada exemplo guarda o `scenario_id`/`template_id`. O split é feito por grupos (GroupShuffleSplit). Quase-duplicados são detectados por MinHash ou por similaridade de n-gramas de caracteres com limiar. O hash do teste é registado e verificado antes da avaliação final.

### D5 — Decoding e parsing
Decoding greedy (temperatura 0) com `max_new_tokens` limitado. O parsing é o mesmo para todas as abordagens: extrai o primeiro objecto JSON e valida com Pydantic. Se falhar, conta como inválido. Decoding restringido por gramática fica como experiência opcional, reportada à parte.

### D6 — Baselines
O zero-shot usa o mesmo system prompt. O few-shot usa k=5 exemplos fixos do treino, escolhidos estratificados por intent. As regras usam palavras-chave e regex de entidades, e servem de baseline mínima.

### D7 — Métricas e variância
scikit-learn para macro-F1 e matriz de confusão. O F1 de entidades é por (tipo, valor normalizado). O recall de alto risco é a métrica prioritária, porque falhar uma escalada é o erro mais caro. Com várias seeds, reporta-se média ± desvio e o IC bootstrap sobre os exemplos de teste.

### D8 — Tracking
JSON por run em `runs/<id>/` (config, versões, GPU, VRAM `torch.cuda.max_memory_allocated`, tempo, curvas e métricas). MLflow é opcional.

### D9 — Serviço
FastAPI que carrega o base + adapter com PEFT (`PeftModel.from_pretrained`) quando há modelo. Em modo demo usa `precomputed.jsonl` e marca as respostas. A UI é mínima (Streamlit ou React simples) com o lado a lado.

### D10 — Documentação pedagógica
`docs/learn/` tem: `00-visao-geral`, `01-quando-fazer-fine-tuning`, `02-dataset-sintetico-e-fugas`, `03-tokenizacao-e-chat-templates`, `04-loss-mascarada-completion-only`, `05-lora-por-dentro`, `06-quantizacao-e-qlora`, `07-hiperparametros-e-dev-set`, `08-avaliacao-e-baselines`, `09-servir-adapters` e `10-perguntas-de-entrevista`.

## Risks / Trade-offs

- [Dados sintéticos geram um modelo que só aprende os templates] → Split por template, revisão humana e um conjunto de teste escrito à mão (pequeno) como verificação extra, reportado à parte.
- [Sem GPU local, a iteração é lenta] → Dry-run e testes locais. O notebook concentra o treino.
- [O LoRA pode não ganhar ao few-shot] → É um resultado válido. O critério de paragem e o relatório negativo estão previstos.
- [APIs do TRL/PEFT mudam entre versões] → Versões fixadas no notebook e no `pyproject`, confirmadas na documentação.
- [Licenças dos modelos] → Verificadas na model card antes de escolher e antes de publicar.

## Migration Plan

Projecto novo. O fluxo é: gerar dados → dry-run local → notebook GPU → descarregar o adapter e os resultados → avaliação → API.

## Open Questions

- Que GPU estará disponível (T4, L4 ou outra)? Afecta só a escolha LoRA vs QLoRA e o tamanho do batch, não as specs.
