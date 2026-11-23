# Estudo de caso: AdaptLab

## Contexto

Uma operadora fictícia (Mukua Telecom) quer encaminhar mensagens de clientes em português de Angola,
com gíria e mistura com inglês. A pergunta do projecto: um modelo pequeno com LoRA bate um sistema de
regras e um modelo maior com prompt, com custo e latência aceitáveis?

## O que foi feito (medido)

1. **Dataset sintético**: 688 exemplos após deduplicação, 28 cenários, 56 templates, splits por cenário,
   zero fugas detectadas. Ver `data/DATASET_CARD.md`.
2. **Dry-run de treino em CPU**: tokenização, máscara de loss e dois passos LoRA com o tokenizer real
   do Qwen2.5-0.5B-Instruct e um modelo tiny local. 191 tokens de prompt mascarados, 51 com label.
3. **Baseline de regras no dev**: macro-F1 0,8351 (IC 95% [0,7728; 0,8809]), F1 de entidades 0,9696,
   recall de alto risco 0,6275. O erro principal são erros de escrita do ruído.
4. **API e UI em modo demo**: testadas com HTTP e no browser. Respostas de fixture marcadas.
5. **Portão de publicação**: bloqueia por critério de paragem pendente (estado real).
6. **Testes**: 84 testes na suite, a passar na última execução completa (exit 0), mais
   ruff sem erros.

## O que não foi feito (e porquê)

- **Treino QLoRA real e avaliação do LoRA**: precisam de GPU CUDA. O notebook está escrito e validado
  sintacticamente, mas não foi executado. Sem estes números, não há resposta à pergunta principal.
- **Zero-shot e few-shot do modelo base**: precisam de descarregar ~1 GB de pesos e de geração.
  Implementados e testados com geradores falsos; sem medição real.
- **Revisão humana PT-AO**: uma pessoa tem de a fazer. O tooling está pronto.
- **Build do Docker e execução da CI**: não há Docker local nem runner do GitHub.
- **Revisão independente** (tarefa 6.4): não pode ser feita por quem implementou.

## Resultado até agora

Com o que está medido, o único candidato medido é a baseline de regras, e ela já atinge 0,83 de
macro-F1 no dev. A aposta do LoRA é ganhar a esse número **e** ao few-shot. Se não ganhar pela
margem de 0,02, o projecto para e o relatório regista o resultado negativo. É um resultado válido.

## Lições

- Um teste de fuga artificial vale mais do que um "zero fugas" sem prova de que o detector funciona.
- Uma tag que está em 100% dos exemplos não informa nada (`entidade_ausente` foi corrigida).
- Usar o caminho real do TRL no dry-run (em vez de uma reimplementação) torna a validação relevante para o treino.
- Uma baseline de regras é barata e dá um número real cedo. Ajuda a não sobrevalorizar o LoRA.
- Portas de rede no PC de desenvolvimento estavam ocupadas por outros serviços; é preciso confirmar
  que a API responde antes de testar.
