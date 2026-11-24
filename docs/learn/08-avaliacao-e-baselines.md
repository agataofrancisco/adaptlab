# 08 · Avaliação e baselines

## O problema

Dizer que um modelo "funciona" não basta. Para o AdaptLab a pergunta é: **o LoRA ganha
às alternativas mais simples, no mesmo conjunto, com o mesmo formato de saída?** Se não
ganhar, o projecto para (ver `specs/model-evaluation`).

## O que entra e o que sai

Entra: o split de dev (180 exemplos) ou de teste (137), a configuração `configs/evaluate.yaml`
e, para o LoRA, uma run de treino (`runs/train-*/adapter`).

Sai: `runs/eval-<abordagem>-<split>-<data>/metrics.json` e `predictions.jsonl`, e o
relatório `reports/evaluation.md` gerado a partir delas.

Resultado medido até agora (baseline de regras, dev, `runs/eval-rules-dev-20261121T065836Z`):

| Métrica | Valor | IC 95% (bootstrap, 1000 amostras) |
|---|---|---|
| Macro-F1 de intent | 0,8351 | [0,7728; 0,8809] |
| F1 de entidades | 0,9696 | (não calculado) |
| JSON válido | 1,0000 | — |
| Recall de alto risco | 0,6275 | (não calculado) |
| Latência média | 0,0005 s | — |

Para as abordagens com modelo, o relatório mostra `TODO: medir`:

| Abordagem | Estado |
|---|---|
| Regras | medido (dev) |
| Base zero-shot | medida exploratória: dev estratificado, n=20; não é comparável à baseline completa de regras |
| Base few-shot (k=5) | TODO: medir |
| LoRA (QLoRA NF4) | TODO: medir (treino GPU ainda não feito; tarefa 4.3 pendente) |

## Como funciona por dentro

- **Mesmo parsing** para todas as abordagens: `parse_label` (`adaptlab/schema.py`).
  Texto à volta do JSON é aceite; JSON inválido ou fora do schema conta como erro.
- **Métricas** (`adaptlab/evaluation/metrics.py`):
  - Macro-F1 com `sklearn.metrics.f1_score`, sobre as classes presentes no gold ou nas
    predições (sem o marcador `__invalid__`). Saída inválida é uma predição errada em todas
    as classes gold.
  - F1 de entidades: micro-F1 sobre pares `(tipo, valor normalizado)`.
  - Recall de alto risco: `requires_human` ou `priority == "alta"`. Saída inválida não
    detecta nada.
- **Bootstrap** (`bootstrap_ci`): reamostra exemplos com reposição e recalcula a métrica.
  O IC de 95% é o intervalo percentil.
- **Baseline de regras** (`adaptlab/evaluation/rules.py`): palavras-chave ordenadas por
  prioridade (portabilidade, cancelamento, faturação, recarga, avaria, pacote) e regex para
  número e valor. Devolve sempre um label válido.
- **Few-shot**: cinco exemplos fixos do treino, um por intent distinto, escolhidos com seed
  (`select_few_shot`) e passados como turnos user/assistant antes da mensagem.
- **Critério de paragem** (`adaptlab/evaluation/stopping.py::decide`): compara a macro-F1 do
  LoRA no dev com a melhor baseline. Aprovado se ganhar pela margem (0,02), bloqueado se não,
  pendente se faltar alguma medição.
- **Teste intocado**: `scripts/evaluate.py --split test` exige `--final` e verifica o hash
  do teste contra `manifest.json` antes de ler (`verify_test_hash`).

## Porquê esta escolha

- **Regras como piso**: se o LoRA não bater um conjunto de regras simples, o problema é
  mal definido ou o dataset não ensina nada que as regras não saibam.
- **Mesmo conjunto e mesmo parsing**: a comparação só é justa assim.
- **Macro-F1 e recall de alto risco**: a macro-F1 não deixa uma classe grande esconder as
  pequenas. O recall de alto risco reflecte o custo: falhar uma escalada é o erro mais caro.
- **IC bootstrap**: com 180 exemplos e 51 de escalonamento, uma diferença de 0,01 na macro-F1
  não é robusta. O relatório mostra o intervalo.

## Como falha

- **Regras falham em erros de escrita**: o ruído do dataset introduz typos ("cncelem",
  "deix"), e a baseline por palavras-chave não os reconhece. Na matriz de confusão do dev,
  17 dos 29 exemplos de `cancelamento` caem em `outro`. Isto é um resultado real, não um bug.
- **Localidades e datas do catálogo**: as regras usam as mesmas listas do gerador, o que
  favorece a baseline. Declarado como limitação.
- **Escalonamento escasso**: dev tem 51 positivos, teste tem 11. O IC do recall de alto
  risco no teste é muito largo; por isso o teste não serve para decidir.
- **Uma só seed**: o relatório diz `seeds: 1` e não apresenta diferenças como robustas.
- **Contaminação**: o teste só é lido com `--final` e com hash verificado. Escolhas feitas
  no dev não usam o teste.
- Testes: `tests/test_evaluation.py` (métricas com exemplos feitos à mão, regras, paragem
  com LoRA que não melhora, teste adulterado).

## Perguntas de entrevista

**1. Porque é que comparas com regras se o objectivo é um LLM?**
Porque é o custo mínimo de referência. Se um conjunto de regras atinge 0,83 de macro-F1,
o modelo tem de ganhar de forma clara para justificar treino, GPU e manutenção.

**2. O que é macro-F1 e porque não accuracy?**
Macro-F1 é a média da F1 por classe, sem ponderar pelo tamanho da classe. Accuracy deixa
uma classe grande (como `outro`) dominar. Aqui queremos que as classes pequenas, como
escalonamento, também funcionem.

**3. Como tratas uma saída inválida na métrica?**
Como erro. Não a removo do denominador, porque isso premiaria modelos que falham muito.
Por isso a taxa de JSON válido é uma métrica separada e a macro-F1 conta-a como errada.

**4. Porque é que a métrica de risco é recall e não precisão?**
Porque o erro caro é deixar passar uma escalada. Um falso positivo custa uma revisão
humana extra; um falso negativo custa um cliente sem resposta.

**5. Como sabes se uma diferença de 0,01 é real?**
Não sei só com a diferença. Uso o IC bootstrap sobre os exemplos avaliados e, quando
houver mais seeds, a variância entre runs. Com um único conjunto de 180 exemplos, a
diferença de 0,01 fica dentro do intervalo.

**6. Porque é que o teste tem um hash?**
Para detectar alterações depois de se verem resultados. Se o teste mudar, a avaliação
final recusa-se a correr.

**7. O que acontece se o LoRA não ganhar?**
O critério bloqueia a exportação para publicação. O relatório fica com o resultado negativo
e com as hipóteses (rank, learning rate, mais dados). Um resultado negativo é válido.

## Teste intocado: registo auditável

- Cada `--final` acrescenta uma linha a `runs/final_eval_log.jsonl` **antes** de gerar predições: data UTC,
  número de uso, commit git, hash SHA-256 da configuração, modelo/adapter, hash do teste, número de
  exemplos e motivo (se for uma nova leitura).
- Ordem das verificações: (1) recusa se for uma nova leitura sem `--allow-retest` (sem tocar no teste);
  (2) lê o ficheiro e verifica o hash contra o manifesto; (3) se o hash mudou, recusa (código 4) sem registar;
  (4) regista; (5) gera predições.
- O relatório (`reports/evaluation.md`, secção "Uso do conjunto de teste") mostra o número de leituras e o
  motivo de cada nova leitura.
- Testes: `tests/test_final_log.py` (primeira leitura registada, segunda recusada, motivo registado,
  motivo vazio recusado, teste alterado recusado sem registo, contagem no relatório).

## Medição zero-shot exploratória — 2026-11-24

A run `eval-zero_shot-dev-20261124T064440766696Z-b244de` avaliou o
`Qwen/Qwen2.5-0.5B-Instruct`, revisão
`7ae557604adf67be50417f59c2c2f167def9a775`, em CPU com `bfloat16`. Usou uma amostra
estratificada por intenção de 20 exemplos do split `dev`, seed 42. Os dados completos e
as predições estão em `runs/eval-zero_shot-dev-20261124T064440766696Z-b244de/`.

| Métrica | Resultado | IC bootstrap 95% |
|---|---:|---:|
| Macro-F1 de intenção | 0,1524 | 0,0625–0,2568 |
| F1 de entidades | 0,4615 | — |
| Precisão de entidades | 0,5217 | — |
| Recall de entidades | 0,4138 | — |
| JSON válido | 90% | — |
| Recall de alto risco (6 positivos) | 0,1667 | 0–0,5 |
| Latência média em CPU | 12,8771 s | — |

Estes intervalos bootstrap não compensam a amostra pequena. Seis exemplos de alto risco
não permitem concluir que o recall real seja adequado. A medição serve para detectar
problemas e preparar uma comparação maior; não declara o modelo pronto nem permite
compará-lo directamente com a baseline de regras, que usou o dev completo.

A tentativa equivalente do Qwen 1.5B não concluiu o download e não gerou métricas.
Ainda é necessário comparar os dois candidatos com a mesma amostra, prompt e parser,
treinar e avaliar o adapter no mesmo dev e, depois de congelar a decisão, avaliar uma vez
no test.
