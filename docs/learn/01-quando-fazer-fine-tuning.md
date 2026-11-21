# 01 · Quando fazer fine-tuning (e quando não fazer)

## O problema

Uma operadora de telecomunicações recebe mensagens de clientes em português de Angola,
com gíria, erros de escrita e mistura com inglês ("my data is down", "top up"). O
sistema precisa de devolver sempre o mesmo tipo de resposta: um JSON com `intent`,
`priority`, `entities` e `requires_human`. A pergunta de engenharia é: **que ferramenta
resolve isto com o menor custo e o maior controlo?**

Há quatro opções habituais, da mais barata à mais cara de manter:

1. **Regras determinísticas** (palavras-chave e regex). Rápidas, explicáveis, mas frágeis
   perante erros de escrita e variantes linguísticas. Servem de linha de base mínima.
2. **Prompt com um modelo grande** (zero-shot ou few-shot). Não treina nada; cada pedido
   paga tokens de instrução e latência de um modelo maior.
3. **Fine-tuning com LoRA/QLoRA num modelo pequeno** (0,5B a 1,5B parâmetros neste projecto).
   Aprende o formato e o vocabulário do domínio uma vez; depois cada pedido é barato.
4. **Fine-tuning completo ou RAG com base de conhecimento**. Mais caro, ou resolve um
   problema que aqui não existe (factos que mudam).

## Quando o fine-tuning compensa

Compensa quando o que se quer ensinar é **formato e classificação num domínio e variante
específicos**, e não factos. Ensinar "o pacote de 5 GB custa 2.000 Kz" por fine-tuning é
má ideia: o preço muda, o modelo fica desactualizado e não há forma fácil de o corrigir.
Ensinar "uma mensagem como 'a net ta fraca no kilamba' é uma avaria de rede" é uma boa
ideia: é um padrão estável, e um modelo pequeno aprende-o bem com poucos exemplos.

Três sinais de que vale a pena tentar:

- A tarefa tem saída estruturada e validável (JSON com schema fixo).
- O vocabulário e a variante linguística são específicos (PT-AO, mistura PT/EN).
- Existe uma forma honesta de medir: um conjunto de teste sem fuga e baselines justas.

## O que o AdaptLab exige antes de aceitar o fine-tuning

O LoRA só é aceite se **ganhar** a alternativas mais simples no mesmo conjunto de teste.
Por isso o `evaluate.py` compara sempre:

| Abordagem | Custo por pedido | Treino | Papel no relatório |
|---|---|---|---|
| Regras | mínimo | nenhum | piso mínimo |
| Base zero-shot | um modelo pequeno com prompt | nenhum | referência sem adaptação |
| Base few-shot | mais tokens (k=5 exemplos) | nenhum | referência forte sem treino |
| LoRA | o modelo pequeno, sem prompt longo | ~minutos em GPU | o candidato |

Se a macro-F1 do LoRA no dev ficar abaixo do melhor few-shot mais a margem configurada, o
pipeline **para** antes de publicar (ver `specs/model-evaluation`). Um resultado negativo
é um resultado válido, e fica registado.

## O que o fine-tuning não resolve aqui

- **Factos que mudam** (preços, cobertura, promoções): continuam a vir de uma base de
  dados ou de uma regra, nunca dos pesos do modelo.
- **Alta carga em produção**: o serviço de demonstração é FastAPI simples. vLLM, batching
  e cache ficam como evolução.
- **Preferências humanas** (RLHF, DPO): fora de âmbito.

## Como isto se reflecte no código

- `scripts/prepare_data.py` cria o dataset e valida-o antes de qualquer treino.
- `scripts/train.py --dry-run` valida a tokenização e a máscara de loss em CPU.
- O notebook GPU (`notebooks/`) faz o treino real.
- `scripts/evaluate.py` compara as quatro abordagens no mesmo teste.

Ver `00-visao-geral.md` para o fluxo completo e `08-avaliacao-e-baselines.md` para o critério
de paragem.

## Como falha

- **Fine-tuning que "aprende os templates"**: o modelo acerta no teste porque o teste repete
  padrões do treino. Mitigação: splits por cenário e verificação de fugas
  (`tests/test_data.py::test_artificial_leak_is_detected`).
- **Comparação injusta**: a baseline few-shot recebe exemplos piores ou outro parsing.
  Mitigação: o mesmo `SYSTEM_PROMPT` (`adaptlab/training/format.py`) e o mesmo parsing
  (`adaptlab/schema.py::parse_label`) em todas as abordagens.
- **Escolher o modelo pela métrica do teste**: contamina a avaliação. Mitigação: só o dev
  guia decisões; o teste tem hash registado (`manifest.json`, campo `test_sha256`).

## Perguntas de entrevista

**1. Quando é que não deves fazer fine-tuning?**
Quando o conhecimento muda com frequência ou quando um prompt bem feito com um modelo
maior resolve o problema com custo aceitável. Fine-tuning ensina padrões estáveis e
formato; não é uma base de dados actualizável.

**2. Porque é que comparas com few-shot e não só com zero-shot?**
Porque few-shot é a alternativa realista sem treino. Se o LoRA só ganha ao zero-shot, o
ganho pode vir simplesmente de os exemplos de treino serem parecidos com o teste. A comparação
com few-shot mede o valor do treino em si.

**3. O que é uma fuga de dados (data leakage) neste projecto e como a evitas?**
Quando informação do teste entra no treino, directa (mesma mensagem) ou indirecta
(mesmo template com outros valores). Evito-a com splits por cenário, deduplicação exacta e
quase-duplicada (Jaccard de 5-gramas) e verificação de fugas que falha o pipeline.

**4. Porque é que o teste tem hash registado?**
Para garantir que não foi alterado depois de se ver resultados (por exemplo, para remover
exemplos difíceis). Qualquer alteração muda o SHA-256 e a avaliação final recusa-se a correr.

**5. Porque é que escolhes modelos de 0,5B a 1,5B?**
Cabem em GPUs de Colab/Kaggle (T4 com 16 GB), treinam em minutos com LoRA, e a inferência
é barata. Acima disso, o custo de treino e de servir não se justifica para uma tarefa de
classificação com saída curta.
