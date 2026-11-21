# Dataset card: AdaptLab (Mukua Telecom, sintético)

Gerado por `scripts/prepare_data.py` com a configuração `configs/data.yaml`.
Seed: `20261120`. Hash SHA-256 do teste: `8850e7c0a283d4ed81c576a67def9874e96f3936f86918943d6ec6dc5ebd219d`.

## Origem

Dados sintéticos gerados por templates e cenários escritos para o AdaptLab. A empresa
Mukua Telecom é fictícia. Nenhuma mensagem foi copiada de clientes reais.

## Licença

CC-BY-4.0 (escolha do autor do projecto; **proposta a confirmar**). Os dados são sintéticos, criados para este projecto, e não contêm dados pessoais reais.

## Classes e distribuições

Total: **688** exemplos após deduplicação. Intents: `recarga`, `avaria_rede`,
`faturacao`, `portabilidade`, `cancelamento`, `pacote_dados`, `outro`.
Prioridades: `baixa`, `media`, `alta`.

| Split | Exemplos | requires_human=true | Mensagem média (caracteres) |
|---|---|---|---|
| train | 371 | 85 | 62.44 |
| dev | 180 | 51 | 63.37 |
| test | 137 | 11 | 58.53 |

Distribuição de intents por split:

| Intent | train | dev | test |
|---|---|---|---|
| avaria_rede | 60 | 13 | 32 |
| cancelamento | 59 | 29 | 17 |
| faturacao | 60 | 23 | 8 |
| outro | 26 | 19 | 11 |
| pacote_dados | 46 | 32 | 24 |
| portabilidade | 64 | 32 | 13 |
| recarga | 56 | 32 | 32 |

Dificuldades (um exemplo pode ter várias tags):

| Dificuldade (tag) | Exemplos |
|---|---|
| entidade_ausente | 95 |
| escalonamento | 147 |
| incompleto | 46 |
| mistura_pt_en | 183 |
| ruido | 357 |

## Método de geração

1. Catálogo de 28 cenários (4 por intent), cada um com 2 a 3 templates (`adaptlab/data/catalog.py`).
2. Cada template é realizado 16 vezes com valores sorteados (número,
   valor em kwanzas, localidade, data). Ruído PT-AO, erros de escrita e minúsculas só afectam o texto
   literal, nunca os valores das entidades.
3. Segmentos opcionais omitem entidades (`incompleto`, `entidade_ausente`).
4. Labels validados contra o schema Pydantic (`adaptlab/schema.py`). Cada entidade não nula tem de
   aparecer literalmente na mensagem (normalização de espaços e pontuação).
5. Deduplicação: 147 duplicados exactos e 61
   quase-duplicados removidos (Jaccard de 5-gramas >= 0.85).
   Gerados antes da deduplicação: 896.

## Splits e prevenção de fugas

Os splits são por cenário, estratificados por intent (`{'train': 2, 'dev': 1, 'test': 1}`).
Nenhum template aparece em dois splits (`shared_templates = 0`). Quase-duplicados entre
splits são verificados com limiar 0.6: pares encontrados = 0.
O conjunto de teste tem hash registado (`manifest.json`, campo `test_sha256`) e só deve ser
usado na avaliação final.

## Revisão humana

Estado: **PENDENTE**. Amostra de 40 exemplos do treino
(`data/review/sample.jsonl`). Decisões registadas: 0; pendentes:
40; rejeitados: 0; corrigidos: 0.
A revisão do léxico PT-AO por uma pessoa ainda não foi feita. Enquanto estiver pendente, o
naturalismo do português de Angola **não está validado**.

## Limitações

- Dados sintéticos: o modelo pode aprender os templates e não a linguagem real dos clientes.
- Poucos cenários por intent (4) e um teste de tamanho limitado.
- Português de Angola representado por léxico escolhido pelo autor, ainda sem revisão externa.
- Sem mensagens de voz transcritas, sem emojis e sem erros de digitação reais de telemóvel.

## Riscos de enviesamento

- Localidades, nomes de bairros e padrões de número são escolhidos a dedo e não representam a
  distribuição real de clientes.
- Mensagens em inglês são uma fracção artificial, com mistura controlada.
- Escalonamento (`requires_human=true`) é definido pelos autores, não por política da operadora.

## Dados privados

Não foram incluídos dados pessoais, números reais de clientes, facturas reais nem registos
de atendimento. Os números de telefone são gerados aleatoriamente.

## Reprodução

```
.venv\Scripts\python scripts\prepare_data.py --check
.venv\Scripts\python scripts\prepare_data.py
.venv\Scripts\python -m pytest tests -q
```
