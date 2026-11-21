# 02 · Dataset sintético e fugas de dados

## O problema

Não há dados reais de clientes da Mukua Telecom (é uma operadora fictícia), e os
dados reais teriam dados pessoais. Precisamos de um dataset rotulado, reproduzível,
e dividido de forma que a avaliação seja credível. O risco principal é a **fuga de
dados**: o modelo "acerta" no teste porque viu o mesmo exemplo, ou o mesmo template
com outros valores, durante o treino.

## O que entra e o que sai

Entra: `configs/data.yaml` (seed 20261120, 16 realizações por template) e o catálogo de
28 cenários com 56 templates (`adaptlab/data/catalog.py`).

Sai (execução real de `scripts/prepare_data.py`):

```
{"train": 371, "dev": 180, "test": 137}
revisao: pendente (pendentes=40)
escrito em .../data/processed e .../data/DATASET_CARD.md
```

Estatísticas medidas no `data/processed/manifest.json`:

| Split | Exemplos | requires_human=true | Mensagem média (caracteres) |
|---|---|---|---|
| train | 371 | 85 | 62,4 |
| dev | 180 | 51 | 63,4 |
| test | 137 | 11 | 58,5 |

Geração: 896 exemplos brutos. Removidos: 147 duplicados exactos e 61 quase-duplicados
(Jaccard de 5-gramas >= 0,85). Restam 688.

Um exemplo real de treino (sem ruído aplicado):

```
boa, fiz uma recarga de 1 mil kwanzas e o saldo nao mudou
{'intent': 'recarga', 'priority': 'baixa',
 'entities': {'numero': None, 'valor': '1 mil kwanzas', 'localidade': None, 'data': None},
 'requires_human': False}
```

Note-se o `nao` (sem acento) e o `valor` copiado literalmente da mensagem.

## Como funciona por dentro

- **Geração determinística** (`adaptlab/data/generator.py::generate`): um único
  `random.Random` por (seed, cenário, template) controla valores, ruído e omissões.
  Mesma seed, mesmo dataset (`tests/test_data.py::test_generation_is_reproducible`).
- **Ruído só no texto literal**: `add_noise` altera o fragmento do template, nunca o
  valor da entidade. Por isso a validação "entidade literal na mensagem" é exacta.
- **Validação do label** (`adaptlab/data/quality.py::validate_example`): primeiro o
  schema Pydantic (`adaptlab/schema.py::Label`, `extra="forbid"`), depois cada entidade
  não nula tem de aparecer na mensagem após normalização (minúsculas, sem acentos, sem
  espaços nem pontos). Exemplo medido:

  ```
  validate bad entity: ["entities.numero: '999 999 999' not in message"]
  ```

- **Deduplicação**: exacta por texto normalizado; quase-duplicada por Jaccard sobre
  conjuntos de 5-gramas de caracteres (`shingles`, `jaccard`). É O(n^2), aceitável para
  menos de mil exemplos.
- **Splits por cenário, estratificados por intent** (`adaptlab/data/splits.py`): para
  cada intent, 2 cenários vão para treino, 1 para dev e 1 para teste. Um template nunca
  está em dois splits.
- **Detecção de fugas** (`quality.py::check_leakage`): templates partilhados e pares com
  Jaccard >= 0,6 entre splits diferentes. O pipeline **falha** se houver qualquer um dos
  dois. Teste artificial medido, com um exemplo do teste copiado para o treino e com
  " ok" acrescentado:

  ```
  pair: [('LEAK', 'recarga_saldo_nao_entrou/t0/r00', 0.9464)]
  ```

  Resultado real do pipeline: `shared_templates = 0`, `near_duplicate_pairs_across_splits = 0`.
- **Hash do teste**: `canonical_hash` calcula SHA-256 sobre o JSON canónico ordenado por
  id. Está no manifesto (`test_sha256 = 8850e7c0...5ebd219d`). A avaliação final deve
  verificar este hash antes de correr (ver `docs/learn/08`).
- **Revisão humana**: 40 exemplos do treino são amostrados (`data/review/sample.jsonl`).
  `apply_decisions` remove os rejeitados e troca os corrigidos (com validação de
  entidades). Pendentes ficam registados. Nada disto é feito pelo código: o estado
  actual é **pendente**.

## Porquê esta escolha

- Splits por cenário, e não aleatórios: um split aleatório deixaria o mesmo template em
  treino e teste, e a métrica mediria memorização de templates.
- Quase-duplicados por 5-gramas: detecta reformulações com um valor diferente, que a
  deduplicação exacta não apanha.
- Tags de dificuldade (`ruido`, `mistura_pt_en`, `incompleto`, `entidade_ausente`,
  `escalonamento`): permitem ver que tipos de erro o modelo comete.

Bug encontrado durante a implementação: a tag `entidade_ausente` estava em 100% dos
exemplos, porque o dicionário de entidades tem sempre as quatro chaves. A tag passou a
marcar só placeholders omitidos por segmentos opcionais ou mensagens sem entidades
(`tests/test_data.py::test_missing_entity_tag_is_not_universal`).

## Alternativas

- Dataset real anonimizado: melhor para o léxico, mas exige dados e licença que não temos.
- `GroupShuffleSplit` do scikit-learn: equivalente ao que fazemos, sem a estratificação
  por intent, que aqui é necessária porque há poucos cenários.
- MinHash/LSH para quase-duplicados: escalaria melhor, mas para 688 exemplos o O(n^2)
  exacto é mais simples e mais fácil de explicar.

## Como falha

- Fuga por reformulação: coberta pelo limiar de 0,6 e pelo teste acima.
- Escalonamento escasso no teste: o teste tem só 11 exemplos com `requires_human=true`.
  Qualquer métrica de recall de alto risco sobre 11 positivos tem intervalo de confiança
  muito largo. Isto é uma limitação declarada no dataset card, não resolvida.
- Localidade ausente no dev: o cenário de dev não usa localidades. Isto é uma limitação
  do catálogo.
- Revisão pendente: o léxico PT-AO ainda não foi validado por uma pessoa.
- Testes: `tests/test_data.py` (reprodutibilidade, validação, fuga artificial, revisão,
  card) e `tests/test_schema.py` (parsing defensivo).

## Perguntas de entrevista

**1. Porque é que o split é por cenário e não por exemplo?**
Porque exemplos do mesmo template são quase iguais. Um split por exemplo põe variantes
do mesmo template nos dois lados, e o modelo parece bom por memorização.

**2. Como sabes que não há fuga?**
Verifico três coisas: nenhum template em dois splits, nenhum par com Jaccard de 5-gramas
acima de 0,6 entre splits, e o hash do teste fica registado. Se qualquer uma falhar, o
pipeline não escreve os ficheiros.

**3. Porque é que o ruído não toca nas entidades?**
Porque a validação "entidade literal na mensagem" precisa de saber o valor exacto.
Se o ruído alterasse "1.500 Kz" para "1500kz", o label deixaria de ser verificável sem
normalização extra. Mantive o ruído no texto do template.

**4. O que é Jaccard de n-gramas e porque n=5?**
É |A ∩ B| / |A ∪ B| sobre os conjuntos de substrings de n caracteres. Com n=5 apanha
variações de palavras e pontuação, e não dá falsos positivos por palavras comuns como
"de" ou "o".

**5. Porque é que um teste de fuga artificial é necessário?**
Para provar que o detector funciona. O teste injecta uma cópia de um exemplo no teste e
espera que o detector a encontre (`test_artificial_leak_is_detected`). Sem isto, "zero
fugas" poderia significar um detector partido.

**6. Qual é a limitação mais séria do dataset?**
Ser sintético e escrito pelo autor: o modelo pode aprender os templates e não a fala real
dos clientes. Por isso o dataset card declara a revisão PT-AO como pendente.
</content>
