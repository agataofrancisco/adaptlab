# Auto-revisão de segurança e fugas (para o supervisor)

Esta é uma **auto-revisão** feita por quem implementou. Não substitui a revisão independente
(tarefa 6.4, que fica por marcar para o supervisor). Cada ponto diz o que foi verificado e como.

## 1. Fugas de dados

- Splits por cenário; nenhum template em dois splits (`manifest.json`, `leakage.shared_templates = 0`).
- Quase-duplicados entre splits: 0 pares acima de 0,6 (`leakage.near_duplicate_pairs_across_splits = 0`).
- Teste artificial que o detector apanha: `tests/test_data.py::test_artificial_leak_is_detected`.
- Hash do teste no manifesto; `evaluate.py --split test` recusa-se sem `--final` e falha se o hash mudar
  (`tests/test_evaluation.py::test_tampered_test_set_is_refused`).
- **Risco residual**: `--final` não tem registo de uso. Nada impede correr o teste mais de uma vez.
  Mitigação pedida ao supervisor: registar cada execução final.
- **Risco residual**: as escolhas de design (limiar, margem) não foram feitas olhando para o teste, mas
  o teste nunca foi lido antes de `evaluate.py --final` (ainda não correu para modelos).

## 2. Licenças

- Modelo base: Qwen2.5-0.5B-Instruct, Apache-2.0 (verificado na API do Hugging Face).
- Portão de publicação (`adaptlab/publication/gate.py`): bloqueia licenças fora de Apache-2.0/MIT,
  critério não aprovado e falta de confirmação. Testado com `llama3.2` (recusado).
- Dataset: CC-BY-4.0 é **proposta** (`configs/data.yaml`), a confirmar pelo autor.
- Exportação sem pesos do base (`tests/test_serving.py::test_export_copies_only_adapter_files`).
- **Risco residual**: a verificação de licença do base é um valor escrito no `configs/train.yaml`, não
  uma consulta automática. Se a licença mudar no Hub, o portão não o nota.

## 3. Honestidade das métricas

- Só há um número medido de modelo: a baseline de regras (dev, macro-F1 0,8351, IC bootstrap).
- Todos os outros valores estão `TODO: medir` (relatório, card, docs). Verificado com `grep "TODO"`.
- Cada número do relatório vem de um `metrics.json` em `runs/`, que está no git (metadados e previsões).
- **Risco residual**: as curvas e o gráfico PNG não existem (matplotlib não instalado). Declarado.

## 4. Modo demo

- Cada resposta de demo traz `mode: precomputed_demo`, `fixture_source` e `note`
  (`tests/test_serving.py::test_demo_fixture_message_is_marked_precomputed`).
- Mensagens fora do conjunto recebem 503 com explicação (`test_demo_new_message_returns_503_with_explanation`).
- As fixtures são previsões da baseline de regras, não do modelo. A UI rotula as colunas como fixture.
- Verificado com a API a correr (HTTP 200 e 503) e com a UI no browser.

## 5. API e entrada

- Entrada validada: 1 a 2000 caracteres, sem mensagem só com espaços (422 se não).
- Saída inválida devolvida como erro estruturado, sem campos inventados.
- Sem autenticação, sem rate limit e sem CORS configurado. `/metrics` e `/health` são públicos.
  **Antes de expor a API fora de localhost**: adicionar autenticação e limites de pedidos.
- A UI chama a API sem credenciais; funciona só em localhost por omissão.
- O Dockerfile corre como utilizador não-root, mas não foi construído (sem Docker local).

## 6. Segredos e dados pessoais

- Varrimento por padrões de tokens e chaves (`hf_`, `api_key=`, `secret=`, `password=`, chaves privadas): sem resultados.
- Nenhum `.env` no repositório; `.env` está no `.gitignore`.
- Nenhum peso, cache ou adapter no git (`git ls-files` limpo). `exports/` e `runs/**/adapter*` estão ignorados.
- Dados sintéticos; números de telefone gerados aleatoriamente; nenhuma mensagem real.

## 7. Execução e dependências

- Versões fixadas em `pyproject.toml`; instalação CPU de torch a partir do índice oficial.
- Testes: 84 a passar. Um teste de dry-run que bloqueava foi corrigido (processo filho com timeout de 900 s
  e `stdin` fechado; ver o relatório de execução).
- **Risco residual**: a causa exacta do bloqueio (leitura de stdin herdado) não foi confirmada por
  inspecção de código; foi observada a correcção.

## Pendências para o supervisor

1. Revisão independente de fugas, licenças e modo demo (6.4).
2. Construir o Docker e correr a CI num runner (6.1).
3. Executar o notebook em GPU e medir o zero-shot, o few-shot e o LoRA (3.3, 3.4, 4.3, 4.4).
4. Revisão humana PT-AO (2.4).
5. Decidir a licença do dataset (CC-BY-4.0 é proposta).
