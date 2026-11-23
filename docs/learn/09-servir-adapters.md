# 09 · Servir adapters

## O problema

Um adapter LoRA não é um modelo: é um ficheiro pequeno que se aplica por cima de um modelo
base. Servi-lo exige carregar o base, aplicar o adapter, validar a saída e nunca apresentar
uma resposta pré-calculada como se fosse inferência. O último ponto é o que mais importa
para a honestidade do projecto.

## O que entra e o que sai

Entra: uma mensagem (texto), via `POST /predict?model=base|adapted`.

Sai, medido com a API a correr (`uvicorn api.main:app`, modo demo, CPU):

```
GET /health
{"status":"ok","mode":"precomputed_demo","device":"cpu","model_loaded":false,"models":["regras_fixture"]}

POST /predict?model=adapted   (mensagem presente nas fixtures)
{"mode":"precomputed_demo","model":"regras_fixture","model_requested":"adapted",
 "label":{"entities":{"data":"esta manhã","localidade":null,"numero":"925 504 121","valor":"300 Kz"},
          "intent":"recarga","priority":"baixa","requires_human":false},
 "error":null,"requires_human":false,"latency_ms":null,
 "note":"Resposta pré-calculada (fixture). Não é uma inferência do modelo adaptado."}  -> HTTP 200

POST /predict?model=adapted   (mensagem nova)
{"mode":"precomputed_demo","error":"demo_message_not_precomputed",
 "explanation":"Modo demo sem GPU: ... Corre a API com ADAPTLAB_MODE=real ...","requires_human":true}  -> HTTP 503

GET /model-card  -> 404 enquanto não há reports/model_card.md   (depois passa a 200)
GET /metrics     -> {"requests_total":3,"requests_by_status":{"200":1,"503":2}, ...}
```

A UI (`ui/app.py`) foi verificada no browser com a mesma mensagem: duas colunas, ambas
rotuladas como "fixture (não é o modelo)", com as diferenças entre campos.

## Como funciona por dentro

- **Carregamento** (`adaptlab/serving/loader.py`): o base pelo commit fixado em
  `configs/train.yaml` e o adapter com `PeftModel.from_pretrained`
  (`adaptlab/evaluation/generation.py`). Dois geradores, `base` e `adapted`, para comparação.
- **Predição** (`adaptlab/serving/predictor.py`): `build_prompt` (o mesmo `SYSTEM_PROMPT` do
  treino), geração greedy, depois `parse_label`. Se o parsing falhar, a resposta é
  `error: "invalid_model_output"` com `requires_human: true` e `label: null`. Nenhum campo é
  inventado.
- **Modo demo** (`DemoPredictor`): lê `data/demo/precomputed.jsonl`. Cada linha tem `source`,
  o caminho da run medida de onde veio. Nunca chama um modelo.
- **API** (`api/main.py`): `create_app(predictor)` recebe o predictor, o que permite testar
  com um gerador falso. O modo é escolhido por `ADAPTLAB_MODE` (`demo` por omissão).
- **Validação de entrada**: a mensagem é cortada e tem de ter entre 1 e 2000 caracteres
  (`PredictRequest`). Mensagens vazias dão 422.
- **Métricas**: contadores em memória (pedidos, estados, JSON inválido, latência média).
  Não substituem um sistema de observabilidade.

## Porquê esta escolha

- **Demo com fixtures rotuladas** e não com respostas "de modelo": o demo funciona sem GPU e
  nunca se confunde com inferência. A fixture vem da baseline de regras, medida no dev, e a
  própria API o diz (`fixture_source`).
- **Saída inválida como dado, não como excepção**: a UI e quem chama a API conseguem tratar o
  caso com `requires_human`.
- **Predictor injectado**: os testes correm sem modelo e sem rede.

## Alternativas

- **vLLM ou TGI** para servir com batching e cache de KV: a evolução natural. Não usado porque
  o volume é baixo e o objectivo é demonstrar o pipeline (ver `docs/decisions.md`).
- **Fundir o adapter** (`merge_and_unload`) antes de servir: menos latência por pedido, mas
  perde a comparação lado a lado na mesma sessão.
- **Decoding restrito por gramática** (JSON guiado): garante JSON válido. Fica como experiência
  opcional, reportada à parte (ver `docs/learn/06` e decisões).

## Como falha

- **Modelo ausente**: `ADAPTLAB_ADAPTER_DIR` sem `adapter_config.json` falha ao arrancar, com
  mensagem clara (`loader.py`).
- **Entrada hostil ou enorme**: rejeitada pelo schema (422), sem chegar ao modelo.
- **Porta ocupada**: no PC de desenvolvimento a porta 8000 já estava em uso por outro
  serviço; a API foi arrancada numa porta livre. Não é um bug da API, mas convém saber.
- **Demo mal interpretado**: coberto pela marcação `precomputed_demo`, pelo `note` e pela
  rotulagem das colunas na UI.
- Testes: `tests/test_serving.py` (demo, 503, saída inválida, entrada inválida, métricas,
  portão de publicação, exportação sem pesos do base).

## Perguntas de entrevista

**1. Como serves um adapter LoRA?**
Carregas o modelo base e aplicas o adapter com PEFT (`PeftModel.from_pretrained`). Podes
fundir os pesos (`merge_and_unload`) se quiseres latência mínima e não precisares de
comparar com o base.

**2. Porque é que a API devolve 200 numa saída inválida?**
Porque é uma resposta estruturada, não uma falha do servidor. O cliente recebe
`error: "invalid_model_output"` e `requires_human: true`, e pode encaminhar o caso para uma
pessoa. Um 500 faria o cliente tratar como falha de infra-estrutura.

**3. Como garantes que um demo não é confundido com inferência?**
Marcação explícita em cada resposta (`mode: "precomputed_demo"`), `fixture_source` com a run
de origem, `note` na resposta e rotulagem das colunas na UI. Mensagens fora do conjunto
recebem 503, não uma resposta inventada.

**4. Porque é que o predictor é injectado na app?**
Para testar a API sem carregar um modelo. Em produção o predictor é construído pela
configuração; nos testes, com um gerador falso que devolve texto controlado.

**5. Porque é que o tokenizer tem de ser o mesmo do treino?**
Os ids dos tokens mudam com o tokenizer. Um adapter treinado com um tokenizer e servido com
outro produz texto sem sentido. Por isso o commit do modelo base é fixado no config e usado
nos dois lados.

**6. O que é `requires_human` e porque não é só a prioridade?**
É o sinal de encaminhamento. Escalar depende também de palavras como "queixa" ou "dados
pessoais", que nem sempre mudam a prioridade. A API marca `requires_human` como verdadeiro
quando o label diz isso ou quando a prioridade é alta.
