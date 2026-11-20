# Spec Delta

## Purpose

Servir o modelo adaptado por API e numa pequena UI, sem nunca fazer passar resultados pré-calculados por inferência real.

## ADDED Requirements

### Requirement: Endpoint de predição
`POST /predict` SHALL receber a mensagem do cliente e devolver o JSON estruturado validado, o modelo usado (base ou adaptado), a latência e, quando disponível, uma confiança calibrada. Saídas que falham a validação MUST ser devolvidas como erro estruturado com `requires_human: true`.

#### Scenario: Mensagem de avaria
- **WHEN** chega "boa tarde mano, a net ta fraca no kilamba desde ontem"
- **THEN** a resposta é JSON válido com um intent, uma prioridade, as entidades e `requires_human`

#### Scenario: Saída inválida do modelo
- **WHEN** o modelo gera JSON inválido
- **THEN** a API devolve `requires_human: true` e `error: "invalid_model_output"`, sem inventar campos

### Requirement: Modo demo identificado
Sem GPU, o serviço MAY correr em modo demo com outputs pré-calculados para um conjunto fixo de mensagens, e MUST marcar cada resposta com `mode: "precomputed_demo"`. Mensagens fora do conjunto MUST receber uma resposta a explicar que a inferência real não está disponível.

#### Scenario: Mensagem nova em modo demo
- **WHEN** se envia uma mensagem que não está no conjunto pré-calculado
- **THEN** a resposta é `503` com `mode: "precomputed_demo"` e a explicação

### Requirement: Comparação base vs adaptado
A UI SHALL permitir ver lado a lado a saída do modelo base e do adaptado para a mesma mensagem.

#### Scenario: Lado a lado
- **WHEN** um utilizador escreve uma mensagem na UI
- **THEN** vê as duas saídas, a latência de cada uma e as diferenças destacadas

### Requirement: Saúde, métricas e model card
O serviço SHALL expor `GET /health` (modelo carregado, dispositivo, modo), `GET /metrics` (pedidos, latência, taxa de JSON inválido) e `GET /model-card`.

#### Scenario: Health em CPU
- **WHEN** o serviço corre sem GPU
- **THEN** `/health` indica `device: "cpu"` e o modo activo
