# Spec Delta

## Purpose

Documentar e partilhar o adapter de forma responsável, respeitando as licenças do modelo base e dos dados.

## ADDED Requirements

### Requirement: Model card
O projecto SHALL produzir um model card com o modelo base e a revisão exacta, os hiperparâmetros, o dataset e a licença, as métricas da avaliação (com a run de origem), a utilização prevista e não prevista, as limitações, os riscos de enviesamento e como reproduzir.

#### Scenario: Card consultável
- **WHEN** se chama `GET /model-card`
- **THEN** o conteúdo corresponde ao card gerado pela última avaliação aprovada

### Requirement: Exportação do adapter
`export_adapter.py` SHALL exportar apenas os pesos do adapter e a configuração PEFT, com hash, e NÃO SHALL incluir os pesos do modelo base, a menos que a licença permita redistribuição e isso seja pedido explicitamente.

#### Scenario: Exportação por omissão
- **WHEN** se exporta o adapter
- **THEN** a pasta contém só o adapter, a configuração, o card e o hash, e o tamanho do adapter é registado

### Requirement: Publicação condicionada
A publicação no Hugging Face SHALL exigir a verificação das licenças do modelo base e do dataset, a aprovação no critério de paragem e uma confirmação manual. Sem estas condições, a publicação MUST ser bloqueada.

#### Scenario: Licença não permite
- **WHEN** a licença do modelo base proíbe a redistribuição de derivados
- **THEN** o script recusa publicar e explica o motivo
