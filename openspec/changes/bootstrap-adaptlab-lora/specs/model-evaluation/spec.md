# Spec Delta

## Purpose

Determinar com uma comparação justa se o adapter LoRA é melhor do que as alternativas mais simples e parar se não for.

## ADDED Requirements

### Requirement: Baselines obrigatórias
O `evaluate.py` SHALL avaliar no mesmo conjunto de teste: o modelo base zero-shot com um prompt fixo, o modelo base few-shot com exemplos fixos do treino, o modelo com LoRA e, opcionalmente, regras determinísticas.

#### Scenario: Comparação completa
- **WHEN** a avaliação final corre
- **THEN** o relatório tem uma linha por abordagem com as mesmas métricas e o mesmo conjunto de teste

### Requirement: Métricas
A avaliação SHALL medir a macro-F1 de intent, o F1 de entidades, a taxa de JSON válido, o recall de pedidos de alto risco (`requires_human` ou prioridade alta), a latência por pedido e o custo estimado de inferência, e produzir a matriz de confusão e os erros por tipo.

#### Scenario: JSON inválido
- **WHEN** um modelo devolve JSON inválido num exemplo
- **THEN** o exemplo conta como erro em todos os campos e para a taxa de validade

### Requirement: Variância entre seeds
Quando o orçamento o permite, o treino SHALL ser repetido com pelo menos 3 seeds, e o relatório MUST mostrar a média e o desvio. Com uma única seed, o relatório MUST dizê-lo explicitamente.

#### Scenario: Uma só seed
- **WHEN** só foi possível treinar uma vez
- **THEN** o relatório indica `seeds: 1` e não apresenta a diferença como robusta

### Requirement: Critério de paragem
Se o LoRA não superar a melhor baseline no dev pela margem mínima configurada, o processo SHALL parar antes de publicar, e o relatório MUST registar o resultado negativo e as hipóteses.

#### Scenario: LoRA não melhora
- **WHEN** a macro-F1 do LoRA no dev fica abaixo do few-shot mais a margem
- **THEN** o pipeline não exporta para publicação e o relatório explica o resultado

### Requirement: Relatório sem números inventados
O relatório SHALL conter apenas valores produzidos pelas runs registadas, com gráficos de loss, matriz de confusão, tamanho do adapter, VRAM máxima, tempo de treino e quantização usada. Os valores não medidos MUST ficar `TODO: medir`.

#### Scenario: Relatório gerado
- **WHEN** a avaliação termina
- **THEN** cada número do relatório tem a run de origem associada
