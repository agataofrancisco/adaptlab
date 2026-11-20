# Spec Delta

## Purpose

Produzir um dataset rotulado, licenciado, documentado e dividido sem contaminação, que torne a avaliação do fine-tuning credível.

## ADDED Requirements

### Requirement: Gerador sintético com labels controlados
O gerador SHALL produzir pelo menos 500 exemplos com `message` e um label JSON (`intent`, `priority`, `entities`, `requires_human`) a partir de cenários e templates. MUST incluir ruído ortográfico, mistura PT/EN, mensagens incompletas, entidades ausentes e casos que exigem escalonamento humano.

#### Scenario: Geração reprodutível
- **WHEN** o gerador corre duas vezes com a mesma seed
- **THEN** produz exactamente o mesmo dataset

#### Scenario: Cobertura de dificuldades
- **WHEN** se consultam as estatísticas
- **THEN** cada intent e cada tipo de dificuldade tem uma contagem mínima configurada

### Requirement: Validação de schema dos labels
Cada exemplo MUST passar a validação do schema de saída, e as entidades MUST aparecer literalmente na mensagem (ou estar marcadas como inferidas).

#### Scenario: Entidade inexistente na mensagem
- **WHEN** um label tem uma entidade `numero` que não está no texto
- **THEN** a validação falha e indica o exemplo

### Requirement: Splits sem fuga
Os splits train/dev/test SHALL ser feitos por cenário/template, de modo que nenhum template do teste apareça no treino. O sistema MUST detectar duplicados exactos e quase-duplicados entre splits.

#### Scenario: Quase-duplicado entre treino e teste
- **WHEN** duas mensagens quase iguais caem em splits diferentes
- **THEN** a verificação de fugas falha e indica o par

#### Scenario: Teste intocado
- **WHEN** se ajustam hiperparâmetros
- **THEN** só o dev é usado, e o teste fica com um hash registado que confirma que não foi alterado

### Requirement: Revisão humana do léxico PT-AO
Uma amostra aleatória SHALL ser revista por um humano quanto à naturalidade do português de Angola. O resultado da revisão (aceites, corrigidos, rejeitados) MUST ficar registado.

#### Scenario: Registo de revisão
- **WHEN** a revisão da amostra termina
- **THEN** existe um ficheiro com a decisão por exemplo e as correcções aplicadas

### Requirement: Dataset card
O dataset SHALL ter um card com a origem, a licença, as classes e distribuições, o método de geração, as limitações e os riscos de enviesamento. Dados privados NÃO SHALL ser incluídos.

#### Scenario: Card completo
- **WHEN** se valida o card
- **THEN** todas as secções obrigatórias estão preenchidas
