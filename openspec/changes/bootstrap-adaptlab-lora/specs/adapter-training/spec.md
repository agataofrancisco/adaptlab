# Spec Delta

## Purpose

Treinar um adapter LoRA/QLoRA sobre um modelo pequeno de forma reproduzível, documentada e possível sem GPU local.

## ADDED Requirements

### Requirement: Selecção justificada do modelo base
A escolha do modelo base (0,5B a 1,5B parâmetros) SHALL ser documentada com a licença, o suporte de chat template, a memória estimada e os resultados zero-shot no dev de pelo menos dois candidatos.

#### Scenario: Registo da decisão
- **WHEN** a selecção termina
- **THEN** existe uma tabela de candidatos com licença, tamanho, VRAM estimada e macro-F1 zero-shot no dev

### Requirement: Dry-run em CPU
`train.py --dry-run` SHALL validar o dataset, aplicar o chat template, tokenizar, verificar os comprimentos e a máscara de labels e correr um ou dois passos num modelo minúsculo ou numa amostra, sem precisar de GPU.

#### Scenario: Máquina sem CUDA
- **WHEN** se executa o dry-run num PC sem GPU
- **THEN** termina com sucesso e reporta tokens por exemplo, truncagens e se os tokens de label estão mascarados correctamente

#### Scenario: Treino real sem CUDA
- **WHEN** se tenta o treino QLoRA real sem CUDA
- **THEN** o script falha com uma mensagem clara a indicar o notebook GPU

### Requirement: Treino reproduzível
O treino SHALL ser configurado por ficheiro (modelo, revisão/commit, rank, alpha, dropout, módulos alvo, learning rate, épocas, batch, seed, quantização) e MUST registar a configuração, as versões das bibliotecas, a GPU, a VRAM máxima medida, o tempo e as curvas de loss de treino e de validação.

#### Scenario: Registo de uma run
- **WHEN** uma run de treino termina
- **THEN** o tracking contém a configuração completa, a VRAM máxima, o tempo e as curvas

### Requirement: Notebook GPU
O projecto SHALL incluir um notebook Colab/Kaggle com versões fixadas que corre o treino e a avaliação de ponta a ponta e exporta o adapter e os resultados.

#### Scenario: Execução do notebook
- **WHEN** o notebook corre numa GPU suportada
- **THEN** produz o adapter, o relatório de avaliação e o ficheiro de tracking
