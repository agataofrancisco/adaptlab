# 07 · Hiperparâmetros e o dev set

## O problema

Escolher hiperparâmetros (rank, learning rate, épocas) olhando para o conjunto de teste
é **contaminação**: o teste deixa de medir generalização e passa a medir o que
escolhemos. Precisamos de um conjunto separado para decidir, e de um teste que só se
toca no fim.

## O que entra e o que sai

Entra: `configs/train.yaml`, com valores de partida e um critério explícito.

Valores de partida (não ajustados ainda):

| Parâmetro | Valor | Onde |
|---|---|---|
| learning_rate | 2e-4 | `training` |
| num_train_epochs | 3 | `training` |
| per_device_train_batch_size x grad_accum | 8 x 2 (efectivo 16) | `training` |
| warmup_ratio / scheduler | 0,03 / cosine | `training` |
| seed | 42 | `training` |
| eval_strategy | epoch (no dev) | `training` |

Sai: os valores que o treino real usou ficam em `runs/train-*/config.json`, e a curva de
loss de treino e de dev em `metrics.json`. Nesta fase **não há números de treino**:
`TODO: medir` no notebook.

## Como funciona por dentro

- O `SFTTrainer` recebe `eval_dataset=dev` (`adaptlab/training/trainer.py`) e avalia no
  fim de cada época. Essa loss é a única métrica de decisão de treino.
- O conjunto de teste **não** entra no trainer. Só `scripts/evaluate.py --split test --final`
  o lê, e só depois de verificar o `test_sha256` do manifesto.
- O batch efectivo é `per_device_train_batch_size x gradient_accumulation_steps`: 16
  exemplos por passo de optimização, com a memória de 8.
- O warmup aumenta a taxa de aprendizagem de 0 até ao valor máximo nos primeiros 3% dos
  passos; depois o cosseno desce-a. Evita divergência no início, quando o adapter ainda
  está quase a zero.

## Porquê esta escolha

- **Valores de partida padrão** para LoRA em modelos de 0,5B: 2e-4 é comum em QLoRA. O
  batch efectivo de 16 é pequeno o suficiente para ter passos suficientes com 371
  exemplos de treino (cerca de 23 passos por época, 371 / 16).
- **Três épocas**: com poucos dados, mais épocas sobreajustam. A decisão de parar é
  feita pela loss no dev.
- **Seed fixa**: reprodutibilidade de uma run. Variância entre seeds é medida à parte
  (três seeds, quando houver orçamento; ver `specs/model-evaluation`).

## Alternativas

- **Pesquisa em grelha** de rank e learning rate: honesta se for só no dev e reportada
  toda. Não feita: custa várias runs de GPU.
- **Early stopping pelo dev**: útil; ainda não activado (`save_strategy: "no"`). Pode
  entrar quando houver checkpoints.
- **Validação cruzada por cenário**: mais robusta, mas custa k treinos.

## Como falha

- **Teste usado para escolher**: contaminação. Mitigação: o hash do teste é verificado
  antes da avaliação final, e o teste só é lido com `--final`.
- **Dev pequeno**: com 180 exemplos e 51 de escalonamento, a macro-F1 tem variância alta.
  Declarado como limitação; o relatório mostrará o IC bootstrap.
- **Learning rate alto**: a loss oscila ou sobe. Visível nas curvas em `metrics.json`.
- **Sobreajuste aos templates**: a loss no dev desce, mas o dev partilha as ideias do
  treino. Mitigação estrutural: os splits são por cenário (`docs/learn/02`).

## Perguntas de entrevista

**1. Porque é que não usas o teste para escolher hiperparâmetros?**
Porque deixa de ser um teste. A escolha ficaria optimizada para ele e o número reportado
seria optimista.

**2. Para que serve o warmup?**
Para evitar passos grandes no início, quando os gradientes são instáveis e o adapter
ainda não está calibrado.

**3. O que é o batch efectivo e porque o usas?**
É o número de exemplos entre duas actualizações de peso. Com acumulação de gradiente
consegues um batch grande com a memória de um pequeno.

**4. Como decides quando parar de treinar?**
Pela loss ou métrica no dev, por época. Com `save_strategy` desligado não há checkpoint
escolhido; por isso o treino actual corre até ao fim das épocas. É uma limitação
declarada.

**5. Porque é que alpha/r é um hiperparâmetro e não só `alpha`?**
Porque a escala efectiva da actualização é `alpha/r`. Mudar `r` sozinho muda a
magnitude da actualização.
