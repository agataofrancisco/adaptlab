# 05 · LoRA por dentro

## O problema

Treinar todos os pesos de um modelo de 0,5B parâmetros exige memória para os pesos, os
gradientes e o estado do optimizador (Adam guarda 2 valores por parâmetro). Para
fine-tuning de uma tarefa pequena, isso é caro e desnecessário. Queremos adaptar o
comportamento mudando uma fracção pequena dos parâmetros.

## O que entra e o que sai

Entra: o modelo base (congelado) e a configuração LoRA de `configs/train.yaml`:
`r: 16`, `alpha: 32`, `dropout: 0.05`, módulos `q,k,v,o,gate,up,down`.

Sai (dry-run real, modelo tiny, mesma configuração LoRA):

```
trainable_parameters: {"trainable": 8192, "total": 4870848}
```

No modelo real (Qwen2.5-0.5B) o número de parâmetros treináveis é medido na run GPU e
não está aqui: `TODO: medir` no treino real (campo a registar em `metrics.json`).

## Como funciona por dentro

Para uma matriz de pesos `W` de tamanho `d_out x d_in`, LoRA congela `W` e aprende
duas matrizes pequenas:

```
W' = W + (alpha / r) * B A
A: r x d_in   (inicializada com ruído aleatório)
B: d_out x r  (inicializada a zero)
```

- Como `B` começa a zero, `W' = W` no início: o modelo arranca exactamente como o base.
- `r` é o **rank**: a actualização tem rank no máximo `r`. Com `r=16` e uma camada de
  `896 x 896`, treinamos `2 x 896 x 16 = 28 672` valores em vez de `802 816`.
- `alpha / r` é um factor de escala. Com `alpha=32` e `r=16`, a escala é 2. Se mudares
  `r` sem mudar `alpha`, mudas a magnitude da actualização. Por isso o par é tratado
  como um todo.
- PEFT aplica isto aos módulos listados em `target_modules` (`peft.LoraConfig`, usado em
  `adaptlab/training/dryrun.py` e `trainer.py`).

Porque é que funciona: a adaptação de uma tarefa costuma estar num subespaço de baixa
dimensão. Uma actualização de rank baixo captura a mudança sem reescrever o modelo.

## Porquê esta escolha

- **Módulos de atenção e MLP** (`q,k,v,o,gate,up,down`): cobre quase todos os pesos
  lineares do bloco. É o padrão que se usa em modelos Qwen/Llama.
- **r=16, alpha=32**: ponto de partida conservador, de uso comum. Não foi ajustado: a
  escolha final vem do dev (`docs/learn/07`).
- **Dropout 0.05**: regularização leve sobre a entrada do adapter.

## Alternativas

- **Fine-tuning completo**: mais memória, mais risco de esquecer capacidades gerais, e
  um checkpoint inteiro para distribuir.
- **Prefix/prompt tuning**: menos parâmetros, mas geralmente pior em tarefas de formato
  com poucos dados.
- **Adapters em série**: adicionam latência na inferência; LoRA pode ser fundido nos
  pesos (`merge_and_unload`) se for preciso.

## Como falha

- **Rank demasiado baixo**: o modelo não aprende o formato e a macro-F1 fica baixa.
  Detectável no dev.
- **Alpha mal escolhido**: a actualização fica grande ou pequena demais; a loss
  diverge ou não desce. Detectável nas curvas de loss.
- **Módulos errados**: se `target_modules` não existir no modelo, o PEFT falha ao
  criar o adapter. Coberto pelo dry-run, que cria o adapter no modelo tiny com os
  mesmos nomes.
- Testes: `tests/test_training.py` (configuração inválida, dry-run com trainer).

## Perguntas de entrevista

**1. Porque é que `B` começa a zero e não `A`?**
Para que o produto `B A` seja zero no início e o modelo comece igual ao base. Inicializar
ambos a zero impediria o gradiente de fluir para `A`.

**2. O que significa o rank r?**
É o rank máximo da actualização `B A`. Quanto maior, mais capacidade e mais parâmetros
treináveis, e mais risco de sobreajuste com poucos dados.

**3. Porque dividir alpha por r?**
Para que a escala da actualização não dependa do rank. Ao mudar `r`, mantém-se a
magnitude relativa, e a taxa de aprendizagem continua a fazer sentido.

**4. Quantos parâmetros treináveis tem LoRA num modelo de 0,5B?**
Depende de `r` e dos módulos. Para `r=16` nos sete módulos, é uma pequena fracção do total
(ver o número medido no treino real). No modelo tiny do dry-run a fracção é 8192 em 4,87M.

**5. Porque é que não se funde o adapter antes de servir?**
Pode fundir-se (`merge_and_unload`) para latência mínima. Mantemos o adapter separado
para comparar base e adaptado na mesma sessão, e para publicar só o adapter (ver `09`).
