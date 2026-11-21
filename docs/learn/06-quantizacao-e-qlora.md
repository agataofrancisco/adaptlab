# 06 · Quantização e QLoRA

## O problema

Os pesos de um modelo em bf16 ocupam 2 bytes por parâmetro. Um modelo de 1,5B ocupa
~3 GB só em pesos, antes de activações e optimizador. Numa T4 (16 GB) é viável, mas com
margem curta. QLoRA reduz a memória dos pesos congelados sem tocar nos adapters.

## O que entra e o que sai

Entra: a configuração `quantization: nf4` (`configs/train_gpu.yaml`) ou `none`
(`configs/train.yaml`).

Sai, nesta máquina (sem CUDA):

```
ERRO de ambiente: QLoRA (quantization: nf4) precisa de GPU CUDA. Este PC não tem CUDA.
Corre o treino real no notebook GPU (notebooks/train_colab.ipynb, Colab/Kaggle) ou usa
'quantization: none' em configs/train.yaml.
```

(É o comportamento pedido: falha clara, sem fallback silencioso. Testado em
`tests/test_training.py::test_qlora_without_cuda_fails_with_clear_message`.)

## Como funciona por dentro

- **Quantização linear**: cada peso `w` é guardado como um inteiro de poucos bits, com
  uma escala. A reconstrução é `w ≈ escala x q`, com `q` em 4 bits.
- **NF4** (NormalFloat 4): os 16 níveis são escolhidos para que cada nível tenha a mesma
  probabilidade sob uma distribuição normal. Os pesos de uma rede costumam ser
  aproximadamente normais, por isso o erro de quantização é menor do que com níveis
  uniformes.
- **Double quantization**: as próprias escalas são quantizadas (`bnb_4bit_use_double_quant=True`),
  poupando ~0,4 bits por parâmetro.
- **Computação**: na passagem para a frente, os pesos 4-bit são dequantizados para
  `bnb_4bit_compute_dtype` (bf16 ou fp16) e a multiplicação corre nesse tipo. Os
  gradientes só existem nos adapters LoRA, que estão em precisão mais alta.

Ou seja: os pesos base ficam em 4 bits e congelados; o LoRA treina em precisão normal.
É isso que QLoRA significa.

Código: `adaptlab/training/trainer.py`, bloco `BitsAndBytesConfig`. A guarda de CUDA e
bitsandbytes está em `adaptlab/training/tracking.py::require_quantization_support`.

## Porquê esta escolha

- **NF4 + double quant + compute bf16/fp16**: é a configuração recomendada no artigo
  original do QLoRA e a default da integração do transformers.
- **Quantização só no treino real**: o dry-run não precisa dela, e bitsandbytes só tem
  suporte CUDA. Por isso o PC local nunca tenta 4-bit.
- **Opção `none` no config local**: o mesmo código corre em fp32 no dry-run, e o
  treino sem quantização é possível numa GPU com memória suficiente.

## Alternativas

- **LoRA em fp16/bf16 (sem quantização)**: mais simples e um pouco mais preciso, mas
  exige mais memória. Para 0,5B cabe na T4; para 1,5B pode ser apertado com batch maior.
- **Quantização 8-bit (LLM.int8)**: mais precisão do que 4-bit, mais memória e mais lenta.
- **GPTQ/AWQ**: quantização pós-treino, para servir. Não se aplica ao treino com adapters.

## Como falha

- **Sem CUDA ou sem bitsandbytes**: falha explícita antes de carregar o modelo (coberto).
- **Perda de qualidade por quantização**: pode reduzir a macro-F1 face a fp16. Só se
  mede comparando as duas configurações no dev (`TODO: medir` no notebook).
- **bf16 numa T4**: a T4 não suporta bf16 nativo. O código detecta com
  `torch.cuda.is_bf16_supported()` e usa fp16.
- **Incompatibilidade de versões**: bitsandbytes e torch têm de ser compatíveis.
  Por isso o notebook fixa `bitsandbytes==0.50.2`.

## Perguntas de entrevista

**1. O que é QLoRA em uma frase?**
Treinar LoRA sobre um modelo base quantizado a 4 bits (NF4) e congelado, com os
adapters em precisão normal.

**2. Porque é que o NF4 é melhor do que INT4 uniforme?**
Porque os níveis seguem a distribuição normal típica dos pesos, por isso o erro de
quantização médio é menor para o mesmo número de bits.

**3. Quanta memória poupa?**
Os pesos passam de ~2 bytes para ~0,5 byte por parâmetro. O total poupado depende das
activações e do batch; o valor real medido é o `max_vram_mib` da run, não uma estimativa.

**4. A quantização afecta o gradiente?**
Os gradientes só são calculados para os adapters. A quantização afecta a computação
da passagem para a frente, por isso afecta indirectamente a loss.

**5. Porque é que não quantizaste o modelo para servir?**
Para servir, a quantização pós-treino (GPTQ/AWQ) ou fp16 são as opções normais. Esse
passo fica fora de âmbito; a API carrega o modelo com PEFT (ver `09`).
