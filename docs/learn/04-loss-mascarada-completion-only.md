# 04 · Loss mascarada (completion-only)

## O problema

A loss de língua mede o erro de prever cada token seguinte. Se a aplicarmos a todo o
exemplo, o modelo também aprende a **gerar a mensagem do cliente**. Isso gasta capacidade
do LoRA num problema que não queremos resolver (imitar clientes), e pode diluir o sinal
do JSON, que é o que importa.

## O que entra e o que sai

Entra: um exemplo prompt/completion com ~254 tokens em média.

Sai (dry-run real, `masking` no relatório):

```
total_tokens: 242
masked_prompt_tokens: 191     # labels = -100, a loss ignora-os
labeled_tokens: 51            # só o JSON da resposta e o <|im_end|>
contiguous_labels: true       # a parte com label é um bloco contíguo no fim
```

## Como funciona por dentro

- O PyTorch usa `-100` como "ignorar" em `CrossEntropyLoss` (`ignore_index`).
- No TRL 1.18.2, o collator constrói `labels` a partir de `input_ids` e depois põe
  `-100` onde `completion_mask == 0`
  (`.venv/Lib/site-packages/trl/trainer/sft_trainer.py`, linhas 502-503). Os tokens de
  padding ficam também a `-100` pela `attention_mask`, na linha anterior.
- A `completion_mask` é 0 no prompt e 1 na completion. É construída ao concatenar o
  prompt e a completion (`sft_trainer.py`, linhas 461-464).
- A opção `completion_only_loss` está em `SFTConfig` (`sft_config.py`, linha 261). Com
  `None` (omissão), o TRL liga-a para datasets prompt/completion
  (`sft_trainer.py`, linha 1019-1020). Nós passamos `true` explicitamente em `configs/train.yaml`.
- `inspect_masking` (`adaptlab/training/dryrun.py`, linha 39) mostra o que a loss vê, a
  partir do mesmo `trainer.data_collator` que o treino usa. Não reimplementamos a máscara.

## Porquê esta escolha

- Só a resposta é alvo. O modelo aprende o mapeamento mensagem → JSON, que é a tarefa.
- A verificação é feita no caminho real do treino (collator do TRL), não numa cópia.
  Se uma versão futura do TRL mudar a máscara, o dry-run falha.

## Alternativas

- **Loss em tudo** (`completion_only_loss: false`): mais simples, pior foco. Fica como
  experiência de controlo, não usada.
- **`assistant_only_loss`**: útil para conversas com vários turnos. Para um prompt e uma
  resposta, o prompt/completion é mais direto. Requer chat template com
  `{% generation %}`, que não temos.
- **Máscara manual**: funciona, mas duplica código que o TRL já mantém e testa.

## Como falha

- **Máscara vazia**: se a completion for truncada além de `max_length`, não sobra label.
  O dry-run falha com `fail_masking` quando `masked_prompt_tokens == 0` ou
  `contiguous_labels` é falso.
- **Fim de turno fora da máscara**: o modelo aprenderia a não parar. Verificado pelo
  `labeled_text` que termina em `<|im_end|>`.
- Teste: `tests/test_training.py::test_dry_run_masks_prompt_and_trains_two_steps`.

## Perguntas de entrevista

**1. O que é `ignore_index=-100` e porque não 0?**
0 é um id de token válido. Usar -100 deixa a loss ignorar as posições sem confundir com
um token real.

**2. Porque é que a loss de uma completion-only pode parecer maior?**
Porque mede menos tokens e cada um é mais difícil (o JSON tem valores específicos). Não
é sinal de pior treino por si só; compara-se no dev.

**3. Como sabes que a máscara está certa?**
Não confio no código de cabeça: o dry-run decodifica os tokens com label e mostra-os.
Vê-se que o texto com label é o JSON e o `<|im_end|>`.

**4. O que acontece se a completion for truncada?**
O modelo treina em JSON incompleto e aprende a produzi-lo. Por isso o dry-run conta
truncagens e o pipeline reporta `warn_truncation`.

**5. Qual é a diferença entre `completion_mask` e `attention_mask`?**
`attention_mask` diz que tokens são reais (não padding) e entram no cálculo da atenção.
`completion_mask` diz quais tokens entram na loss. Um token pode estar na atenção e
fora da loss (o prompt).
