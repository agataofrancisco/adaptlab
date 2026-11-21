# 03 · Tokenização e chat templates

## O problema

Um modelo de linguagem não vê texto: vê **tokens**, números inteiros que indexam uma
tabela. Para ensinar o modelo a responder com JSON, temos de dizer-lhe onde começa a
mensagem do utilizador, onde começa a resposta e onde acaba. Um erro de formato aqui
estraga o treino inteiro, sem sinal óbvio na loss.

## O que entra e o que sai

Entra: um exemplo do dataset (`message` + `label`), convertido para mensagens
(`adaptlab/training/format.py::to_prompt_completion`).

Sai (medido pelo dry-run, `runs/dry_run-20261121T063259Z/dry_run_report.json`):

```
tokenizer_vocab_size: 151665
tokens_per_example: min 242, max 277, mean 254.6   (8 exemplos, system prompt incluído)
max_length: 512, truncated_examples: 0
```

O texto que o modelo vê no fim do prompt e o texto com label (medido com o mesmo
collator que o treino usa):

```
prompt_tail_text: "as e o saldo nao mudou<|im_end|>\n<|im_start|>assistant\n"
labeled_text:     "{\"intent\": \"recarga\", \"priority\": \"baixa\", \"entities\": {\"numero\": null,
                   \"valor\": \"1 mil kwanzas\", \"localidade\": null, \"data\": null},
                   \"requires_human\": false}<|im_end|>\n"
```

## Como funciona por dentro

1. **Mensagens estruturadas**: `build_prompt` devolve `[system, user]`. O `SYSTEM_PROMPT`
   (`format.py`, linha 18) é o mesmo que as baselines e o serviço usam.
2. **Chat template**: cada modelo define um template Jinja no `tokenizer_config.json`
   (chave `chat_template`). Para o Qwen2.5 ele insere `<|im_start|>role ... <|im_end|>`.
   O TRL aplica-o ao prompt e à completion.
3. **Tokenização**: o tokenizer BPE do Qwen transforma o texto em ids. Os tokens especiais
   (`<|im_start|>`, `<|im_end|>`) têm ids fixos.
4. **Fim de turno**: o `<|im_end|>` no fim da completion é o que ensina o modelo a parar.
   No `labeled_text` acima ele aparece, por isso o modelo aprende a terminar.

Verificação com código real: o dry-run aplica `tokenizer.apply_chat_template(..., tokenize=False)`
e conta tokens com `tokenizer(text, add_special_tokens=False)`
(`adaptlab/training/dryrun.py`, função `run_dry_run`).

## Porquê esta escolha

- **Chat template do próprio modelo**, e não um formato inventado: um modelo instruct foi
  treinado com aquele formato. Usar outro degrada a qualidade sem aviso.
- **System prompt com o schema**: o modelo vê as chaves e os valores permitidos. O
  parsing em `schema.py` valida a saída de qualquer forma.
- **Um único `SYSTEM_PROMPT`** para treino, zero-shot, few-shot e serviço: a comparação
  entre abordagens não pode depender de diferenças de prompt.

## Alternativas

- Formato de texto simples (`Mensagem: ... Resposta: ...`): funciona, mas ignora o
  template do modelo e perde o token de fim de turno.
- `formatting_func` do TRL: é incompatível com `completion_only_loss` (ver
  `sft_trainer.py`, linha ~1071), por isso não o usamos.

## Como falha

- **Template ausente**: se o tokenizer não tiver `chat_template`, `apply_chat_template`
  falha. Coberto pelo dry-run.
- **Truncagem silenciosa**: uma completion truncada ensina JSON incompleto. O dry-run
  conta `truncated_examples` e marca `warn_truncation` (`dryrun.py`).
- **Espaços e quebras de linha** diferentes entre treino e serviço mudam os tokens. Mitigação:
  os dois usam o mesmo tokenizer com a mesma `revision` fixa (`configs/train.yaml`).
- Testes: `tests/test_training.py::test_prompt_completion_has_system_user_and_assistant_json`,
  `test_dry_run_masks_prompt_and_trains_two_steps`.

## Perguntas de entrevista

**1. O que é um chat template e porque é que importa?**
É o formato que o modelo aprendeu durante o treino de instrução: onde está cada papel
e onde termina cada turno. Usar outro formato faz o modelo responder pior, mesmo com o
mesmo conteúdo.

**2. O que é BPE?**
Byte-Pair Encoding: começa com caracteres (ou bytes) e funde os pares mais frequentes
até atingir um vocabulário fixo. Palavras comuns viram um token; palavras raras viram
vários. Por isso "saldo" pode ser um token e "kilamba" vários.

**3. Porque é que o tamanho em tokens importa para o treino?**
Custo de memória e de tempo crescem com o comprimento da sequência. Exemplos mais longos
obrigam a reduzir o batch ou a aumentar `max_length`.

**4. Porque é que o `<|im_end|>` está na completion?**
Sem ele o modelo nunca aprende a parar e continuaria a gerar texto após o JSON.

**5. Porque é que fixas a revisão do tokenizer?**
Um repositório pode mudar o tokenizer ou o template depois do treino. A revisão fixa
(commit SHA) garante que treino e serviço usam exactamente os mesmos ids.
