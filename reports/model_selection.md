# Selecção do modelo base (tarefa 3.3)

Data da recolha: 2026-11-21. Fonte: API pública do Hugging Face (`/api/models/<id>`) e
`tokenizer_config.json` (quando o repositório não é restrito). Nada aqui foi inventado: onde
falta uma medição, está `TODO: medir`.

## Candidatos

| Modelo | Parâmetros (API) | Licença (tag) | Acesso | Chat template | Peso bf16 (estimado) | Peso NF4 (estimado) | Zero-shot macro-F1 dev |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | 494 032 768 | apache-2.0 | aberto | sim (verificado) | ~0,99 GB | ~0,25 GB | 0,1524* |
| Qwen/Qwen2.5-1.5B-Instruct | 1 543 714 304 | apache-2.0 | aberto | sim (verificado) | ~3,1 GB | ~0,8 GB | TODO: medir (download incompleto) |
| google/gemma-3-1b-it | 999 885 952 | gemma (termos próprios) | restrito (manual) | TODO: verificar (repo restrito) | ~2,0 GB | ~0,5 GB | fora de alcance sem aceitação dos termos |
| meta-llama/Llama-3.2-1B-Instruct | 1 235 814 400 | llama3.2 (licença própria) | restrito (manual) | TODO: verificar (repo restrito) | ~2,5 GB | ~0,6 GB | fora de alcance sem aceitação dos termos |
| HuggingFaceTB/SmolLM2-360M-Instruct | 361 821 120 | apache-2.0 | aberto | sim (verificado) | ~0,7 GB | ~0,2 GB | excluído: abaixo de 0,5B |

Notas:
- "Peso bf16" = parâmetros x 2 bytes. "Peso NF4" = parâmetros x 0,5 byte, uma estimativa
  grosseira dos pesos apenas (sem activações, sem o optimizador, sem embeddings em fp16).
  A VRAM real de treino só se mede numa run (`manifest` de `runs/train-*`, campo `max_vram_mib`).
- SmolLM2-1.7B-Instruct fica acima de 1,5B e está fora de âmbito (proposal).

## Decisão (provisória)

**Qwen/Qwen2.5-0.5B-Instruct**, revisão `7ae557604adf67be50417f59c2c2f167def9a775`.

Razões verificadas: licença Apache-2.0 sem restrição de acesso, chat template presente, o
menor peso entre os candidatos aceites, e cabe confortavelmente numa T4 com QLoRA.

Condição para tornar a decisão final: medir a macro-F1 zero-shot no dev dos dois
candidatos Qwen (0,5B e 1,5B), com o mesmo `SYSTEM_PROMPT` e o mesmo parsing. Essa medição
precisa de descarregar ~1 GB e ~3 GB de pesos e de correr geração em CPU ou no notebook.
Uma amostra estratificada preliminar de 20 exemplos do dev foi concluída para o 0,5B; o
download do shard do 1,5B não avançou e não produziu métricas. A decisão fica por isso
provisória e a tarefa 3.3 permanece aberta (ver `docs/learn/07-hiperparametros-e-dev-set.md`).

## Medição CPU preliminar — 2026-11-24

`0,1524*` é macro-F1 de intent numa amostra estratificada de 20 exemplos do dev (`seed=42`),
com bfloat16 no CPU e revisão `7ae557604adf67be50417f59c2c2f167def9a775`. A run completa, incluindo
intervalos de confiança, confusão e predições, está em
`runs/eval-zero_shot-dev-20261124T064440766696Z-b244de/metrics.json` e `predictions.jsonl`.
Com apenas 20 exemplos, o intervalo bootstrap é largo (0,0625–0,2568); este número é apenas
exploratório e não justifica uma escolha final.

A tentativa equivalente do Qwen 1.5B usou revisão
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, mas o download ficou parado num shard incompleto
depois de 0 bytes durante vários minutos. A execução foi interrompida sem gerar resultados;
nenhum valor de desempenho é inferido para esse candidato.

## Licenças a verificar antes de publicar

- Qwen2.5 (Apache-2.0): redistribuição de derivados permitida com aviso de licença. Confirmar
  na model card no momento da publicação.
- Gemma e Llama: não escolhidos nesta fase. Se entrarem no futuro, exigem aceitação manual
  dos termos e as respectivas condições de redistribuição.
