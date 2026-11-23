# Decisões

Formato: decisão, razão, alternativa rejeitada, onde está.

**D1. Modelo base provisório: Qwen2.5-0.5B-Instruct.** Licença Apache-2.0 sem gating, chat template
verificado, menor peso entre os candidatos aceites. Final só depois de medir o zero-shot no dev
(`reports/model_selection.md`). Rejeitados: Gemma-3-1B e Llama-3.2-1B (repositórios restritos, termos
próprios); SmolLM2-1.7B (acima de 1,5B).

**D2. Dados sintéticos com splits por cenário.** Sem dados reais, e um split aleatório deixaria
templates em treino e teste (`docs/learn/02`).

**D3. Ruído só no texto do template.** Os valores de entidade são exactos, para que a validação
"entidade literal na mensagem" seja determinística.

**D4. Deduplicação exacta e Jaccard de 5-gramas.** Limiar 0,85 para dentro do dataset; 0,6 para fugas
entre splits (que falha o pipeline).

**D5. Dry-run com tokenizer real e modelo tiny construído localmente.** Evita descarregar ~1 GB de pesos
e exercita o caminho real do TRL (`SFTTrainer`, collator, máscara).

**D6. Completion-only loss pelo TRL, verificada na prática.** Não reimplementamos a máscara; o dry-run
decodifica os tokens com label (`docs/learn/04`).

**D7. QLoRA falha explicitamente sem CUDA.** Sem fallback silencioso para fp32 com 4-bit ou para um
modelo diferente.

**D8. Parsing único (`parse_label`) para todas as abordagens.** Comparação justa.

**D9. Baseline de regras como piso medido.** Já tem resultado real no dev (macro-F1 0,8351). Limitação:
usa as mesmas localidades do catálogo.

**D10. Critério de paragem com margem de 0,02 sobre a melhor baseline, só no dev.** O estado sem
medição é `pendente`, nunca `aprovado` por omissão.

**D11. Métricas de alto risco como recall.** Escalar mal é o erro mais caro (`docs/learn/08`).

**D12. Demo com fixtures de uma run medida e rotuladas.** As fixtures são previsões da baseline de
regras, não do modelo; a API e a UI dizem isso.

**D13. Saída inválida devolvida como 200 com `error` e `requires_human: true`.** Dado estruturado,
não falha de servidor.

**D14. Exportação só do adapter, sem pesos do base.** E publicação nunca automática: o portão só diz se
estaria permitida. Licenças permitidas para redistribuir derivados: Apache-2.0 e MIT.

**D15. Runs: metadados e previsões no git, adapters e pesos fora.** `runs/*` com negações explícitas
(`.gitignore`).

**D16. Torch CPU a partir do índice oficial.** Build pequena para o PC sem GPU; CUDA só no notebook.

**D17. Dataset card gerado a partir do manifesto.** Os números são sempre os medidos na run; o estado da
revisão PT-AO fica "PENDENTE".

**D18. Licença do dataset: CC-BY-4.0 (proposta).** Dados criados para o projecto, sem dados pessoais.
Confirmar antes de publicar.

**D19. Sem `matplotlib`.** Curvas em tabela, gráfico PNG marcado `TODO`. Evita uma dependência que não
é necessária para os critérios.

**D20. Tracking em JSON, não MLflow.** Opcional na spec (D8 do design); menos dependências.
