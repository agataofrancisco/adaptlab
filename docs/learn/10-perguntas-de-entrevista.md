# 10 · Perguntas de entrevista

Respostas curtas, ligadas ao código e aos números medidos neste projecto. Onde algo não foi medido,
a resposta diz isso.

## Fine-tuning e dados

**1. Quando não deves fazer fine-tuning?**
Quando o conhecimento muda ou quando um prompt bem feito com um modelo maior resolve. Fine-tuning ensina
formato e padrões estáveis (`docs/learn/01`).

**2. Porque comparas com few-shot e não só com zero-shot?**
Few-shot é a alternativa realista sem treino. Só assim se mede o valor do treino em si.

**3. O que é uma fuga de dados aqui e como a detectas?**
Informação do teste no treino, directa ou por template. Splits por cenário, deduplicação e Jaccard de
5-gramas entre splits; o pipeline falha se houver fuga. Medido: zero pares acima de 0,6.

**4. Porque é que o teste tem hash?**
Para detectar alterações depois de se ver resultados. A avaliação final recusa-se a correr se o hash mudar.

**5. Como sabes que o detector de fugas funciona?**
Teste artificial: injecta uma cópia de um exemplo no teste e espera que seja apanhada
(`test_artificial_leak_is_detected`).

**6. Porque é que o ruído não altera as entidades?**
A validação "entidade literal na mensagem" precisa do valor exacto. O ruído fica no texto do template.

## Tokenização e formato

**7. O que é um chat template?**
O formato de papéis e fins de turno com que o modelo foi treinado para instrução. Outro formato degrada a
qualidade sem aviso.

**8. O que é BPE?**
Byte-Pair Encoding: funde os pares mais frequentes até um vocabulário fixo. Palavras comuns viram um
token; raras viram vários.

**9. Porque o `<|im_end|>` deve estar na completion?**
Sem ele o modelo não aprende a parar.

**10. Porque fixas a revisão do tokenizer?**
Um repositório pode mudar o tokenizer. A revisão (SHA) garante os mesmos ids no treino e no serviço.

## Loss e treino

**11. O que é `ignore_index=-100`?**
Um valor que a cross-entropy ignora. Não é 0 porque 0 é um id de token válido.

**12. Como sabes que a loss só vê a resposta?**
O dry-run usa o collator do TRL e decodifica os tokens com label: 191 tokens mascarados e 51 com label,
e o texto com label é o JSON mais o fim de turno.

**13. A loss parece maior numa completion-only. Está mal?**
Não necessariamente: mede menos tokens e cada um é mais difícil. Compara-se no dev, não entre formatos
de loss.

**14. Quanto vale a loss inicial de um modelo aleatório?**
Aproximadamente ln(V). Com V=151665 dá cerca de 11,93; o dry-run mediu 11,96 no primeiro passo.

## LoRA e quantização

**15. O que é LoRA?**
Congelar os pesos e treinar `B·A` de rank `r`, escalado por `alpha/r`. `B` começa a zero para que o modelo
arranque igual ao base (`docs/learn/05`).

**16. Porque dividir alpha por r?**
Para a magnitude da actualização não depender do rank.

**17. O que é NF4?**
Quantização de 4 bits com níveis escolhidos para uma distribuição normal, adequada aos pesos de uma rede.

**18. O que é QLoRA?**
LoRA sobre um base em NF4 congelado, com adapters em precisão normal (`docs/learn/06`).

**19. Porque o treino QLoRA falha sem CUDA?**
Porque bitsandbytes só tem suporte CUDA. Falhar com mensagem clara é melhor do que um fallback silencioso
para outra configuração.

## Avaliação

**20. Porque comparas com regras?**
Para ter um piso barato. A baseline de regras já mede macro-F1 0,8351 no dev; o LoRA precisa de ganhar
claramente a isso.

**21. Macro-F1 ou accuracy?**
Macro-F1: cada classe pesa o mesmo, por isso uma classe grande como `outro` não esconde as pequenas.

**22. Como tratas saídas inválidas na métrica?**
Como erro em todos os campos. Não saem do denominador, para não premiar modelos que falham.

**23. Porque o recall de alto risco e não a precisão?**
Deixar passar uma escalada custa mais do que uma revisão extra.

**24. Como sabes se uma diferença de 0,01 é real?**
Com o IC bootstrap sobre os exemplos avaliados e, quando houver, a variância entre seeds. Com 180
exemplos de dev, 0,01 fica dentro do intervalo.

**25. O que acontece se o LoRA não ganhar?**
O critério de paragem bloqueia a exportação para publicação e o relatório regista o resultado negativo.

**26. O que é o `TODO: medir` e porque existe?**
Marca o que ainda não foi medido. Um número sem run de origem não entra no relatório.

## Serviço e publicação

**27. Como serves um adapter?**
Carregas o base e aplicas o adapter com PEFT. Podes fundir (`merge_and_unload`) para latência mínima, mas
perdes a comparação lado a lado.

**28. Porque a API devolve 200 numa saída inválida?**
É uma resposta estruturada (`error: invalid_model_output`, `requires_human: true`), não uma falha de
servidor.

**29. Como evitas que um demo seja confundido com inferência?**
Marcação `precomputed_demo` em cada resposta, `fixture_source` com a run de origem, e colunas rotuladas
como fixture na UI.

**30. Porque o predictor é injectado na app?**
Para testar a API sem modelo nem rede, com um gerador falso.

**31. O que pode ser publicado e o que não?**
Só o adapter (e a configuração PEFT), com hashes. Nunca os pesos do base, a menos que a licença o permita
e seja pedido. A publicação exige licenças compatíveis, critério aprovado e confirmação manual.

**32. Porque Apache-2.0 e MIT e não outras licenças?**
Permitem redistribuir derivados com aviso. Llama e Gemma têm termos próprios que o portão recusa até
haver verificação explícita.

## Engenharia

**33. Porque o dataset card é gerado?**
Para que os números sejam sempre os da última execução. Uma edição manual podia ficar desfasada.

**34. Porque o `.gitignore` usa negações para `runs/`?**
Para guardar metadados, métricas e previsões (pequenos e necessários para rastrear os números), mas nunca
adapters ou pesos.

**35. O que faz o `.gitattributes` com `text=auto`?**
Normaliza fins de linha no repositório para LF, evitando diffs de CRLF entre Windows e Linux.

**36. Porque é que os testes correm sem modelo?**
A maioria usa geradores falsos e dados pequenos. Só o dry-run usa o tokenizer real, e não descarrega
pesos.

**37. Que limitações tem o dataset?**
Sintético e escrito pelo autor, com poucos escalonamentos no teste (11). O IC do recall de alto risco é
largo. Declarado no card.

**38. O que falta para uma resposta honesta sobre o LoRA?**
Correr o notebook em GPU, medir o zero-shot e o few-shot, e a revisão PT-AO. Tudo registado como pendente.
