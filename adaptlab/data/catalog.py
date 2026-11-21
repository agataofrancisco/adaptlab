"""Catálogo de cenários e templates (dados sintéticos da Mukua Telecom, fictícia).

Cada cenário agrupa templates com o mesmo intent e a mesma intenção
comunicativa. O split é feito por cenário (ver ``splits.py``), por isso
templates do mesmo cenário nunca passam de treino para teste.

Marcadores:
- ``{numero}``, ``{valor}``, ``{localidade}``, ``{data}``: entidades preenchidas
  com valores sintéticos; se o template as usar, o label fica com esse valor.
- ``[[...]]``: segmento opcional. Pode ser omitido (gera ``incompleto`` e,
  se a entidade desaparecer, ``entidade_ausente``).
- ``mix=True``: o cenário mistura português e inglês de forma deliberada.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    intent: str
    priority: str
    requires_human: bool
    templates: tuple[str, ...]
    mix: bool = False


SCENARIOS: tuple[Scenario, ...] = (
    # --- recarga -------------------------------------------------------------
    Scenario(
        "recarga_saldo_nao_entrou",
        "recarga",
        "baixa",
        False,
        (
            "boa, fiz uma recarga de {valor} e o saldo não mudou",
            "comprei {valor} de crédito hoje mas nada entrou [[no {numero}]]",
        ),
    ),
    Scenario(
        "recarga_erro_sistema",
        "recarga",
        "baixa",
        False,
        (
            "a recarga de {valor} ta a dar erro no {numero} [[desde {data}]]",
            "tentei recarregar {valor} e aparece erro, o número é {numero}",
        ),
    ),
    Scenario(
        "recarga_ingles_misto",
        "recarga",
        "baixa",
        False,
        (
            "I tried to top up {valor} but it failed, meu número é {numero}",
            "can I recharge {valor} [[pelo telemóvel]]? sou de {localidade}",
        ),
        mix=True,
    ),
    Scenario(
        "recarga_como_fazer",
        "recarga",
        "baixa",
        False,
        (
            "como faço para recarregar {valor} pelo telemóvel?",
            "quero recarregar no número {numero} [[mas não sei como]], me ajudam",
        ),
    ),
    # --- avaria_rede ---------------------------------------------------------
    Scenario(
        "avaria_sinal_fraco_bairro",
        "avaria_rede",
        "alta",
        False,
        (
            "boa tarde mano, a net ta fraca no {localidade} desde {data}",
            "sem sinal nenhum no {localidade} desde {data}, ninguém consegue fazer chamadas",
        ),
    ),
    Scenario(
        "avaria_dados_moveis",
        "avaria_rede",
        "alta",
        False,
        (
            "não tenho dados móveis no {numero} mas as chamadas funcionam [[, estou em {localidade}]]",
            "my data is down since {data}, o número é {numero}",
        ),
        mix=True,
    ),
    Scenario(
        "avaria_escalada_tecnica",
        "avaria_rede",
        "alta",
        True,
        (
            "estou sem rede há {data} no {localidade} e já liguei três vezes, quero o responsável técnico",
            "a rede caiu no {localidade} há uma semana e ninguém resolve, vou apresentar queixa",
        ),
    ),
    Scenario(
        "avaria_sem_detalhes",
        "avaria_rede",
        "media",
        False,
        (
            "não funciona nada aqui, resolvam isso por favor",
            "isto ta parado outra vez, já é a terceira vez este mês",
        ),
    ),
    # --- faturacao -----------------------------------------------------------
    Scenario(
        "fatura_valor_a_mais",
        "faturacao",
        "media",
        False,
        (
            "a minha fatura veio com {valor} a mais, não consumi isso",
            "the bill came with {valor} extra charge, podem verificar?",
        ),
        mix=True,
    ),
    Scenario(
        "fatura_debito_duplicado",
        "faturacao",
        "alta",
        True,
        (
            "fui debitado duas vezes ({valor}) no mesmo dia {data}, quero o reembolso",
            "[[no número {numero}, ]]cobraram-me {valor} duas vezes e ninguém responde",
        ),
    ),
    Scenario(
        "fatura_reenvio",
        "faturacao",
        "baixa",
        False,
        (
            "não recebi a fatura deste mês no meu número {numero}, podem reenviar?",
            "quero saber o valor da fatura de {data}, sou de {localidade}",
        ),
    ),
    Scenario(
        "fatura_duvida_sem_dados",
        "faturacao",
        "baixa",
        False,
        (
            "tenho uma dúvida sobre a fatura, não percebo o valor cobrado",
            "não entendo esta cobrança na conta, me expliquem",
        ),
    ),
    # --- portabilidade -------------------------------------------------------
    Scenario(
        "portabilidade_prazo",
        "portabilidade",
        "baixa",
        False,
        (
            "quero mudar o meu número {numero} para outra operadora, quanto tempo demora?",
            "how long does portabilidade take? tenho o número {numero}",
        ),
        mix=True,
    ),
    Scenario(
        "portabilidade_recusada",
        "portabilidade",
        "media",
        False,
        (
            "a portabilidade do {numero} foi recusada, não sei porquê",
            "pedi a portabilidade do {numero} ontem e já apareceu como recusada",
        ),
    ),
    Scenario(
        "portabilidade_urgente_escalada",
        "portabilidade",
        "alta",
        True,
        (
            "preciso de portar o {numero} hoje, perdi o acesso à conta e ninguém me ajuda",
            "o {numero} tem de ser portado até amanhã por causa do trabalho, é urgente",
        ),
    ),
    Scenario(
        "portabilidade_sem_numero",
        "portabilidade",
        "baixa",
        False,
        (
            "quero fazer portabilidade, como faço?",
            "queria saber os passos para mudar de operadora",
        ),
    ),
    # --- cancelamento --------------------------------------------------------
    Scenario(
        "cancelamento_pacote",
        "cancelamento",
        "media",
        False,
        (
            "quero cancelar o meu pacote, o número é {numero}",
            "please cancel my plan, número {numero}",
        ),
        mix=True,
    ),
    Scenario(
        "cancelamento_retencao_supervisor",
        "cancelamento",
        "alta",
        True,
        (
            "quero cancelar tudo, já estou farto do serviço em {localidade}, passem-me um supervisor",
            "cancelem a linha {numero} agora, já não aguento mais esta rede",
        ),
    ),
    Scenario(
        "cancelamento_custo_fidelizacao",
        "cancelamento",
        "media",
        False,
        (
            "quanto me custa cancelar agora? ainda tenho {valor} de fidelização",
            "se eu cancelar hoje há alguma taxa de {valor}?",
        ),
    ),
    Scenario(
        "cancelamento_data_futura",
        "cancelamento",
        "media",
        False,
        (
            "cancelem o serviço a partir de {data}, por favor",
            "quero que a linha {numero} deixe de estar activa no dia {data}",
        ),
    ),
    # --- pacote_dados --------------------------------------------------------
    Scenario(
        "pacote_compra_falhou",
        "pacote_dados",
        "baixa",
        False,
        (
            "não consegui comprar o pacote de {valor}, aparece erro",
            "a compra do pacote de dados de {valor} não concluiu [[no {numero}]]",
        ),
    ),
    Scenario(
        "pacote_preco_duvida",
        "pacote_dados",
        "baixa",
        False,
        (
            "quanto custa o pacote de dados de {valor} e quantos gigas tem?",
            "o pacote de {valor} inclui chamadas? quero saber antes de comprar",
        ),
    ),
    Scenario(
        "pacote_consumo_acabou",
        "pacote_dados",
        "media",
        False,
        (
            "o meu pacote acabou em {data} e ainda tenho consumo, porque?",
            "a internet acabou antes da validade, o pacote tinha começado a {data}",
        ),
    ),
    Scenario(
        "pacote_ativar_ingles",
        "pacote_dados",
        "baixa",
        False,
        (
            "activate the data bundle {valor}, por favor, número {numero}",
            "I bought the bundle but no data came in, número {numero}",
        ),
        mix=True,
    ),
    # --- outro ---------------------------------------------------------------
    Scenario(
        "outro_fora_ambito",
        "outro",
        "baixa",
        False,
        (
            "vocês também fazem internet em casa? quero um orçamento",
            "têm vagas de emprego na vossa loja de {localidade}?",
        ),
    ),
    Scenario(
        "outro_ameaca_reclamacao",
        "outro",
        "alta",
        True,
        (
            "isto é uma vergonha, vou denunciar à entidade reguladora e às redes sociais",
            "se não resolverem hoje vou levar isto à imprensa, vergonha total",
        ),
    ),
    Scenario(
        "outro_dados_pessoais",
        "outro",
        "alta",
        True,
        (
            "quero apagar os meus dados pessoais e o histórico do {numero}",
            "queria saber quem tem acesso aos meus dados, pedido de privacidade",
        ),
    ),
    Scenario(
        "outro_saudacao_vaga",
        "outro",
        "baixa",
        False,
        (
            "bom dia, preciso de ajuda",
            "boa, queria falar com alguém sobre uma coisa",
        ),
    ),
)
