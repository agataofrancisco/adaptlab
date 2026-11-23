"""UI mínima (Streamlit): saída do modelo base e do adaptado lado a lado, para a mesma mensagem.

Corre com a API a funcionar (``uvicorn api.main:app``) e, opcionalmente, ``ADAPTLAB_API_URL``.
No modo demo, as respostas vêm marcadas como ``precomputed_demo`` e mensagens fora do conjunto
mostram a explicação da API. Não há inferência escondida na UI.

    .venv\\Scripts\\streamlit run ui\\app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("ADAPTLAB_API_URL", "http://127.0.0.1:8000").rstrip("/")
FIELDS = ("intent", "priority", "requires_human")


def call(message: str, model: str) -> dict:
    try:
        response = httpx.post(f"{API_URL}/predict", params={"model": model}, json={"message": message}, timeout=120)
    except httpx.HTTPError as exc:
        return {"status": "erro_de_ligacao", "body": {"error": str(exc)}}
    try:
        body = response.json()
    except ValueError:
        body = {"error": response.text}
    return {"status": response.status_code, "body": body}


def diff_rows(base: dict, adapted: dict) -> list[dict]:
    base_label = (base.get("label") or {}) if isinstance(base, dict) else {}
    adapted_label = (adapted.get("label") or {}) if isinstance(adapted, dict) else {}
    rows = []
    for field in FIELDS:
        b, a = base_label.get(field), adapted_label.get(field)
        rows.append({"campo": field, "base": b, "adaptado": a, "diferente": b != a})
    b_ent = base_label.get("entities") or {}
    a_ent = adapted_label.get("entities") or {}
    for key in ("numero", "valor", "localidade", "data"):
        b, a = b_ent.get(key), a_ent.get(key)
        rows.append({"campo": f"entities.{key}", "base": b, "adaptado": a, "diferente": b != a})
    return rows


def column(title: str, result: dict) -> None:
    body = result["body"]
    if body.get("mode") == "precomputed_demo":
        # No modo demo a coluna NÃO é o modelo: é uma fixture pré-calculada (baseline de regras medida).
        title = f"{title} · fixture (não é o modelo)"
    st.subheader(title)
    st.caption(f"HTTP {result['status']} · modo: {body.get('mode', '—')}")
    if body.get("error") and not body.get("label"):
        st.warning(f"{body.get('error')}: {body.get('explanation') or body.get('detail') or ''}")
    if body.get("label"):
        st.json(body["label"])
    st.caption(f"latência (ms): {body.get('latency_ms', '—')} · requires_human: {body.get('requires_human')}")


st.set_page_config(page_title="AdaptLab", layout="wide")
st.title("AdaptLab: base vs adaptado")
st.write("Escreve uma mensagem de cliente (pt-PT / PT-AO) e compara as duas saídas.")

message = st.text_area("Mensagem do cliente", value="boa tarde mano, a net ta fraca no kilamba desde ontem", height=90)
if st.button("Classificar", type="primary") and message.strip():
    base = call(message, "base")
    adapted = call(message, "adapted")
    left, right = st.columns(2)
    with left:
        column("Base", base)
    with right:
        column("Adaptado (LoRA)", adapted)
    if isinstance(base["body"], dict) and isinstance(adapted["body"], dict):
        st.subheader("Diferenças")
        rows = diff_rows(base["body"], adapted["body"])
        st.dataframe(rows, use_container_width=True)
