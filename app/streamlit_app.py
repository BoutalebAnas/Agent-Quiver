"""AgentQuiver front-end (Streamlit).

The app ONLY talks to the backend over HTTP (/health, /users, /recommend,
/log_interaction, /interactions). It never imports the model.

Run (API already running on port 8000):
    uv run streamlit run app/streamlit_app.py
"""
import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.getenv("AQ_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="AgentQuiver", layout="wide")
st.title("AgentQuiver")
st.caption("The next open-source AI packages for each developer, grouped by their interests.")


def call(method, path, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=30, **kw)
        return r
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {API}. Start it with: "
                 "`AQ_RECOMMENDER=model uv run uvicorn src.api.main:app --reload`")
        st.stop()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    health = call("GET", "/health").json()
    if health["is_mock"]:
        st.warning(f"API running the MOCK ({health['model_version']})")
    else:
        st.success(f"Model: {health['model_version']}")
    st.caption(f"API: {API}")

    known = call("GET", "/users", params={"limit": 50}).json()
    choice = st.selectbox("Pick a known user", known)
    typed = st.text_input("…or type a user id", "")
    username = typed.strip() or choice
    k = st.slider("Number of recommendations", 5, 30, 10)
    go = st.button("Recommend", type="primary", use_container_width=True)

# ---------------------------------------------------------------- recommend
if go:
    t0 = time.perf_counter()
    r = call("GET", "/recommend", params={"username": username, "k": k})
    st.session_state["latency"] = time.perf_counter() - t0
    if r.status_code == 200:
        st.session_state["result"] = r.json()
    else:
        st.session_state.pop("result", None)
        st.error(f"{r.status_code}: {r.json().get('detail')}")

res = st.session_state.get("result")
if res:
    user = res["username"]
    st.subheader(f"Recommendations for {user}")
    st.caption(f"{res['model_version']} · is_mock={res['is_mock']} · "
               f"response in {st.session_state['latency']:.3f} s · score = {res['score_type']}")

    for it in res["interests"]:
        with st.container(border=True):
            st.markdown(f"#### {it['label']}")
            st.caption(f"{it['explanation']} (from your history: {', '.join(it['evidence'])})")
            for rec in sorted(it["recommendations"], key=lambda x: x["rank"]):
                item = rec["repo"]
                c1, c2, c3 = st.columns([6, 1, 1])
                name = f"[{item['repo_id']}]({item['url']})" if item.get("url") else item["repo_id"]
                c1.markdown(f"**#{rec['rank']}** {name}  \n"
                            f"<small>{item.get('description') or 'No description'} · "
                            f"score {rec['score']:.2f}</small>", unsafe_allow_html=True)
                for col, action, label in ((c2, "useful", "Useful"), (c3, "not_useful", "Not useful")):
                    if col.button(label, key=f"{action}-{item['repo_id']}"):
                        call("POST", "/log_interaction", json={
                            "user_id": user, "repo_id": item["repo_id"],
                            "interest_id": it["interest_id"], "rank": rec["rank"], "action": action})
                        st.toast(f"Saved: {label} for {item['repo_id']}")

    # live proof that feedback reaches the database
    st.divider()
    st.markdown("#### Feedback recorded in the database")
    rows = call("GET", "/interactions", params={"user_id": user, "limit": 10}).json()
    if rows:
        st.dataframe(pd.DataFrame(rows)[["interaction_id", "repo_id", "action", "rank",
                                         "interest_id", "timestamp"]], hide_index=True)
    else:
        st.caption("No feedback yet. Click Useful / Not useful above.")
else:
    st.info("Pick a user in the sidebar and click Recommend.")
