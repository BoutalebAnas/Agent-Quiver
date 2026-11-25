"""AgentQuiver front-end (Streamlit).

The app ONLY talks to the backend over HTTP (/health, /users, /history, /recommend, /recommend_stack,
/better_tools, /log_interaction, /interactions). It never imports the model.

Run (API already running on port 8000):
    python -m streamlit run app/streamlit_app.py
"""
import html
import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.getenv("AQ_API", "http://127.0.0.1:8000")
DEMOS = {  # projects that make a good live demo (all in the dataset)
    "pypi:langchain-google-vertexai": "LangChain + Vertex AI",
    "gh:DhruvTambekar24": "GitHub dev · React + Gemini",
    "pypi:crewai": "CrewAI agents",
    "pypi:gradio": "Gradio",
    "pypi:prefect": "Prefect (better tools)",
}
SAMPLE_STACK = "langchain-core\ngoogle-cloud-storage\nhttpx>=0.27\nrequests"
ECO = {"pypi": ("PyPI", "eco-pypi"), "npm": ("npm", "eco-npm"), "tool": ("AI tool", "eco-tool")}

st.set_page_config(page_title="AgentQuiver", page_icon="🏹", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
:root { --navy:#0f1f38; --amber:#f0a92e; --amber-dark:#a86a00; --ink2:#4a5870; --line:#e2ddd0; }
html, body, [class*="css"], .stMarkdown, .stButton button { font-family: Inter, system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4, .aq-title { font-family: 'Space Grotesk', Inter, sans-serif !important; letter-spacing: -0.3px; }
.aq-hero { background: var(--navy); color: #fff; border-radius: 20px; padding: 26px 30px; display: flex;
  align-items: center; justify-content: space-between; gap: 20px; position: relative; overflow: hidden; margin-bottom: 6px; }
.aq-hero svg.deco { position: absolute; right: -40px; top: -60px; opacity: .35; }
.aq-brand { display: flex; align-items: center; gap: 14px; position: relative; }
.aq-brand .name { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 30px; letter-spacing: -0.6px; }
.aq-brand .tag { color: #c6d0de; font-size: 15px; margin-top: 2px; }
.aq-pill { position: relative; display: inline-flex; align-items: center; gap: 8px; background: rgba(255,255,255,.1);
  border: 1px solid rgba(255,255,255,.18); border-radius: 999px; padding: 7px 14px; font-size: 13px; white-space: nowrap; }
.aq-dot { width: 9px; height: 9px; border-radius: 50%; background: #3ccf8e; box-shadow: 0 0 0 3px rgba(60,207,142,.25); }
.aq-dot.mock { background: var(--amber); box-shadow: 0 0 0 3px rgba(240,169,46,.25); }
.aq-kpis { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin: 14px 0 4px; }
.aq-kpi { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 12px 16px; }
.aq-kpi b { font-family: 'Space Grotesk', sans-serif; font-size: 24px; display: block; letter-spacing: -0.5px; }
.aq-kpi span { color: var(--ink2); font-size: 12.5px; }
.aq-section { font-size: 11.5px; font-weight: 700; letter-spacing: 1.4px; color: var(--ink2); text-transform: uppercase; margin: 4px 0 8px; }
.aq-card-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.aq-card-head .lbl { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 19px; }
.aq-why { color: var(--ink2); font-size: 13.5px; margin: 2px 0 6px; }
.aq-ev { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; background: #fdf3df; color: #6b4a00;
  padding: 2px 7px; border-radius: 6px; margin-right: 4px; }
.aq-rec { display: flex; gap: 12px; align-items: flex-start; padding: 4px 0; }
.aq-rank { flex: 0 0 30px; height: 30px; border-radius: 9px; background: var(--navy); color: #fff; display: flex;
  align-items: center; justify-content: center; font-weight: 700; font-size: 13px; }
.aq-rank.top { background: var(--amber); color: var(--navy); }
.aq-pkg { font-size: 14px; font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-weight: 500; font-size: 14.5px; color: var(--navy); text-decoration: none; }
a.aq-pkg, a.aq-pkg:visited { text-decoration: none !important; color: var(--navy) !important; border-bottom: 1.5px solid #f0d49a; }
.aq-pkg:hover { text-decoration: underline; }
.aq-desc { color: var(--ink2); font-size: 13px; line-height: 1.35; margin-top: 2px; }
.aq-badge { white-space: nowrap; font-size: 10.5px; font-weight: 700; letter-spacing: .5px; padding: 2px 7px; border-radius: 999px; margin-left: 6px; vertical-align: 2px; }
.eco-pypi { background: #e6eefb; color: #2457b3; } .eco-npm { background: #fde8e8; color: #b32424; }
.eco-tool { background: #e3f5ee; color: #1c7357; } .eco-other { background: #eee; color: #555; }
.aq-bar { height: 4px; background: #ece8dc; border-radius: 4px; margin-top: 6px; width: 100%; max-width: 220px; }
.aq-bar i { display: block; height: 4px; background: var(--amber); border-radius: 4px; }
.aq-stack-row .aq-pkg { font-size: 12.5px; }
.aq-stack-row { display: flex; justify-content: space-between; gap: 8px; padding: 6px 0; border-bottom: 1px dashed var(--line); font-size: 13px; }
.aq-stack-row:last-child { border-bottom: 0; }
.aq-stack-row .d { color: var(--ink2); font-size: 12px; white-space: nowrap; }
.aq-stack-row.gone .aq-pkg { text-decoration: line-through; color: #8a93a3; }
.aq-switch { display: flex; align-items: center; gap: 10px; padding: 10px 12px; background: #fff; border: 1px solid var(--line);
  border-radius: 12px; margin-bottom: 8px; flex-wrap: wrap; }
.aq-switch .old { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #8a93a3; text-decoration: line-through; }
.aq-switch .new { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-weight: 700; color: var(--navy); background: #fdf3df; padding: 2px 8px; border-radius: 6px; }
.aq-switch .n { margin-left: auto; font-size: 12.5px; color: var(--ink2); }
.aq-chip { display: inline-block; font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; padding: 3px 9px; border-radius: 7px; margin: 0 6px 6px 0; }
.aq-chip.ok { background: #e3f5ee; color: #1c7357; } .aq-chip.ko { background: #f1efe9; color: #8a93a3; text-decoration: line-through; }
.aq-step { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 16px; height: 100%; }
.aq-step .n { font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: var(--amber-dark); font-size: 13px; letter-spacing: 1px; }
.aq-step h4 { margin: 4px 0 6px; font-size: 17px; }
.aq-step p { color: var(--ink2); font-size: 13.5px; margin: 0; line-height: 1.45; }
.aq-foot { color: #8a93a3; font-size: 12px; text-align: center; margin-top: 28px; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-radius: 16px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
@media (max-width: 800px) { .aq-kpis { grid-template-columns: repeat(2, minmax(0,1fr)); } .aq-hero { flex-direction: column; align-items: flex-start; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LOGO = ('<svg width="46" height="46" viewBox="0 0 46 46" aria-hidden="true"><rect x="1" y="1" width="44" height="44" rx="11" '
        'fill="#ffffff"/><path d="M13 33 L32 14" stroke="#0f1f38" stroke-width="3" stroke-linecap="round"/>'
        '<path d="M23 13 L33 13 L33 23" fill="none" stroke="#0f1f38" stroke-width="3" stroke-linecap="round" '
        'stroke-linejoin="round"/><path d="M13 33 L10 30 M13 33 L16 36" stroke="#f0a92e" stroke-width="2.4" '
        'stroke-linecap="round"/></svg>')
DECO = ('<svg class="deco" width="300" height="300" viewBox="0 0 300 300" aria-hidden="true"><g fill="none" stroke="#f0a92e" '
        'stroke-width="2" stroke-linecap="round"><path d="M40 270 L250 60"/><path d="M90 290 L290 90"/><path d="M210 60 L250 60 '
        'L250 100"/><path d="M250 90 L290 90 L290 130"/></g></svg>')


def esc(x) -> str:
    return html.escape(str(x or ""))


def call(method, path, **kw):
    try:
        return requests.request(method, f"{API}{path}", timeout=30, **kw)
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {API}. Start it first (see the README): "
                 "`AQ_RECOMMENDER=model python -m uvicorn src.api.main:app`")
        st.stop()


def detail(r) -> str:
    try:
        return r.json().get("detail", r.text)
    except ValueError:
        return r.text


def short(item_id: str) -> str:
    return item_id.split(":", 1)[-1]


def badge(item_id: str) -> str:
    label, cls = ECO.get(item_id.split(":", 1)[0], (item_id.split(":", 1)[0], "eco-other"))
    return f'<span class="aq-badge {cls}">{esc(label)}</span>'


# ---------------------------------------------------------------- hero
t0 = time.perf_counter()
health = call("GET", "/health").json()
ping = time.perf_counter() - t0
mock = health["is_mock"]
status = f'{"Mock data" if mock else "Live model"} · {esc(health["model_version"])}'
st.markdown(f"""
<div class="aq-hero">{DECO}
  <div class="aq-brand">{LOGO}<div><div class="name">AgentQuiver</div>
  <div class="tag">Your next package, before you search. Learned from what real projects actually adopted.</div></div></div>
  <div class="aq-pill"><span class="aq-dot{' mock' if mock else ''}"></span>{status}</div>
</div>
<div class="aq-kpis">
  <div class="aq-kpi"><b>44,703</b><span>real adoption decisions</span></div>
  <div class="aq-kpi"><b>3,840</b><span>projects and developers</span></div>
  <div class="aq-kpi"><b>7,561</b><span>packages and AI tools (PyPI, npm)</span></div>
  <div class="aq-kpi"><b>39</b><span>tool switches verified by hand</span></div>
</div>
""", unsafe_allow_html=True)
if mock:
    st.warning("The API is serving the MOCK recommender. Restart it with AQ_RECOMMENDER=model for real results.")

if "toast" in st.session_state:
    st.toast(st.session_state.pop("toast"))


# ---------------------------------------------------------------- shared renderers
def render_interests(interests, user=None, key=""):
    """One card per interest. With a user, each item gets Useful / Not useful buttons (saved by the API)."""
    if not interests:
        st.info("No recommendation left for this profile.")
        return
    top = max((r["score"] for it in interests for r in it["recommendations"]), default=1) or 1
    interests = sorted(interests, key=lambda it: min((r["rank"] for r in it["recommendations"]), default=99))
    for it in interests:
        with st.container(border=True):
            ev = "".join(f'<span class="aq-ev">{esc(short(e))}</span>' for e in it["evidence"])
            st.markdown(f'<div class="aq-card-head"><span class="lbl">{esc(it["label"].replace("pypi:", "").replace("npm:", "").replace("tool:", ""))}</span></div>'
                        f'<div class="aq-why">{esc(it["explanation"].split(" (")[0].replace("pypi:", "").replace("npm:", "").replace("tool:", ""))}</div>'
                        f'<div style="margin-bottom:6px">{ev}</div>', unsafe_allow_html=True)
            for rec in sorted(it["recommendations"], key=lambda x: x["rank"]):
                item = rec["repo"]
                rid = item["repo_id"]
                name = (f'<a class="aq-pkg" href="{esc(item["url"])}" target="_blank">{esc(short(rid))}</a>'
                        if item.get("url") else f'<span class="aq-pkg">{esc(short(rid))}</span>')
                width = max(6, min(100, int(100 * rec["score"] / top)))
                row = (f'<div class="aq-rec"><div class="aq-rank{" top" if rec["rank"] <= 3 else ""}">{rec["rank"]}</div>'
                       f'<div style="flex:1;min-width:0">{name}{badge(rid)}'
                       f'<div class="aq-desc">{esc(item.get("description") or "No description")}</div>'
                       f'<div class="aq-bar" title="score {rec["score"]:.2f}"><i style="width:{width}%"></i></div></div></div>')
                if user:
                    c1, c2, c3 = st.columns([9, 1, 1], vertical_alignment="center")
                    c1.markdown(row, unsafe_allow_html=True)
                    c2.button(":material/thumb_up:", key=f"u-{key}-{rid}", help="Useful", on_click=rate,
                              args=(user, rid, it["interest_id"], rec["rank"], "useful", "Useful"))
                    c3.button(":material/thumb_down:", key=f"n-{key}-{rid}", help="Not useful", on_click=rate,
                              args=(user, rid, it["interest_id"], rec["rank"], "not_useful", "Not useful"))
                else:
                    st.markdown(row, unsafe_allow_html=True)


def render_better_tools(tools):
    st.markdown('<div class="aq-section">Better tools · what other projects switched to</div>', unsafe_allow_html=True)
    if not tools:
        st.caption("Nothing to replace: no package here matches a switch that other projects made.")
        return
    for t in tools[:6]:
        st.markdown(f'<div class="aq-switch"><span class="old">{esc(short(t["replace"]))}</span><span>→</span>'
                    f'<span class="new">{esc(short(t["with"]))}</span><span class="n">{t["projects"]} projects switched</span></div>',
                    unsafe_allow_html=True)
        if t.get("note"):
            st.caption(t["note"])
    st.caption("Learned from PyPI release histories: projects that dropped one package and adopted the other "
               "in the same release. Each rule was checked by hand.")


# ---------------------------------------------------------------- explore a project
def fetch(user, n):
    t0 = time.perf_counter()
    r = call("GET", "/recommend", params={"username": user, "k": n})
    st.session_state["latency"] = time.perf_counter() - t0
    st.session_state["query"] = (user, n)
    if r.status_code == 200:
        st.session_state["result"] = r.json()
        st.session_state.pop("error", None)
    else:
        st.session_state.pop("result", None)
        st.session_state["error"] = f"{r.status_code}: {detail(r)}"


def rate(user, item_id, interest_id, rank, action, label):
    """Save the rating, then refresh: the item leaves the list and appears in the feedback table."""
    call("POST", "/log_interaction", json={"user_id": user, "repo_id": item_id,
                                           "interest_id": interest_id, "rank": rank, "action": action})
    st.session_state["toast"] = f"Saved: {label} for {short(item_id)}"
    fetch(*st.session_state["query"])


def pick_demo(user):
    st.session_state["typed"] = user
    fetch(user, st.session_state.get("k", 10))


tab1, tab2, tab3 = st.tabs(["🔎  Explore a project", "📋  Paste your stack", "⚙️  How it works"])

with tab1:
    st.markdown('<div class="aq-section">Try a real project</div>', unsafe_allow_html=True)
    demo_cols = st.columns(len(DEMOS))
    for col, (uid, label) in zip(demo_cols, DEMOS.items()):
        col.button(label, key=f"demo-{uid}", on_click=pick_demo, args=(uid,), width="stretch")

    users = call("GET", "/users", params={"limit": 200}).json()
    known = list(DEMOS) + [u for u in users if u not in DEMOS]
    c1, c2, c3, c4 = st.columns([3, 3, 2, 1.3], vertical_alignment="bottom")
    choice = c1.selectbox("Pick a user from the dataset", known)
    typed = c2.text_input("…or type a user id", key="typed", placeholder="pypi:prefect  or  gh:username")
    k = c3.slider("Recommendations", 5, 30, 10, key="k")
    if c4.button("Recommend", type="primary", width="stretch"):
        fetch(typed.strip() or choice, k)

    if st.session_state.get("error"):
        st.error(st.session_state["error"])
    res = st.session_state.get("result")
    if res:
        user = res["username"]
        n_items = sum(len(i["recommendations"]) for i in res["interests"])
        st.markdown(f"### Recommendations for `{user}`")
        st.caption(f"{n_items} packages in {len(res['interests'])} interests · answered in "
                   f"{st.session_state['latency']:.2f} s · score = {res['score_type']}")
        left, right = st.columns([1, 2.4], gap="large")
        with left:
            with st.container(border=True):
                hist = call("GET", "/history", params={"username": user})
                st.markdown('<div class="aq-section">Your stack · what the model reads</div>', unsafe_allow_html=True)
                if hist.status_code == 200:
                    h = hist.json()
                    rows = "".join(
                        f'<div class="aq-stack-row{"" if x["still_in_use"] else " gone"}"><span>'
                        f'<span class="aq-pkg">{esc(short(x["package"]))}</span>{badge(x["package"])}</span>'
                        f'<span class="d">{esc(x["date"][:7]) if x["dated"] else "first release"}</span></div>'
                        for x in reversed(h[-14:]))
                    st.markdown(rows, unsafe_allow_html=True)
                    st.caption(f"{len(h)} adoptions, most recent first"
                               + (f" (showing 14)" if len(h) > 14 else "") + ". Struck out: removed since.")
                else:
                    st.caption(detail(hist))
            with st.container(border=True):
                bt = call("GET", "/better_tools", params={"username": user})
                render_better_tools(bt.json() if bt.status_code == 200 else [])
        with right:
            render_interests(res["interests"], user=user, key="u")

        with st.expander("Feedback saved in the database (POST /log_interaction → SQLite)"):
            rows = call("GET", "/interactions", params={"user_id": user, "limit": 10}).json()
            if rows:
                st.dataframe(pd.DataFrame(rows)[["interaction_id", "repo_id", "action", "rank",
                                                 "interest_id", "timestamp"]], hide_index=True,
                             width="stretch")
            else:
                st.caption("No feedback yet. Rate a package with the thumbs: it is saved and leaves the list.")
    elif not st.session_state.get("error"):
        st.info("Pick a project above, or choose a user and click Recommend.")

# ---------------------------------------------------------------- paste your stack
with tab2:
    st.markdown('<div class="aq-section">Any project, even one we have never seen</div>', unsafe_allow_html=True)
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        text = st.text_area("Paste a requirements.txt, a package.json, or package names (one per line or comma separated)",
                            SAMPLE_STACK, height=170)
    with b:
        k2 = st.slider("Recommendations", 5, 30, 10, key="k2")
        go2 = st.button("Recommend for this stack", type="primary", width="stretch")
        st.caption("Versions, extras, comments and options (-r, -e, urls) are ignored. "
                   "Known packages become the history; names we don't know are listed and skipped.")
    if go2:
        t0 = time.perf_counter()
        r = call("POST", "/recommend_stack", json={"text": text, "k": k2})
        st.session_state["stack"] = (r.status_code, r.json() if r.headers.get("content-type", "").startswith(
            "application/json") else {"detail": r.text}, time.perf_counter() - t0)
    if "stack" in st.session_state:
        code, data, dt = st.session_state["stack"]
        if code != 200:
            st.error(f"{code}: {data.get('detail')}")
        else:
            chips = "".join(f'<span class="aq-chip ok">{esc(short(m))}</span>' for m in data["matched"])
            chips += "".join(f'<span class="aq-chip ko" title="not in the catalogue">{esc(u)}</span>' for u in data["unknown"])
            st.markdown(f'<div class="aq-section">Read from your stack · {len(data["matched"])} known, '
                        f'{len(data["unknown"])} unknown · {dt:.2f} s</div>{chips}', unsafe_allow_html=True)
            left, right = st.columns([1, 2.4], gap="large")
            with left:
                with st.container(border=True):
                    render_better_tools(data["better_tools"])
            with right:
                render_interests(data["interests"], key="s")

# ---------------------------------------------------------------- how it works
with tab3:
    steps = [
        ("01 · DATA", "Real adoptions", "Release histories of PyPI projects (wheel metadata repaired) and GitHub "
         "developers' manifests: who adopted which package, and when: 44,703 adoption decisions."),
        ("02 · MODEL", "Multi-interest Transformer", "A small Transformer reads the last 50 adoptions (package, "
         "description, age) and builds 4 interest vectors. Each candidate is scored against its closest interest."),
        ("03 · EVALUATION", "Honest splits", "Leave-last-out and a strict time split, bootstrap confidence intervals, "
         "leakage checks. On LOO the model finds 3.7× more right packages than popularity for GitHub developers; "
         "on the time split it ties."),
        ("04 · SYSTEM", "API first", "FastAPI serves the trained checkpoint; this app only calls the API. "
         "Feedback goes to SQLite; recommendations stay in the user's ecosystems."),
    ]
    cols = st.columns(4)
    for col, (n, h, p) in zip(cols, steps):
        col.markdown(f'<div class="aq-step"><div class="n">{n}</div><h4>{h}</h4><p>{p}</p></div>',
                     unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="aq-section">API endpoints used by this page</div>', unsafe_allow_html=True)
    st.code("GET  /health                      model loaded, mock or not\n"
            "GET  /users                       users present in the dataset\n"
            "GET  /history?username=           the user's adoptions (what the model reads)\n"
            "GET  /recommend?username=&k=      top-k packages, grouped by interest\n"
            "POST /recommend_stack {text, k}   same, for a pasted requirements.txt / package.json\n"
            "GET  /better_tools?username=      verified switches (A -> B) other projects made\n"
            "POST /log_interaction             save a useful / not useful click (SQLite)\n"
            "GET  /interactions                read the saved feedback",
            language="text")
    st.caption(f"API at {API} · /health answered in {ping * 1000:.0f} ms · interactive docs at {API}/docs")

st.markdown('<div class="aq-foot">AgentQuiver · Ahmed Amine Zahraoui · Anas Boutaleb · Ahmed El Monaser · '
            'Lionel Monteiro Wora · ECE Paris</div>', unsafe_allow_html=True)
