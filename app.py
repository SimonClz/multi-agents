import os
import streamlit as st

# Clé API en premier
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

from langchain_core.messages import HumanMessage
from graph import create_graph

# ============================================================
# CONFIGURATION PAGE
# ============================================================
st.set_page_config(
    page_title="Assistant Personnel",
    page_icon="🪴",
    layout="centered"
)

# Couleur principale (sage green)
COLOR = "#6B7348"
COLOR_DARK = "#525A38"

# ============================================================
# CSS PERSONNALISÉ
# ============================================================
st.markdown(f"""
<style>
/* --- Input border --- */
.stChatInput > div {{
    border-color: {COLOR} !important;
    border-radius: 12px !important;
}}
.stChatInput > div:focus-within {{
    border-color: {COLOR} !important;
    box-shadow: 0 0 0 1.5px {COLOR} !important;
}}

/* --- Boutons principaux --- */
.stButton > button {{
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    background-color: {COLOR} !important;
    border-color: {COLOR} !important;
    color: white !important;
}}
.stButton > button:hover {{
    background-color: {COLOR_DARK} !important;
    border-color: {COLOR_DARK} !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(107, 115, 72, 0.35) !important;
}}

/* --- Agents sidebar (non cliquables) --- */
.agent-tag {{
    display: flex;
    align-items: center;
    padding: 5px 10px;
    margin: 4px 0;
    background-color: #f0f2eb;
    border-left: 3px solid {COLOR};
    border-radius: 0 6px 6px 0;
    font-size: 0.85em;
    color: #444;
    cursor: default;
    user-select: none;
}}

/* --- Welcome message --- */
.welcome-box {{
    text-align: center;
    padding: 60px 20px;
    color: #888;
}}
.welcome-box h3 {{
    color: #555;
    margin-bottom: 10px;
}}

/* --- Masquer éléments Streamlit par défaut --- */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LABELS AGENTS
# ============================================================
AGENT_LABELS = {
    "assessment": "📋 Assessment Personnalité",
    "suivi_psy":  "🧠 Suivi Psychologique",
    "couple":     "💑 Couple",
    "education":  "👶 Éducation & Enfants",
    "synthese":   "📊 Synthèse & Évolution"
}

# ============================================================
# CHARGEMENT DU GRAPHE
# ============================================================
@st.cache_resource
def get_graph():
    return create_graph()

graph = get_graph()

# ============================================================
# INITIALISATION SESSION
# ============================================================
def init_session():
    defaults = {
        "user_selected":        False,
        "thread_id":            None,
        "prenom":               None,
        "messages_display":     [],
        "assessment_triggered": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

# ============================================================
# PAGE 1 — SÉLECTION UTILISATEUR
# ============================================================
if not st.session_state.user_selected:

    st.markdown("<h1 style='text-align:center; margin-top:40px'>🪴 Assistant Personnel</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray; margin-bottom:30px'>Psychologie · Couple · Famille · Développement personnel</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<h3 style='text-align:center'>👤 Qui utilise le système ?</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👨 Simon", use_container_width=True):
            st.session_state.thread_id            = "user_simon"
            st.session_state.prenom               = "Simon"
            st.session_state.user_selected        = True
            st.session_state.messages_display     = []
            st.session_state.assessment_triggered = False
            st.rerun()

    with col2:
        if st.button("👩 Mallorie", use_container_width=True):
            st.session_state.thread_id            = "user_mallorie"
            st.session_state.prenom               = "Mallorie"
            st.session_state.user_selected        = True
            st.session_state.messages_display     = []
            st.session_state.assessment_triggered = False
            st.rerun()

# ============================================================
# PAGE 2 — INTERFACE CHAT
# ============================================================
else:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # --- Récupération des infos de session ---
    try:
        current_state = graph.get_state(config)
        vals = current_state.values
        session_count   = vals.get("session_count", 0)
        assessment_done = vals.get("assessment_done", False)
        nb_messages     = len(vals.get("messages", []))
        assessment_status = "✅ Complété" if assessment_done else "⏳ En cours"
    except:
        session_count     = 0
        assessment_done   = False
        nb_messages       = 0
        assessment_status = "⏳ À démarrer"

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.prenom}")
        st.markdown("---")

        # Agents (non cliquables)
        st.markdown("**Agents disponibles**")
        for label in AGENT_LABELS.values():
            st.markdown(f'<div class="agent-tag">{label}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Infos de suivi
        st.markdown("**📊 Votre suivi**")
        st.caption(f"🗂 Assessment : {assessment_status}")
        st.caption(f"💬 Sessions : {session_count}")
        st.caption(f"📝 Messages échangés : {nb_messages}")

        st.markdown("---")
        if st.button("🔄 Changer d'utilisateur", use_container_width=True):
            st.session_state.user_selected        = False
            st.session_state.messages_display     = []
            st.session_state.assessment_triggered = False
            st.rerun()

    # --- Header ---
    st.markdown(f"## 🪴 Bonjour {st.session_state.prenom} !")

    # --- Auto-trigger assessment nouveaux utilisateurs ---
    if not st.session_state.assessment_triggered:
        try:
            etat = graph.get_state(config)
            est_nouveau = len(etat.values.get("messages", [])) == 0
        except:
            est_nouveau = True

        if est_nouveau and not st.session_state.messages_display:
            with st.spinner("⏳ Démarrage de l'assessment initial..."):
                result = graph.invoke(
                    {"messages": [HumanMessage(content="Bonjour, je commence.")], "next_agent": ""},
                    config=config
                )
                reponse       = result["messages"][-1].content
                agent_utilise = result.get("last_agent", "assessment")
                st.session_state.messages_display.append({
                    "role":    "assistant",
                    "content": reponse,
                    "agent":   agent_utilise
                })

        st.session_state.assessment_triggered = True
        st.rerun()

    # --- Message d'accueil si chat vide ---
    if not st.session_state.messages_display:
        st.markdown("""
        <div class="welcome-box">
            <div style="font-size: 3em; margin-bottom: 16px">🪴</div>
            <h3>Bienvenue dans votre espace personnel</h3>
            <p>Votre assistant est là pour vous accompagner dans votre développement personnel,<br>
            votre vie de couple et l'éducation de vos enfants.</p>
            <p style="margin-top: 20px; font-size: 0.9em">
                Envoyez un premier message pour commencer votre parcours.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- Historique des messages ---
    for msg in st.session_state.messages_display:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🪴"):
                agent = msg.get("agent", "")
                label = AGENT_LABELS.get(agent, "🪴 Assistant")
                st.caption(f"*Agent actif : {label}*")
                st.write(msg["content"])

    # --- Zone de saisie ---
    if user_input := st.chat_input(f"Écrivez votre message, {st.session_state.prenom}..."):

        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        st.session_state.messages_display.append({
            "role":    "user",
            "content": user_input
        })

        with st.chat_message("assistant", avatar="🪴"):
            with st.spinner("⏳ Traitement en cours..."):
                try:
                    result = graph.invoke(
                        {"messages": [HumanMessage(content=user_input)], "next_agent": ""},
                        config=config
                    )
                    reponse       = result["messages"][-1].content
                    agent_utilise = result.get("last_agent", "")
                    label         = AGENT_LABELS.get(agent_utilise, "🪴 Assistant")

                    st.caption(f"*Agent actif : {label}*")
                    st.write(reponse)

                    st.session_state.messages_display.append({
                        "role":    "assistant",
                        "content": reponse,
                        "agent":   agent_utilise
                    })

                    if result.get("assessment_done"):
                        st.success("✅ Assessment complété ! Votre profil est établi.")

                except Exception as e:
                    st.error(f"⚠ Erreur : {e}")

        st.rerun()