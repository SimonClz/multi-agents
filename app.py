import os
import streamlit as st

# ⚠ Clé API chargée EN PREMIER — avant tout autre import
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

from langchain_core.messages import HumanMessage
from graph import create_graph

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Assistant Personnel",
    page_icon="🧠",
    layout="centered"
)

# ============================================================
# LABELS DES AGENTS
# ============================================================
AGENT_LABELS = {
    "assessment": "📋 Assessment Personnalité",
    "suivi_psy":  "🧠 Suivi Psychologique",
    "couple":     "💑 Couple",
    "education":  "👶 Éducation & Enfants",
    "synthese":   "📊 Synthèse & Évolution"
}

# ============================================================
# CHARGEMENT DU GRAPHE (une seule fois, mis en cache)
# ============================================================
@st.cache_resource
def get_graph():
    return create_graph()

graph = get_graph()

# ============================================================
# INITIALISATION DE LA SESSION
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

    st.markdown("<h1 style='text-align:center'>🧠 Assistant Personnel</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray'>Psychologie · Couple · Famille · Développement personnel</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.warning("""
    **⚠ Disclaimer**  
    Ce système est un outil d'exploration personnelle.  
    Il ne remplace pas un professionnel de santé, psychologue, thérapeute ou médecin agréé.  
    En cas de souffrance importante, consultez un professionnel.
    """)

    st.markdown("---")
    st.markdown("### 👤 Qui utilise le système ?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👨 Simon", use_container_width=True, type="primary"):
            st.session_state.thread_id            = "user_simon"
            st.session_state.prenom               = "Simon"
            st.session_state.user_selected        = True
            st.session_state.messages_display     = []
            st.session_state.assessment_triggered = False
            st.rerun()

    with col2:
        if st.button("👩 Mallorie", use_container_width=True, type="primary"):
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

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.prenom}")
        st.markdown("---")
        st.markdown("**🤖 Agents disponibles**")
        for label in AGENT_LABELS.values():
            st.markdown(f"• {label}")
        st.markdown("---")
        if st.button("🔄 Changer d'utilisateur", use_container_width=True):
            st.session_state.user_selected        = False
            st.session_state.messages_display     = []
            st.session_state.assessment_triggered = False
            st.rerun()

    # --- Header ---
    st.markdown(f"## 🧠 Bonjour {st.session_state.prenom} !")

    # --- Auto-démarrage assessment (nouveaux utilisateurs) ---
    if not st.session_state.assessment_triggered:
        try:
            etat     = graph.get_state(config)
            est_nouveau = len(etat.values.get("messages", [])) == 0
        except:
            est_nouveau = True

        if est_nouveau and not st.session_state.messages_display:
            with st.spinner("⏳ Démarrage de l'assessment initial..."):
                result = graph.invoke(
                    {
                        "messages":   [HumanMessage(content="Bonjour, je commence.")],
                        "next_agent": ""
                    },
                    config=config
                )
                reponse      = result["messages"][-1].content
                agent_utilise = result.get("last_agent", "assessment")

                st.session_state.messages_display.append({
                    "role":    "assistant",
                    "content": reponse,
                    "agent":   agent_utilise
                })

        st.session_state.assessment_triggered = True
        st.rerun()

    # --- Historique des messages ---
    for msg in st.session_state.messages_display:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                agent = msg.get("agent", "")
                label = AGENT_LABELS.get(agent, "🤖 Assistant")
                st.caption(f"*Agent actif : {label}*")
                st.write(msg["content"])

    # --- Zone de saisie ---
    if user_input := st.chat_input(f"Écrivez votre message, {st.session_state.prenom}..."):

        # Affichage immédiat du message utilisateur
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        st.session_state.messages_display.append({
            "role":    "user",
            "content": user_input
        })

        # Appel au graphe et affichage de la réponse
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("⏳ Traitement en cours..."):
                try:
                    result = graph.invoke(
                        {
                            "messages":   [HumanMessage(content=user_input)],
                            "next_agent": ""
                        },
                        config=config
                    )

                    reponse       = result["messages"][-1].content
                    agent_utilise = result.get("last_agent", "")
                    label         = AGENT_LABELS.get(agent_utilise, "🤖 Assistant")

                    st.caption(f"*Agent actif : {label}*")
                    st.write(reponse)

                    st.session_state.messages_display.append({
                        "role":    "assistant",
                        "content": reponse,
                        "agent":   agent_utilise
                    })

                    # Notification assessment complété
                    if result.get("assessment_done"):
                        st.success("✅ Assessment complété ! Profil établi — le suivi personnalisé commence.")

                except Exception as e:
                    st.error(f"⚠ Erreur : {e}")

        st.rerun()