import os
from typing import TypedDict, Annotated, List
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

from agents import (
    agent_orchestrateur,
    agent_assessment,
    agent_suivi_psy,
    agent_couple,
    agent_education,
    agent_synthese
)

# ============================================================
# ÉTAT PARTAGÉ
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str
    session_count: int
    assessment_done: bool
    user_profile: str
    domain_notes: dict
    last_agent: str

# ============================================================
# ROUTING
# ============================================================
def routing(state: AgentState) -> str:
    if not state.get("assessment_done", False):
        return "assessment"
    return state["next_agent"]

# ============================================================
# RÉCUPÉRATION DE L'URL BASE DE DONNÉES
# ============================================================
def get_database_url():
    """Lit l'URL PostgreSQL depuis .env (local) ou Streamlit secrets (cloud)."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        try:
            import streamlit as st
            db_url = st.secrets.get("DATABASE_URL", "")
        except Exception:
            pass
    return db_url

# ============================================================
# CONSTRUCTION DU GRAPHE
# ============================================================
def create_graph():
    graph = StateGraph(AgentState)

    graph.add_node("orchestrateur", agent_orchestrateur)
    graph.add_node("assessment",    agent_assessment)
    graph.add_node("suivi_psy",     agent_suivi_psy)
    graph.add_node("couple",        agent_couple)
    graph.add_node("education",     agent_education)
    graph.add_node("synthese",      agent_synthese)

    graph.set_entry_point("orchestrateur")

    graph.add_conditional_edges(
        "orchestrateur",
        routing,
        {
            "assessment": "assessment",
            "suivi_psy":  "suivi_psy",
            "couple":     "couple",
            "education":  "education",
            "synthese":   "synthese",
        }
    )

    graph.add_edge("assessment", END)
    graph.add_edge("suivi_psy",  END)
    graph.add_edge("couple",     END)
    graph.add_edge("education",  END)
    graph.add_edge("synthese",   END)

    # -------------------------------------------------------
    # CHECKPOINTER : PostgreSQL (cloud) ou SQLite (local)
    # -------------------------------------------------------
    db_url = get_database_url()

    if db_url:
        # ✅ Cloud : PostgreSQL Supabase (données permanentes)
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver
        
        conn = psycopg.connect(db_url, autocommit=True)
        memory = PostgresSaver(conn)
        memory.setup()
        print("✅ Connexion PostgreSQL (Supabase) établie")
        
    else:
        # 🔄 Local : SQLite (fallback)
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect("conversations.db", check_same_thread=False)
        memory = SqliteSaver(conn)
        print("🔄 Connexion SQLite locale")

    return graph.compile(checkpointer=memory)