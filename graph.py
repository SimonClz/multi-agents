from typing import TypedDict, Annotated, List
import operator
import sqlite3

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agents import (
    agent_orchestrateur,
    agent_assessment,
    agent_suivi_psy,
    agent_couple,
    agent_education,
    agent_synthese
)

# ============================================================
# ÉTAT PARTAGÉ — 2 nouveaux champs ajoutés
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str
    session_count: int
    assessment_done: bool  # ← NOUVEAU : assessment complété ?
    user_profile: str      # ← NOUVEAU : profil issu de l'assessment
    domain_notes: dict     # ← NOUVEAU : mémoire structurée par domaine
    last_agent: str    # ← NOUVEAU : pour afficher l'agent dans l'UI

# ============================================================
# ROUTING — Force l'assessment si pas encore fait
# ============================================================
def routing(state: AgentState) -> str:
    # Si l'assessment n'est PAS encore complété → toujours assessment
    if not state.get("assessment_done", False):
        return "assessment"
    # Sinon → décision de l'orchestrateur
    return state["next_agent"]

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

    conn = sqlite3.connect("conversations.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    return graph.compile(checkpointer=memory)