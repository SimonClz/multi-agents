import os
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ============================================================
# CHARGEMENT DES SECRETS
# ============================================================
load_dotenv()

try:
    import streamlit as st
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

# ============================================================
# LLM PARTAGÉ
# ============================================================
_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.3
        )
    return _llm_instance

# ============================================================
# PROMPTS TRANSVERSAUX
# ============================================================
PROMPT_TRANSVERSAL = """
RÈGLES CONVERSATIONNELLES STRICTES :
- Traite UN seul sujet principal par réponse.
- Pose UNE seule question principale.
- Une deuxième question n'est autorisée que si elle est directement liée au même sujet et reste légère.
- N'ouvre jamais un nouveau thème dans la même réponse.
- Adapte ton niveau de vocabulaire à celui de l'utilisateur.
- Si l'utilisateur s'exprime avec des mots simples, réponds avec un langage simple, concret et sans jargon.
- Si tu utilises un terme psychologique, explique-le immédiatement en mots simples.
- Distingue toujours : faits rapportés, hypothèses, incertitudes.
- Ne pose jamais de diagnostic médical ou psychiatrique.
- Si un sujet est sensible, reste prudent, clair et humain.
"""

PROMPT_ORCHESTRATEUR = f"""
Tu es l'orchestrateur central d'un système multi-agents d'accompagnement personnel, relationnel et familial.

Ta mission est de choisir l'agent le plus utile pour le prochain tour.

{PROMPT_TRANSVERSAL}

AGENTS DISPONIBLES :
- assessment : construction progressive d'un profil de fonctionnement personnel
- suivi_psy : vécu du moment, émotions, stress, réactions, situations sensibles
- couple : relation de couple, conflits, communication, malentendus
- education : parentalité, enfant, développement, décisions éducatives
- synthese : bilan global, évolution, cohérence transversale

RÈGLES :
- Détresse, confusion, crise, violence, besoin immédiat de soutien → suivi_psy
- Demande de compréhension globale de soi → assessment
- Sujet centré sur la relation de couple → couple
- Sujet centré sur un enfant ou une décision parentale → education
- Demande de bilan global → synthese
- Si plusieurs thèmes coexistent, choisis l'agent le plus utile maintenant
- Si ambiguïté → suivi_psy

Réponds UNIQUEMENT en JSON valide :
{{
  "next_agent": "assessment|suivi_psy|couple|education|synthese",
  "current_topic": "expression_courte"
}}
"""

PROMPT_ASSESSMENT_COMMON = f"""
Tu es un agent d'assessment psychologique non clinique.

Ta mission est de construire progressivement un profil de fonctionnement personnel utile, prudent et évolutif.

{PROMPT_TRANSVERSAL}

RÈGLES SPÉCIFIQUES :
- Tu suis strictement la phase d'assessment indiquée dans le contexte.
- Tu restes sur la phase actuelle sauf si le contexte indique explicitement un module optionnel ou la finalisation.
- Tu explores en profondeur modérée, pas en largeur.
- Tu ne fais pas un interrogatoire.
- Tu ne tires pas de conclusion forte à partir d'un seul indice.
- Tu peux reformuler brièvement si cela aide la clarté.
- Si l'utilisateur change clairement de sujet, tu l'accueilles sans perdre totalement le cadre.
"""

PROMPT_ASSESSMENT_PHASES = {
    "contexte": """
PHASE ACTUELLE : CONTEXTE

OBJECTIF :
- comprendre la situation de vie actuelle
- comprendre ce qui amène l'utilisateur
- identifier la préoccupation principale du moment

ATTENDU :
- rester concret
- une seule question principale
- ne pas ouvrir encore les dimensions profondes si le contexte n'est pas clair
""",
    "emotions": """
PHASE ACTUELLE : FONCTIONNEMENT ÉMOTIONNEL

OBJECTIF :
- comprendre comment l'utilisateur vit le stress ou les émotions difficiles
- identifier ses réactions habituelles
- repérer sa manière de réguler ce qu'il ressent

ATTENDU :
- rester centré sur le vécu émotionnel
- une seule question principale
- ne pas ouvrir d'autres domaines en parallèle
""",
    "relations": """
PHASE ACTUELLE : FONCTIONNEMENT RELATIONNEL

OBJECTIF :
- comprendre comment l'utilisateur fonctionne dans les relations
- repérer son rapport au conflit, à la proximité, au retrait, à l'adaptation
- observer les schémas relationnels généraux

ATTENDU :
- analyser des exemples concrets
- éviter l'étiquetage rapide
- une seule question principale
""",
    "valeurs": """
PHASE ACTUELLE : VALEURS ET BESOINS

OBJECTIF :
- identifier ce qui compte vraiment pour l'utilisateur
- clarifier ses besoins les plus saillants
- repérer ce qui est non négociable pour lui

ATTENDU :
- rester simple et concret
- éviter les grandes abstractions si l'utilisateur n'y va pas lui-même
- une seule question principale
""",
    "objectifs": """
PHASE ACTUELLE : OBJECTIFS DE CHANGEMENT

OBJECTIF :
- clarifier ce que l'utilisateur veut changer
- comprendre ce qu'il espère de cet accompagnement
- définir ce qui ressemblerait à une amélioration utile

ATTENDU :
- viser du concret
- une seule question principale
- préparer la suite de l'assessment
""",
    "module_optionnel": """
PHASE ACTUELLE : MODULE OPTIONNEL

OBJECTIF :
- approfondir un seul domaine optionnel jugé pertinent
- rester ciblé et bref
- produire une meilleure compréhension sans relancer un nouvel assessment complet

ATTENDU :
- se concentrer uniquement sur le module indiqué dans le contexte
- une seule question principale
- rester très concret
- éviter de repartir en largeur
""",
    "finalisation": """
PHASE ACTUELLE : FINALISATION

OBJECTIF :
- ne plus ouvrir de nouveaux sujets
- préparer ou générer le profil final

ATTENDU :
- si une validation légère est nécessaire, elle doit rester minimale
- si les informations sont suffisantes, produire le profil final
"""
}

PROMPT_OPTIONAL_MODULES = {
    "couple": """
MODULE OPTIONNEL ACTIF : COUPLE

OBJECTIF :
- comprendre ce qui pèse le plus dans la relation de couple aujourd'hui
- identifier une dynamique utile sans entrer dans une analyse complète de couple

QUESTIONNEMENT :
- une seule question principale
- rester centré sur ce qui est le plus important maintenant dans le couple
""",
    "parentalite": """
MODULE OPTIONNEL ACTIF : PARENTALITÉ

OBJECTIF :
- comprendre ce qui est le plus difficile actuellement dans le rôle parental
- identifier la difficulté centrale sans ouvrir plusieurs fronts

QUESTIONNEMENT :
- une seule question principale
- rester centré sur le défi parental principal
""",
    "famille_origine": """
MODULE OPTIONNEL ACTIF : FAMILLE D'ORIGINE

OBJECTIF :
- comprendre quel élément de la famille d'origine reste le plus actif aujourd'hui
- faire un lien prudent avec le fonctionnement actuel

QUESTIONNEMENT :
- une seule question principale
- rester centré sur l'élément le plus influent aujourd'hui
""",
    "travail": """
MODULE OPTIONNEL ACTIF : TRAVAIL

OBJECTIF :
- comprendre ce qui pèse le plus dans la vie professionnelle actuelle
- identifier l'impact principal sur l'équilibre personnel

QUESTIONNEMENT :
- une seule question principale
- rester centré sur la difficulté professionnelle la plus importante
"""
}

PROMPT_SUIVI_PSY = f"""
Tu es un agent de suivi psychologique continu, non clinique.

Ta mission est d'aider l'utilisateur à comprendre une situation du moment, ce qu'il ressent, ce qu'il se dit, et ce qu'il fait.

{PROMPT_TRANSVERSAL}

STRUCTURE INTERNE :
1. clarifier la situation actuelle
2. repérer l'émotion dominante
3. repérer la pensée ou l'interprétation centrale si elle apparaît
4. repérer la réaction
5. distinguer ponctuel / récurrent
6. proposer au plus une piste concrète

RÈGLES :
- si l'utilisateur cherche surtout à être compris, ne force pas un exercice
- une seule piste pratique maximum
- si un signal sensible apparaît, privilégie sécurité et clarté

À la toute fin, ajoute exactement :
[NOTE_PSY_JSON]
{{
  "facts": ["..."],
  "hypotheses": ["..."],
  "confidence": "low|medium|high",
  "priority": "...",
  "links": {{
    "couple": "...",
    "education": "...",
    "assessment": "..."
  }}
}}
[/NOTE_PSY_JSON]
"""

PROMPT_COUPLE = f"""
Tu es un agent d'analyse des dynamiques de couple, non clinique.

Ta mission est d'aider à comprendre une interaction ou une boucle relationnelle.

{PROMPT_TRANSVERSAL}

RÈGLES :
- analyse d'abord l'interaction, pas la personnalité
- ne prends jamais parti
- si un seul partenaire parle, rappelle implicitement ou explicitement que tu n'as qu'un seul point de vue
- n'attribue jamais avec certitude une intention à l'autre
- reste concret

STRUCTURE INTERNE :
1. fait interactionnel
2. déclencheur
3. perception de l'utilisateur
4. émotion ou besoin probable
5. boucle relationnelle possible
6. une piste de communication ou d'observation

À la toute fin, ajoute exactement :
[NOTE_COUPLE_JSON]
{{
  "facts": ["..."],
  "hypotheses": ["..."],
  "confidence": "low|medium|high",
  "priority": "...",
  "links": {{
    "psy": "...",
    "education": "...",
    "assessment": "..."
  }}
}}
[/NOTE_COUPLE_JSON]
"""

PROMPT_EDUCATION = f"""
Tu es un agent d'aide à la réflexion éducative, non médical.

Ta mission est d'aider l'utilisateur à comprendre une situation parent-enfant et à comparer des options adaptées.

{PROMPT_TRANSVERSAL}

RÈGLES :
- commence par l'âge, le contexte, la fréquence et l'intensité si nécessaire
- adapte toujours l'analyse au stade de développement
- évite les dogmes
- présente peu d'options, mais clairement
- n'explique pas trop vite le comportement de l'enfant par la psychologie du parent

STRUCTURE INTERNE :
1. difficulté ciblée
2. contexte développemental
3. hypothèses prudentes
4. 2 options maximum
5. avantages / limites
6. ce qu'il faut observer ensuite

À la toute fin, ajoute exactement :
[NOTE_EDUCATION_JSON]
{{
  "facts": ["..."],
  "hypotheses": ["..."],
  "confidence": "low|medium|high",
  "priority": "...",
  "links": {{
    "psy": "...",
    "couple": "...",
    "assessment": "..."
  }}
}}
[/NOTE_EDUCATION_JSON]
"""

PROMPT_SYNTHESE = f"""
Tu es l'agent de synthèse transversale.

Ta mission est de produire un bilan global prudent, utile et clair à partir du profil et des notes disponibles.

{PROMPT_TRANSVERSAL}

RÈGLES :
- distingue ce qui semble confirmé, probable, ou encore à vérifier
- reste structuré
- ne surestime pas la précision des données

STRUCTURE DE SORTIE :
- Ce qui semble stable
- Ce qui a évolué
- Ce qui reste difficile ou répétitif
- Points forts
- Priorités
- Points à vérifier

À la toute fin, ajoute exactement :
[NOTE_SYNTHESE_JSON]
{{
  "facts": ["..."],
  "hypotheses": ["..."],
  "confidence": "low|medium|high",
  "priority": "...",
  "links": {{
    "psy": "...",
    "couple": "...",
    "education": "...",
    "assessment": "..."
  }}
}}
[/NOTE_SYNTHESE_JSON]
"""

PROMPT_PROFILE_GENERATOR = """
Tu génères un profil final d'assessment à partir de données structurées.

RÈGLES :
- tu restes prudent
- tu ne transformes pas une hypothèse en certitude
- tu ne rajoutes rien qui ne soit pas soutenu par les données
- tu utilises un langage clair
- si une section est peu documentée, tu le dis brièvement

Format exact attendu :

[PROFIL_ÉTABLI]
== CONTEXTE ==
...

== FONCTIONNEMENT ÉMOTIONNEL ==
...

== FONCTIONNEMENT RELATIONNEL ==
...

== VALEURS ET BESOINS ==
...

== OBJECTIFS ==
...

== MODULES OPTIONNELS ==
- Couple : ...
- Parentalité : ...
- Famille d'origine : ...
- Travail : ...

== FORCES ==
...

== VIGILANCES ==
...
"""

PROMPT_PROFILE_SUMMARY = """
Tu transformes un profil technique en récapitulatif accessible pour l'utilisateur.

RÈGLES :
- ne rien inventer
- reformuler seulement
- rester chaleureux, clair, structuré, utile
- valoriser les forces sans exagérer
- évoquer les points de vigilance avec tact

Le message doit :
1. confirmer chaleureusement la fin de l'assessment
2. résumer les grands traits utiles
3. mettre en avant 3 ou 4 forces
4. mentionner 2 ou 3 axes de vigilance ou de progression
5. finir de façon ouverte et rassurante

Longueur : 250 à 500 mots.
"""

# ============================================================
# MARQUEURS
# ============================================================
MARQUEUR_PROFIL = "[PROFIL_ÉTABLI]"

NOTE_MARKERS = {
    "psy": ("[NOTE_PSY_JSON]", "[/NOTE_PSY_JSON]"),
    "couple": ("[NOTE_COUPLE_JSON]", "[/NOTE_COUPLE_JSON]"),
    "education": ("[NOTE_EDUCATION_JSON]", "[/NOTE_EDUCATION_JSON]"),
    "synthese": ("[NOTE_SYNTHESE_JSON]", "[/NOTE_SYNTHESE_JSON]")
}

# ============================================================
# UTILITAIRES GÉNÉRAUX
# ============================================================
def now_iso():
    return datetime.utcnow().isoformat()

def get_last_user_message(messages):
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def safe_json_load(text: str, fallback: dict):
    try:
        return json.loads(text)
    except Exception:
        return fallback

def extract_json_block(content: str, start_marker: str, end_marker: str):
    if start_marker not in content or end_marker not in content:
        return content.strip(), None

    before = content.split(start_marker, 1)[0].strip()
    json_part = content.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()

    try:
        parsed = json.loads(json_part)
    except Exception:
        parsed = None

    return before, parsed

def infer_user_style(text: str, previous_style: dict | None = None) -> dict:
    text = (text or "").strip()
    words = text.split()

    if len(words) < 10:
        vocab_level = "simple"
    elif len(words) < 28:
        vocab_level = "standard"
    else:
        vocab_level = "elabore"

    style = {
        "vocab_level": vocab_level,
        "response_length": "medium",
        "prefers_examples": vocab_level != "elabore"
    }

    if previous_style:
        style["response_length"] = previous_style.get("response_length", "medium")
        style["prefers_examples"] = previous_style.get(
            "prefers_examples",
            style["prefers_examples"]
        )

    return style

def detect_risk_flags(text: str) -> dict:
    t = (text or "").lower()

    return {
        "self_harm": any(x in t for x in [
            "suicide", "me tuer", "envie d'en finir", "me faire du mal", "je veux mourir"
        ]),
        "violence": any(x in t for x in [
            "il me frappe", "elle me frappe", "violence", "agression", "menace"
        ]),
        "child_safety": any(x in t for x in [
            "mon enfant est en danger", "maltraitance", "je frappe mon enfant", "enfant en danger"
        ]),
        "acute_distress": any(x in t for x in [
            "je craque", "j'en peux plus", "je n'en peux plus", "à bout", "effondré", "effondrée"
        ])
    }

def normalize_state(state: dict) -> dict:
    s = dict(state or {})

    if "user_profile" not in s:
        s["user_profile"] = ""

    if "user_style" not in s or not isinstance(s.get("user_style"), dict):
        s["user_style"] = {}

    if "domain_notes" not in s or not isinstance(s.get("domain_notes"), dict):
        s["domain_notes"] = {}

    for domain in ["psy", "couple", "education", "synthese"]:
        if domain not in s["domain_notes"] or not isinstance(s["domain_notes"].get(domain), list):
            s["domain_notes"][domain] = []

    if "active_memory" not in s or not isinstance(s.get("active_memory"), dict):
        s["active_memory"] = {
            "facts": [],
            "hypotheses": [],
            "priorities": [],
            "points_to_verify": []
        }

    if "risk_flags" not in s or not isinstance(s.get("risk_flags"), dict):
        s["risk_flags"] = {
            "self_harm": False,
            "violence": False,
            "child_safety": False,
            "acute_distress": False
        }

    if "assessment_state" not in s or not isinstance(s.get("assessment_state"), dict):
        s["assessment_state"] = init_assessment_state()

    if "current_topic" not in s:
        s["current_topic"] = ""

    if "session_count" not in s:
        s["session_count"] = 0

    if "total_turns" not in s:
        s["total_turns"] = 0

    if "assessment_done" not in s:
        s["assessment_done"] = False

    if "last_agent" not in s:
        s["last_agent"] = ""

    if "next_agent" not in s:
        s["next_agent"] = ""

    return s

def add_domain_note(state: dict, domaine: str, note: dict | None) -> dict:
    notes = dict(state.get("domain_notes", {}))
    historique = list(notes.get(domaine, []))

    if note:
        note["timestamp"] = now_iso()
        historique.append(note)

    notes[domaine] = historique[-10:]
    return notes

def build_active_memory(state: dict) -> dict:
    domain_notes = state.get("domain_notes", {}) or {}
    active = {
        "facts": [],
        "hypotheses": [],
        "priorities": [],
        "points_to_verify": []
    }

    for domaine, notes in domain_notes.items():
        if not notes:
            continue

        latest = notes[-1]
        active["facts"].extend(latest.get("facts", [])[:2])
        active["hypotheses"].extend(latest.get("hypotheses", [])[:1])

        priority = latest.get("priority")
        if priority:
            active["priorities"].append(f"{domaine}: {priority}")

    active["facts"] = active["facts"][:5]
    active["hypotheses"] = active["hypotheses"][:3]
    active["priorities"] = active["priorities"][:3]

    return active

# ============================================================
# ASSESSMENT STATE
# ============================================================
def init_assessment_state(existing: dict | None = None) -> dict:
    base = {
        "phase_index": 0,
        "phase_name": "contexte",
        "covered_phases": [],
        "phase_turn_count": 0,
        "last_question_topic": "",
        "ready_to_finalize": False,
        "optional_modules_queue": [],
        "completed_optional_modules": [],
        "current_optional_module": "",
        "max_optional_modules": 2,
        "profile_data": {
            "contexte": {
                "situation_actuelle": "",
                "motif_principal": "",
                "elements_utiles": []
            },
            "emotions": {
                "emotions_dominantes": [],
                "stress_pattern": "",
                "regulation_style": "",
                "elements_utiles": []
            },
            "relations": {
                "relation_style": "",
                "rapport_au_conflit": "",
                "patterns_relationnels": [],
                "elements_utiles": []
            },
            "valeurs": {
                "valeurs_centrales": [],
                "besoins_saillants": [],
                "elements_utiles": []
            },
            "objectifs": {
                "changements_vises": [],
                "vision_amelioration": "",
                "elements_utiles": []
            },
            "modules_optionnels": {
                "couple": [],
                "parentalite": [],
                "famille_origine": [],
                "travail": []
            }
        }
    }

    if existing:
        base.update(existing)
        if "profile_data" in existing:
            for key, value in base["profile_data"].items():
                if key in existing["profile_data"] and isinstance(value, dict):
                    if isinstance(existing["profile_data"][key], dict):
                        value.update(existing["profile_data"][key])

    return base

ASSESSMENT_PHASES = [
    "contexte",
    "emotions",
    "relations",
    "valeurs",
    "objectifs"
]

OPTIONAL_MODULES = ["couple", "parentalite", "famille_origine", "travail"]

def get_current_phase_name(state: dict) -> str:
    astate = init_assessment_state(state.get("assessment_state", {}))
    return astate.get("phase_name", "contexte")

def is_meaningful_text(text: str) -> bool:
    return len((text or "").strip().split()) >= 5

def dedupe_keep_order(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result

def register_optional_signals(assessment_state: dict, text: str) -> dict:
    lower = (text or "").lower()
    mods = assessment_state["profile_data"]["modules_optionnels"]

    if any(k in lower for k in ["couple", "compagne", "compagnon", "conjoint", "mari", "femme", "relation amoureuse"]):
        mods["couple"].append(text[:220])

    if any(k in lower for k in ["enfant", "fils", "fille", "parentalité", "parent", "éducation"]):
        mods["parentalite"].append(text[:220])

    if any(k in lower for k in ["parents", "enfance", "famille d'origine", "chez mes parents", "quand j'étais enfant"]):
        mods["famille_origine"].append(text[:220])

    if any(k in lower for k in ["travail", "job", "boulot", "professionnel", "collègue", "manager", "burn out", "burnout"]):
        mods["travail"].append(text[:220])

    return assessment_state

def update_profile_data_for_phase(assessment_state: dict, phase_name: str, user_text: str) -> dict:
    text = (user_text or "").strip()
    lower = text.lower()
    pdata = assessment_state["profile_data"]

    if not text:
        return assessment_state

    assessment_state = register_optional_signals(assessment_state, text)

    if phase_name == "contexte":
        if not pdata["contexte"]["situation_actuelle"]:
            pdata["contexte"]["situation_actuelle"] = text[:500]
        elif not pdata["contexte"]["motif_principal"]:
            pdata["contexte"]["motif_principal"] = text[:300]
        else:
            pdata["contexte"]["elements_utiles"].append(text[:250])

    elif phase_name == "emotions":
        pdata["emotions"]["elements_utiles"].append(text[:250])

        if any(k in lower for k in ["stress", "angoisse", "anxiété", "pression", "submergé", "submergée"]):
            pdata["emotions"]["stress_pattern"] = text[:250]

        if any(k in lower for k in ["colère", "triste", "fatigue", "peur", "honte", "culpabilité"]):
            pdata["emotions"]["emotions_dominantes"].append(text[:120])

        if any(k in lower for k in ["j'évite", "je me retire", "je parle", "je garde", "je rumine", "je m'isole"]):
            pdata["emotions"]["regulation_style"] = text[:250]

    elif phase_name == "relations":
        pdata["relations"]["elements_utiles"].append(text[:250])

        if any(k in lower for k in ["conflit", "dispute", "désaccord", "tension", "accrochage"]):
            pdata["relations"]["rapport_au_conflit"] = text[:250]

        if any(k in lower for k in ["je me retire", "j'essaie de parler", "je m'adapte", "j'évite", "je me ferme"]):
            pdata["relations"]["relation_style"] = text[:250]

        pdata["relations"]["patterns_relationnels"].append(text[:150])

    elif phase_name == "valeurs":
        pdata["valeurs"]["elements_utiles"].append(text[:250])

        if any(k in lower for k in ["important", "essentiel", "compte", "priorité", "non négociable", "fondamental"]):
            pdata["valeurs"]["valeurs_centrales"].append(text[:140])

        pdata["valeurs"]["besoins_saillants"].append(text[:140])

    elif phase_name == "objectifs":
        pdata["objectifs"]["elements_utiles"].append(text[:250])

        if any(k in lower for k in ["changer", "améliorer", "mieux", "retrouver", "progresser", "évoluer"]):
            pdata["objectifs"]["changements_vises"].append(text[:160])

        if not pdata["objectifs"]["vision_amelioration"]:
            pdata["objectifs"]["vision_amelioration"] = text[:250]

    elif phase_name == "module_optionnel":
        module_name = assessment_state.get("current_optional_module", "")
        if module_name in pdata["modules_optionnels"]:
            pdata["modules_optionnels"][module_name].append(text[:250])

    return assessment_state

def is_phase_complete(assessment_state: dict, phase_name: str) -> bool:
    pdata = assessment_state["profile_data"]
    turns = assessment_state.get("phase_turn_count", 0)

    if phase_name == "contexte":
        return bool(pdata["contexte"]["situation_actuelle"]) and turns >= 1

    if phase_name == "emotions":
        return len(pdata["emotions"]["elements_utiles"]) >= 1 and turns >= 1

    if phase_name == "relations":
        return len(pdata["relations"]["elements_utiles"]) >= 1 and turns >= 1

    if phase_name == "valeurs":
        return len(pdata["valeurs"]["elements_utiles"]) >= 1 and turns >= 1

    if phase_name == "objectifs":
        return len(pdata["objectifs"]["elements_utiles"]) >= 1 and turns >= 1

    if phase_name == "module_optionnel":
        module_name = assessment_state.get("current_optional_module", "")
        if module_name in pdata["modules_optionnels"]:
            return len(pdata["modules_optionnels"][module_name]) >= 2 and turns >= 1
        return turns >= 1

    return False

def compute_optional_modules_queue(assessment_state: dict) -> list:
    pdata = assessment_state["profile_data"]["modules_optionnels"]
    scored = []

    for module in OPTIONAL_MODULES:
        score = len(pdata.get(module, []))
        if score > 0:
            scored.append((module, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    queue = [m for m, _ in scored[:assessment_state.get("max_optional_modules", 2)]]
    return dedupe_keep_order(queue)

def enter_next_optional_module_or_finalize(assessment_state: dict) -> dict:
    queue = assessment_state.get("optional_modules_queue", [])
    completed = assessment_state.get("completed_optional_modules", [])

    remaining = [m for m in queue if m not in completed]

    if remaining:
        assessment_state["phase_name"] = "module_optionnel"
        assessment_state["current_optional_module"] = remaining[0]
        assessment_state["phase_turn_count"] = 0
        assessment_state["phase_index"] = len(ASSESSMENT_PHASES)
    else:
        assessment_state["phase_name"] = "finalisation"
        assessment_state["current_optional_module"] = ""
        assessment_state["ready_to_finalize"] = True
        assessment_state["phase_index"] = len(ASSESSMENT_PHASES) + 1

    return assessment_state

def advance_phase(assessment_state: dict) -> dict:
    phase_name = assessment_state["phase_name"]

    if phase_name not in assessment_state["covered_phases"]:
        assessment_state["covered_phases"].append(phase_name)

    if phase_name in ASSESSMENT_PHASES:
        idx = ASSESSMENT_PHASES.index(phase_name)

        if idx < len(ASSESSMENT_PHASES) - 1:
            assessment_state["phase_index"] = idx + 1
            assessment_state["phase_name"] = ASSESSMENT_PHASES[idx + 1]
            assessment_state["phase_turn_count"] = 0
            return assessment_state

        # Fin des phases coeur → calcul des modules optionnels
        queue = compute_optional_modules_queue(assessment_state)
        assessment_state["optional_modules_queue"] = queue
        return enter_next_optional_module_or_finalize(assessment_state)

    if phase_name == "module_optionnel":
        current_module = assessment_state.get("current_optional_module", "")
        if current_module and current_module not in assessment_state["completed_optional_modules"]:
            assessment_state["completed_optional_modules"].append(current_module)

        return enter_next_optional_module_or_finalize(assessment_state)

    if phase_name == "finalisation":
        assessment_state["ready_to_finalize"] = True

    return assessment_state

def update_assessment_state_with_user_input(state: dict, user_text: str) -> dict:
    astate = init_assessment_state(state.get("assessment_state", {}))
    phase_name = astate["phase_name"]

    if phase_name == "finalisation":
        return astate

    if is_meaningful_text(user_text):
        astate["phase_turn_count"] += 1
        astate = update_profile_data_for_phase(astate, phase_name, user_text)

        if is_phase_complete(astate, phase_name):
            astate = advance_phase(astate)

    return astate

def should_finalize_assessment(assessment_state: dict) -> bool:
    return assessment_state.get("ready_to_finalize", False)

def build_phase_prompt(phase_name: str, current_optional_module: str = "") -> str:
    prompt = PROMPT_ASSESSMENT_COMMON + "\n\n" + PROMPT_ASSESSMENT_PHASES.get(
        phase_name,
        PROMPT_ASSESSMENT_PHASES["contexte"]
    )

    if phase_name == "module_optionnel" and current_optional_module:
        prompt += "\n\n" + PROMPT_OPTIONAL_MODULES.get(current_optional_module, "")

    return prompt

def build_profile_text_from_state(assessment_state: dict) -> str:
    pdata = assessment_state["profile_data"]

    payload = {
        "contexte": pdata["contexte"],
        "emotions": pdata["emotions"],
        "relations": pdata["relations"],
        "valeurs": pdata["valeurs"],
        "objectifs": pdata["objectifs"],
        "modules_optionnels": pdata["modules_optionnels"],
        "completed_optional_modules": assessment_state.get("completed_optional_modules", [])
    }

    response = get_llm().invoke([
        SystemMessage(content=PROMPT_PROFILE_GENERATOR),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2))
    ])

    text = response.content.strip()
    if MARQUEUR_PROFIL not in text:
        text = f"{MARQUEUR_PROFIL}\n" + text
    return text

def build_assessment_summary_for_user(profile_text: str, user_style: dict) -> str:
    vocab_level = user_style.get("vocab_level", "standard")
    prefers_examples = user_style.get("prefers_examples", True)

    system = PROMPT_PROFILE_SUMMARY + f"""

Niveau de vocabulaire attendu : {vocab_level}
Préférence pour exemples concrets : {prefers_examples}
"""

    response = get_llm().invoke([
        SystemMessage(content=system),
        HumanMessage(content=profile_text)
    ])
    return response.content.strip()

# ============================================================
# CONTEXTE INJECTÉ
# ============================================================
def build_context(prompt_base: str, state: dict) -> str:
    s = normalize_state(state)

    user_profile = s.get("user_profile", "")
    user_style = s.get("user_style", {})
    active_memory = s.get("active_memory", {})
    current_topic = s.get("current_topic", "")

    context = prompt_base

    if user_profile:
        context += f"""

---
PROFIL DE RÉFÉRENCE
Ce profil est une synthèse hypothétique et évolutive.
Utilise-le comme repère, pas comme vérité certaine.
Si de nouveaux éléments le contredisent, signale-le explicitement.

{user_profile}
---"""

    if user_style:
        context += f"""

---
STYLE UTILISATEUR
- Niveau de vocabulaire : {user_style.get("vocab_level", "standard")}
- Longueur recommandée : {user_style.get("response_length", "medium")}
- Préférence pour exemples concrets : {user_style.get("prefers_examples", True)}
---"""

    if current_topic:
        context += f"""

---
SUJET EN COURS
Sujet principal actuel : {current_topic}
Reste sur ce sujet sauf si l'utilisateur change clairement de thème.
---"""

    if active_memory:
        context += f"""

---
MÉMOIRE ACTIVE
- Faits récents : {active_memory.get("facts", [])}
- Hypothèses prudentes : {active_memory.get("hypotheses", [])}
- Priorités : {active_memory.get("priorities", [])}
---"""

    return context

def build_assessment_context(state: dict, phase_name: str) -> str:
    s = normalize_state(state)
    astate = init_assessment_state(s.get("assessment_state", {}))
    current_optional_module = astate.get("current_optional_module", "")

    base_prompt = build_phase_prompt(phase_name, current_optional_module)
    context = build_context(base_prompt, s)

    context += f"""

---
ASSESSMENT STATE
- Phase actuelle : {astate.get("phase_name")}
- Phases couvertes : {astate.get("covered_phases", [])}
- Tours dans la phase actuelle : {astate.get("phase_turn_count", 0)}
- Module optionnel actif : {current_optional_module or "aucun"}
- File de modules optionnels : {astate.get("optional_modules_queue", [])}
- Modules optionnels déjà couverts : {astate.get("completed_optional_modules", [])}
- Ready to finalize : {astate.get("ready_to_finalize", False)}
---
"""

    return context

# ============================================================
# AGENTS
# ============================================================
def agent_orchestrateur(state):
    s = normalize_state(state)
    messages = s["messages"]
    last_user_text = get_last_user_message(messages)

    user_style = infer_user_style(last_user_text, s.get("user_style", {}))
    risk_flags = detect_risk_flags(last_user_text)

    if any(risk_flags.values()):
        return {
            "next_agent": "suivi_psy",
            "current_topic": "securite",
            "risk_flags": risk_flags,
            "user_style": user_style
        }

    if not s.get("assessment_done", False):
        urgent_markers = [
            "urgent", "je craque", "je n'en peux plus", "violence", "à bout", "aidez-moi"
        ]
        if any(m in (last_user_text or "").lower() for m in urgent_markers):
            return {
                "next_agent": "suivi_psy",
                "current_topic": "crise",
                "risk_flags": risk_flags,
                "user_style": user_style
            }

        return {
            "next_agent": "assessment",
            "current_topic": s.get("current_topic", "assessment"),
            "risk_flags": risk_flags,
            "user_style": user_style
        }

    response = get_llm().invoke([
        SystemMessage(content=PROMPT_ORCHESTRATEUR),
        HumanMessage(content=f"Message utilisateur : {last_user_text}")
    ])

    parsed = safe_json_load(response.content.strip(), {
        "next_agent": "suivi_psy",
        "current_topic": "suivi_general"
    })

    next_agent = parsed.get("next_agent", "suivi_psy")
    if next_agent not in ["assessment", "suivi_psy", "couple", "education", "synthese"]:
        next_agent = "suivi_psy"

    current_topic = parsed.get("current_topic", "suivi_general")

    print(f"\n🎯 Orchestrateur → Agent : {next_agent} | Sujet : {current_topic}\n")

    return {
        "next_agent": next_agent,
        "current_topic": current_topic,
        "risk_flags": risk_flags,
        "user_style": user_style
    }

def agent_assessment(state):
    s = normalize_state(state)
    messages = s["messages"]
    last_user_text = get_last_user_message(messages)

    assessment_state = update_assessment_state_with_user_input(s, last_user_text)
    phase_name = assessment_state.get("phase_name", "contexte")

    if should_finalize_assessment(assessment_state):
        profile_text = build_profile_text_from_state(assessment_state)
        user_profile = profile_text.split(MARQUEUR_PROFIL, 1)[1].strip() if MARQUEUR_PROFIL in profile_text else profile_text
        recap = build_assessment_summary_for_user(user_profile, s.get("user_style", {}))

        final_message = "## 🎯 Votre profil en résumé\n\n" + recap

        return {
            "messages": [AIMessage(content=final_message)],
            "next_agent": "fin",
            "assessment_done": True,
            "user_profile": user_profile,
            "assessment_state": assessment_state,
            "last_agent": "assessment",
            "total_turns": s.get("total_turns", 0) + 1,
            "session_count": s.get("session_count", 0) + 1,
            "active_memory": build_active_memory(s),
            "current_topic": "assessment_finalise"
        }

    system_prompt = build_assessment_context(
        {**s, "assessment_state": assessment_state},
        phase_name
    )

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])

    current_optional_module = assessment_state.get("current_optional_module", "")
    if phase_name == "module_optionnel" and current_optional_module:
        topic = f"assessment_module_{current_optional_module}"
    else:
        topic = f"assessment_{phase_name}"

    return {
        "messages": [AIMessage(content=response.content.strip())],
        "next_agent": "fin",
        "assessment_done": False,
        "assessment_state": assessment_state,
        "last_agent": "assessment",
        "total_turns": s.get("total_turns", 0) + 1,
        "session_count": s.get("session_count", 0) + 1,
        "active_memory": build_active_memory(s),
        "current_topic": topic
    }

def agent_suivi_psy(state):
    s = normalize_state(state)
    messages = s["messages"]
    system_prompt = build_context(PROMPT_SUIVI_PSY, s)

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])

    response_clean, note = extract_json_block(
        response.content,
        NOTE_MARKERS["psy"][0],
        NOTE_MARKERS["psy"][1]
    )

    domain_notes = add_domain_note(s, "psy", note)
    updated_state = {**s, "domain_notes": domain_notes}
    active_memory = build_active_memory(updated_state)

    return {
        "messages": [AIMessage(content=response_clean)],
        "next_agent": "fin",
        "domain_notes": domain_notes,
        "active_memory": active_memory,
        "last_agent": "suivi_psy",
        "total_turns": s.get("total_turns", 0) + 1,
        "session_count": s.get("session_count", 0) + 1
    }

def agent_couple(state):
    s = normalize_state(state)
    messages = s["messages"]
    system_prompt = build_context(PROMPT_COUPLE, s)

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])

    response_clean, note = extract_json_block(
        response.content,
        NOTE_MARKERS["couple"][0],
        NOTE_MARKERS["couple"][1]
    )

    domain_notes = add_domain_note(s, "couple", note)
    updated_state = {**s, "domain_notes": domain_notes}
    active_memory = build_active_memory(updated_state)

    return {
        "messages": [AIMessage(content=response_clean)],
        "next_agent": "fin",
        "domain_notes": domain_notes,
        "active_memory": active_memory,
        "last_agent": "couple",
        "total_turns": s.get("total_turns", 0) + 1,
        "session_count": s.get("session_count", 0) + 1
    }

def agent_education(state):
    s = normalize_state(state)
    messages = s["messages"]
    system_prompt = build_context(PROMPT_EDUCATION, s)

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])

    response_clean, note = extract_json_block(
        response.content,
        NOTE_MARKERS["education"][0],
        NOTE_MARKERS["education"][1]
    )

    domain_notes = add_domain_note(s, "education", note)
    updated_state = {**s, "domain_notes": domain_notes}
    active_memory = build_active_memory(updated_state)

    return {
        "messages": [AIMessage(content=response_clean)],
        "next_agent": "fin",
        "domain_notes": domain_notes,
        "active_memory": active_memory,
        "last_agent": "education",
        "total_turns": s.get("total_turns", 0) + 1,
        "session_count": s.get("session_count", 0) + 1
    }

def agent_synthese(state):
    s = normalize_state(state)
    messages = s["messages"]
    system_prompt = build_context(PROMPT_SYNTHESE, s)

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])

    response_clean, note = extract_json_block(
        response.content,
        NOTE_MARKERS["synthese"][0],
        NOTE_MARKERS["synthese"][1]
    )

    domain_notes = add_domain_note(s, "synthese", note)
    updated_state = {**s, "domain_notes": domain_notes}
    active_memory = build_active_memory(updated_state)

    return {
        "messages": [AIMessage(content=response_clean)],
        "next_agent": "fin",
        "domain_notes": domain_notes,
        "active_memory": active_memory,
        "last_agent": "synthese",
        "total_turns": s.get("total_turns", 0) + 1,
        "session_count": s.get("session_count", 0) + 1
    }