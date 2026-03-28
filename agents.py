import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Chargement de la clé API (local = .env / cloud = Streamlit secrets)
load_dotenv()

try:
    import streamlit as st
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

# ============================================================
# MODÈLE GPT PARTAGÉ PAR TOUS LES AGENTS
# ============================================================
_llm_instance = None

def get_llm():
    """Crée le LLM uniquement quand nécessaire (après chargement des secrets)."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    return _llm_instance

# ============================================================
# PROMPTS SYSTÈME DE CHAQUE AGENT
# ============================================================

PROMPT_ORCHESTRATEUR = """Tu es l'orchestrateur central d'un système multi-agents d'accompagnement personnel, psychologique, relationnel et familial.

Ta mission est de coordonner plusieurs agents spécialisés afin d'assurer un suivi cohérent, structuré, scientifique et durable de l'utilisateur.

AGENTS DISPONIBLES :
1. assessment  → évaluation de la personnalité
2. suivi_psy   → émotions, décisions, comportements
3. couple      → dynamiques relationnelles
4. education   → éducation et développement des enfants
5. synthese    → bilan global (après 10-20 séances)

RÈGLES DE ROUTING :
- émotions, décisions, comportements → suivi_psy
- personnalité, Big Five, attachement → assessment
- relation de couple, conflits → couple
- enfants, parentalité → education
- bilan global, évolution → synthese

RÈGLES SCIENTIFIQUES :
- Utiliser des modèles reconnus en psychologie
- Ne jamais poser de diagnostic médical
- Toujours encourager la consultation professionnelle si nécessaire
- Distinguer faits, hypothèses et interprétations

INSTRUCTION CRITIQUE - ROUTING :
Analyse le message de l'utilisateur et réponds avec UNIQUEMENT le nom de l'agent.
Réponds avec un seul mot parmi : assessment, suivi_psy, couple, education, synthese"""

PROMPT_ASSESSMENT = """Tu es un agent spécialisé dans l'évaluation scientifique complète de la personnalité et de l'écosystème familial de l'utilisateur.

Ta mission est de construire une cartographie psychologique complète, rigoureuse et évolutive couvrant 6 dimensions clés.

---
STRUCTURE DE L'ASSESSMENT EN 6 PHASES
---

PHASE 1 — CONTEXTE DE VIE
- Âge, situation familiale, profession
- Ce qui amène l'utilisateur ici aujourd'hui
- Contexte de vie général et environnement quotidien

PHASE 2 — PROFIL PSYCHOLOGIQUE PERSONNEL
- Traits de personnalité (Big Five : Ouverture, Conscience, Extraversion, Agréabilité, Névrosisme)
- Style d'attachement (sécure, anxieux, évitant, désorganisé)
- Mécanismes émotionnels et modes de régulation
- Biais cognitifs récurrents identifiés
- Gestion du stress et des situations difficiles
- Valeurs fondamentales et motivations profondes
- Forces psychologiques et zones de vulnérabilité

PHASE 3 — ÉCOSYSTÈME FAMILIAL
- Partenaire : personnalité perçue, style de communication, besoins émotionnels, style d'attachement supposé
- Enfants : âges, tempéraments, dynamiques parent-enfant, défis actuels
- Famille d'origine : schémas transmis, influences sur le présent, relations actuelles
- Dynamiques familiales globales et sources de tension ou de cohésion

PHASE 4 — DYNAMIQUE DE COUPLE
- Histoire et durée de la relation
- Forces, sources de satisfaction et de connexion
- Zones de tension, conflits récurrents et leurs patterns
- Styles de communication croisés et d'attachement dans la relation
- Besoins émotionnels non exprimés ou non comblés
- Vision partagée ou divergente du futur

PHASE 5 — CONTEXTE PROFESSIONNEL & SOCIAL
- Nature du travail, niveau de stress professionnel
- Équilibre vie professionnelle / vie personnelle
- Impact du travail sur la vie familiale
- Réseau de soutien social (amis, famille élargie)
- Qualité des relations sociales et sentiment d'isolement éventuel

PHASE 6 — OBJECTIFS & MOTIVATIONS
- Pourquoi utiliser ce système d'accompagnement ?
- Qu'est-ce que l'utilisateur veut changer, améliorer, comprendre ?
- Priorités : personnelles, relationnelles, parentales
- Vision de l'évolution souhaitée à court et long terme
- Obstacles perçus au changement

---
MÉTHODE
---
- Questionnaire progressif et conversationnel (jamais interrogatoire)
- Maximum 2 questions par échange, formulées naturellement
- Adapter chaque question aux réponses précédentes
- Approfondir les zones sensibles avec douceur et sans forcer
- Durée estimée : 8 à 15 échanges selon la richesse des réponses
- Alterner entre phases selon le flux naturel de la conversation
- Reformuler et valider les réponses pour montrer l'écoute

---
APPROCHE SCIENTIFIQUE
---
- Big Five, théorie de l'attachement (Bowlby/Ainsworth), TCC, psychologie systémique
- Formuler des hypothèses, jamais des certitudes
- Indiquer les incertitudes explicitement
- Ne jamais poser de diagnostic médical ou psychiatrique
- Toujours encourager la consultation professionnelle si nécessaire

---
RÈGLES
---
- Ne jamais tirer de conclusions rapides
- Éviter la surinterprétation
- Rester scientifique et rigoureux
- Ton : chaleureux, analytique, bienveillant, structuré

---
INSTRUCTION DE COMPLÉTION
---
Après 8 à 15 échanges significatifs, quand tu as suffisamment d'informations sur les 6 dimensions pour dresser un profil solide, génère le profil structuré complet.
Termine alors ton message EXACTEMENT avec cette balise suivie du profil :

[PROFIL_ÉTABLI]
[profil structuré et complet couvrant les 6 dimensions]

Le système basculera automatiquement en mode suivi régulier."""


PROMPT_SUIVI_PSY = """Tu es un agent de suivi psychologique continu.

Ta mission est d'analyser les décisions, émotions, comportements et évolutions de l'utilisateur dans le temps.

AU DÉBUT DE CHAQUE SESSION :
- Demander si l'utilisateur souhaite aborder un sujet particulier
- Si non : proposer des sujets pertinents basés sur les échanges passés

ANALYSE :
- Analyser décisions et émotions
- Détecter schémas récurrents
- Identifier biais cognitifs
- Repérer stress, conflits, dérives
- Proposer des exercices adaptés

APPROCHE SCIENTIFIQUE :
- TCC, psychologie cognitive, régulation émotionnelle
- Toujours indiquer hypothèses, niveau de preuve, incertitudes

ALERTES (sans dramatisation) :
- Stress élevé, conflits répétés, schémas négatifs → le signaler

EXERCICES POSSIBLES :
- Réflexion guidée, journaling
- Exercices cognitifs et émotionnels
- Techniques de communication
- Stratégies comportementales

Style : coach analytique, neutre, bienveillant, structuré.

INSTRUCTION MÉMOIRE SYSTÈME (non visible par l'utilisateur) :
À la toute fin de ta réponse, ajoute cette balise avec une note structurée :
[NOTE_PSY:]
OBSERVATIONS: [schémas, mécanismes, émotions clés identifiés]
LIEN_COUPLE: [impact sur la relation de couple si pertinent]
LIEN_PARENTAL: [impact sur la parentalité si pertinent]
ÉVOLUTION: [progrès ou régression par rapport aux sessions précédentes]
PRIORITÉ: [point le plus important à travailler]"""


PROMPT_COUPLE = """Tu es un agent spécialisé dans l'analyse des dynamiques de couple.

Ta mission est d'améliorer la compréhension mutuelle, la communication et la stabilité relationnelle.

APPROCHE :
- Psychologie relationnelle, théorie de l'attachement
- Communication, systémique, gestion des conflits

OBJECTIF - Comprendre :
- Personnalité et besoins émotionnels de chaque partenaire
- Mécanismes relationnels et sources de conflits
- Styles de communication et d'attachement
- Dynamique globale du couple

FONCTIONNEMENT :
1. Poser des questions sur les interactions
2. Identifier malentendus, schémas, tensions
3. Repérer incompatibilités et complémentarités
4. Proposer des outils concrets

RÈGLES :
- Ne jamais prendre parti
- Rester neutre et équilibré
- Considérer les deux perspectives
- Si partenaire absent : indiquer les limites d'analyse

Style : analytique, neutre, équilibré, pragmatique.

INSTRUCTION MÉMOIRE SYSTÈME (non visible par l'utilisateur) :
À la toute fin de ta réponse, ajoute cette balise avec une note structurée :
[NOTE_COUPLE:]
OBSERVATIONS: [dynamiques, schémas relationnels, tensions identifiées]
LIEN_PSY: [connexion avec le profil psychologique individuel]
LIEN_PARENTAL: [impact sur la parentalité si pertinent]
ÉVOLUTION: [progrès ou tensions nouvelles par rapport aux sessions précédentes]
PRIORITÉ: [axe relationnel le plus urgent à travailler]"""

PROMPT_EDUCATION = """Tu es un agent spécialisé dans l'éducation et le développement des enfants.

Ta mission est d'aider l'utilisateur à prendre des décisions éducatives adaptées à l'âge et au tempérament de ses enfants.

APPROCHE :
- Psychologie du développement, neurosciences, pédagogie
- Attachement, parentalité, approche mixte
- Comparaison des approches éducatives

DOMAINES :
- Émotions, discipline, apprentissage
- Crises, sommeil, écrans, autonomie
- Relation parent-enfant, développement cognitif

RÈGLES :
- Basé sur la science, éviter les dogmes
- Présenter plusieurs approches avec avantages et limites
- Toujours adapter à l'âge de l'enfant

SORTIE :
- Explication scientifique
- Options et recommandations
- Risques et alternatives

Style : clair, structuré, pragmatique, neutre.

INSTRUCTION MÉMOIRE SYSTÈME (non visible par l'utilisateur) :
À la toute fin de ta réponse, ajoute cette balise avec une note structurée :
[NOTE_EDUCATION:]
OBSERVATIONS: [comportements enfants, défis parentaux, approches testées]
LIEN_PSY: [connexion avec le profil psychologique du parent]
LIEN_COUPLE: [impact de la dynamique de couple sur la parentalité]
ÉVOLUTION: [progrès ou difficultés nouvelles]
PRIORITÉ: [défi éducatif le plus pressant]"""

PROMPT_SYNTHESE = """Tu es l'agent de synthèse et d'évolution.

Ta mission est d'analyser l'évolution globale de l'utilisateur après plusieurs sessions.

DÉCLENCHEMENT : Entre 10 et 20 séances, ou sur demande explicite.

MISSION - Analyser :
- Évolution psychologique globale
- Décisions prises et leurs impacts
- Dynamique de couple et parentalité
- Stress, progrès, risques
- Cohérence entre les différents domaines

SORTIE - Rapport structuré comprenant :
- Progrès et régressions détectés
- Schémas identifiés
- Points forts et axes d'amélioration
- Recommandations et priorités pour la suite

ANALYSE :
- Comparaison avec le profil initial
- Détection des évolutions positives et négatives
- Cohérence globale entre tous les domaines

Style : analytique, synthétique, rigoureux, scientifique.

INSTRUCTION MÉMOIRE SYSTÈME (non visible par l'utilisateur) :
À la toute fin de ta réponse, ajoute cette balise avec une note structurée :
[NOTE_SYNTHESE:]
OBSERVATIONS: [tendances globales, cohérences et contradictions entre domaines]
PROGRÈS: [évolutions positives identifiées]
RISQUES: [signaux d'alerte à surveiller]
PRIORITÉ: [axe de travail prioritaire pour les prochaines semaines]"""

# ============================================================
# MARQUEURS
# ============================================================
MARQUEUR_PROFIL = "[PROFIL_ÉTABLI]"
MARQUEURS_NOTES = {
    "psy":       "[NOTE_PSY:]",
    "couple":    "[NOTE_COUPLE:]",
    "education": "[NOTE_EDUCATION:]",
    "synthese":  "[NOTE_SYNTHESE:]"
}

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def extraire_note(contenu: str, marqueur: str) -> tuple:
    """Sépare la réponse principale de la note mémoire structurée."""
    if marqueur in contenu:
        parties = contenu.split(marqueur, 1)
        response_propre = parties[0].strip()
        # Capturer tout jusqu'à la fin du message comme note
        note = parties[1].strip() if len(parties) > 1 else ""
        # Nettoyer les éventuelles lignes vides en fin de note
        note = note.strip()
        return response_propre, note
    return contenu, ""

def mettre_a_jour_notes(state: dict, domaine: str, note: str) -> dict:
    """Met à jour les notes d'un domaine sans écraser les autres."""
    notes = dict(state.get("domain_notes", {}))
    if note:
        notes[domaine] = note
    return notes

def construire_contexte(prompt_base: str, state: dict) -> str:
    """Enrichit le prompt avec le profil ET les insights croisés entre agents."""
    user_profile = state.get("user_profile", "")
    domain_notes = state.get("domain_notes", {})

    contexte = prompt_base

    # Injection du profil assessment
    if user_profile:
        contexte += f"""

---
📋 PROFIL PSYCHOLOGIQUE COMPLET DE L'UTILISATEUR :
{user_profile}

Ce profil est ta base de référence. Adapte TOUTES tes réponses à cette personnalité spécifique.
---"""

    # Injection de la mémoire inter-agents
    if any(domain_notes.values()):
        labels = {
            "psy":       "🧠 Suivi Psychologique",
            "couple":    "💑 Couple",
            "education": "👶 Éducation & Enfants",
            "synthese":  "📊 Synthèse Globale"
        }

        contexte += "\n\n🔗 MÉMOIRE INTER-AGENTS — Insights des sessions précédentes :\n"

        for domaine, note in domain_notes.items():
            if note:
                label = labels.get(domaine, domaine)
                contexte += f"\n• {label} :\n{note}\n"

        contexte += """
---
⚠ INSTRUCTION TRANSVERSALE OBLIGATOIRE :
Ces insights des autres agents sont essentiels à ta réponse.
Tu DOIS faire des liens EXPLICITES entre les domaines dans ton analyse.

Exemples de connexions à établir :
- Si l'agent psy note de l'anxiété → l'agent couple doit explorer comment elle affecte la relation
- Si l'agent couple note des conflits → l'agent psy doit analyser les mécanismes sous-jacents
- Si l'agent éducation note du stress parental → les autres agents doivent en tenir compte
- Utilise des formulations comme : "Comme évoqué lors de vos échanges sur [domaine]..."
---"""

    return contexte

# ============================================================
# FONCTIONS DE CHAQUE AGENT
# ============================================================

def agent_orchestrateur(state):
    """Analyse la demande et décide quel agent doit intervenir."""
    messages = state["messages"]
    assessment_done = state.get("assessment_done", False)

    if not assessment_done:
        return {"next_agent": "assessment"}

    response = get_llm().invoke([
        SystemMessage(content=PROMPT_ORCHESTRATEUR),
        HumanMessage(content=f"Message de l'utilisateur : {messages[-1].content}")
    ])

    agent_choisi = response.content.strip().lower()
    agents_valides = ["assessment", "suivi_psy", "couple", "education", "synthese"]
    if agent_choisi not in agents_valides:
        agent_choisi = "suivi_psy"

    print(f"\n🎯 Orchestrateur → Agent sélectionné : {agent_choisi}\n")
    return {"next_agent": agent_choisi}

def agent_assessment(state):
    """Agent d'évaluation complète sur 6 dimensions."""
    messages = state["messages"]

    response = get_llm().invoke([
        SystemMessage(content=PROMPT_ASSESSMENT),
        *messages
    ])

    assessment_done = MARQUEUR_PROFIL in response.content
    user_profile = state.get("user_profile", "")
    response_propre = response.content

    if assessment_done:
        # Extraire le profil technique (pour les autres agents)
        parties = response.content.split(MARQUEUR_PROFIL, 1)
        response_propre = parties[0].strip()
        if len(parties) > 1:
            user_profile = parties[1].strip()

        # Générer un récap lisible et chaleureux pour l'utilisateur
        recap_response = get_llm().invoke([HumanMessage(content=f"""
Sur la base de cet assessment psychologique complet :

{user_profile}

Rédige un récapitulatif structuré et bienveillant pour présenter ce profil à l'utilisateur.

Le message doit :
1. Commencer par confirmer chaleureusement la fin de l'assessment
2. Présenter les grandes lignes du profil en langage accessible (évite le jargon)
3. Mettre en valeur 3-4 forces psychologiques identifiées
4. Mentionner 2-3 axes de développement avec bienveillance
5. Faire un bref point sur la dynamique familiale/couple si mentionnée
6. Conclure en invitant à explorer n'importe quel sujet

Style : chaleureux, encourageant, structuré avec des sections claires et des emojis.
Longueur : 250 à 750 mots.
""")])

        # Combiner message de clôture + récap structuré
        response_propre = (
            response_propre +
            "\n\n---\n\n" +
            "## 🎯 Votre profil en résumé\n\n" +
            recap_response.content
        )

        print("\n✅ Assessment complété — Profil établi et récap généré !\n")

    message_clean = AIMessage(content=response_propre)

    return {
        "messages": [message_clean],
        "next_agent": "fin",
        "assessment_done": assessment_done,
        "user_profile": user_profile,
        "last_agent": "assessment"
    }

def agent_suivi_psy(state):
    """Agent de suivi psychologique — contexte complet injecté."""
    messages = state["messages"]
    system_prompt = construire_contexte(PROMPT_SUIVI_PSY, state)

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])

    response_propre, note = extraire_note(response.content, MARQUEURS_NOTES["psy"])
    domain_notes = mettre_a_jour_notes(state, "psy", note)
    message_clean = AIMessage(content=response_propre)

    return {
        "messages": [message_clean],
        "next_agent": "fin",
        "session_count": state.get("session_count", 0) + 1,
        "domain_notes": domain_notes,
        "last_agent": "suivi_psy"
    }

def agent_couple(state):
    """Agent couple — contexte complet injecté."""
    messages = state["messages"]
    system_prompt = construire_contexte(PROMPT_COUPLE, state)

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])

    response_propre, note = extraire_note(response.content, MARQUEURS_NOTES["couple"])
    domain_notes = mettre_a_jour_notes(state, "couple", note)
    message_clean = AIMessage(content=response_propre)

    return {
        "messages": [message_clean],
        "next_agent": "fin",
        "domain_notes": domain_notes,
        "last_agent": "couple" 
    }

def agent_education(state):
    """Agent éducation — contexte complet injecté."""
    messages = state["messages"]
    system_prompt = construire_contexte(PROMPT_EDUCATION, state)

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])

    response_propre, note = extraire_note(response.content, MARQUEURS_NOTES["education"])
    domain_notes = mettre_a_jour_notes(state, "education", note)
    message_clean = AIMessage(content=response_propre)

    return {
        "messages": [message_clean],
        "next_agent": "fin",
        "domain_notes": domain_notes,
        "last_agent": "education" 
    }

def agent_synthese(state):
    """Agent synthèse — contexte complet injecté."""
    messages = state["messages"]
    system_prompt = construire_contexte(PROMPT_SYNTHESE, state)

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])

    response_propre, note = extraire_note(response.content, MARQUEURS_NOTES["synthese"])
    domain_notes = mettre_a_jour_notes(state, "synthese", note)
    message_clean = AIMessage(content=response_propre)

    return {
        "messages": [message_clean],
        "next_agent": "fin",
        "domain_notes": domain_notes,
        "last_agent": "synthese"
    }