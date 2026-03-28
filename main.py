from langchain_core.messages import HumanMessage
from graph import create_graph

def main():
    print("\n" + "=" * 60)
    print("  🧠 ASSISTANT PERSONNEL MULTI-AGENTS")
    print("=" * 60)

    # Identification de l'utilisateur
    print("\n👤 Qui utilise le système ?")
    print("   1 → Simon")
    print("   2 → Mallorie")
    choix = input("\nTon choix (1 ou 2) : ").strip()

    if choix == "1":
        thread_id = "user_simon"
        prenom = "Simon"
    else:
        thread_id = "user_mallorie"
        prenom = "Mallorie"

    print(f"\n✅ Session ouverte pour : {prenom}")
    print("💡 Tape 'quitter' à tout moment pour terminer.\n")
    print("=" * 60)

    # Création du graphe (DOIT être avant toute utilisation de graph)
    graph = create_graph()

    # thread_id unique par utilisateur → isolation des données
    config = {"configurable": {"thread_id": thread_id}}

    # -------------------------------------------------------
    # Détection nouvel utilisateur → auto-démarrage assessment
    # -------------------------------------------------------
    try:
        etat_actuel = graph.get_state(config)
        est_nouveau = len(etat_actuel.values.get("messages", [])) == 0
    except:
        est_nouveau = True

    if est_nouveau:
        print("\n🆕 Première utilisation détectée.")
        print("⏳ Démarrage de l'assessment initial...\n")
        result = graph.invoke(
            {"messages": [HumanMessage(content="Bonjour, je commence.")], "next_agent": ""},
            config=config
        )
        premiere_reponse = result["messages"][-1].content
        print(f"🤖 Assistant :\n\n{premiere_reponse}")
        print(f"\n{'─' * 60}\n")
    else:
        print(f"\n🤖 Bon retour {prenom} ! Que souhaitez-vous explorer aujourd'hui ?\n")

    # -------------------------------------------------------
    # Boucle de conversation principale
    # -------------------------------------------------------
    while True:
        user_input = input("Vous : ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quitter", "quit", "exit", "bye", "au revoir"]:
            print("\n👋 Session terminée. Vos données sont sauvegardées. À bientôt !")
            break

        print("\n⏳ Traitement en cours...\n")

        try:
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "next_agent": "",
                },
                config=config
            )
            derniere_reponse = result["messages"][-1].content
            print(f"🤖 Assistant :\n\n{derniere_reponse}")
            print(f"\n{'─' * 60}\n")

        except Exception as e:
            print(f"⚠  Une erreur s'est produite : {e}")
            print("Veuillez réessayer.\n")

if __name__ == "__main__":
    main()