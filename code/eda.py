import json
import time
import random
import re
from mistralai import Mistral

def extract_json_from_text(response_text):
    """
    Recherche et extrait un objet JSON valide à partir du texte de la réponse.
    Cherche la partie entourée par des backticks, puis tente de charger le JSON.
    Retourne l'objet JSON si trouvé, sinon None.
    """
    # Rechercher le JSON entouré de backticks
    json_pattern = r'```json\n\{.*\}\n```'  # Expression régulière pour attraper un bloc JSON entre backticks
    match = re.search(json_pattern, response_text, re.DOTALL)  # re.DOTALL permet de capter les nouvelles lignes

    if match:
        json_str = match.group(0)
        # Enlever les backticks et les éventuelles nouvelles lignes autour
        json_str = json_str.strip('```json\n').strip('\n```')
        try:
            return json.loads(json_str)  # Tenter de charger le JSON
        except json.JSONDecodeError:
            print("Erreur lors du parsing du JSON extrait.")
            return None
    else:
        print("Aucun JSON valide trouvé dans la réponse.")
        return None

def get_only_chars(line):
    """
    Nettoie la ligne en retirant les caractères non alphabétiques et en normalisant la casse.
    """
    clean_line = ""
    line = line.replace("’", "").replace("'", "").replace("-", " ").replace("\t", " ").replace("\n", " ").lower()
    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '
    clean_line = re.sub(' +', ' ', clean_line).strip()
    return clean_line

async def call_mistral(prompt, sentence):
    """
    Effectue un appel API vers Mistral pour générer du texte en fonction du prompt et de la phrase.
    Mistral est responsable de générer les variations de la phrase ou autres modifications demandées.
    """
    API_KEY = "your-mistralai-api-key"  # Ta clé API ici
    MODEL_NAME = "ministral-8b-2410"  # Remplacer par le modèle spécifique que tu veux utiliser

    # Ajouter un délai de 1.5 seconde
    time.sleep(1.5)

    # Initialisation du client Mistral avec la clé API et le modèle
    client = Mistral(api_key=API_KEY)

    try:
        # Appel à Mistral pour générer du texte en fonction du prompt et de la phrase
        chat_response = await client.chat.complete_async(
            model=MODEL_NAME,
            temperature=0.3,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": prompt,
            }]
        )


        # Récupérer la réponse du texte généré
        response_text = chat_response.choices[0].message.content  # Assurez-vous que "generated_text" est le bon champ dans la réponse

        # Tenter d'extraire et de parser le JSON à partir du texte
        response_json = extract_json_from_text(response_text)

        return response_json['sentences']  # Retourner la réponse sous forme de dictionnaire

    except Exception as e:
        print(f"Erreur lors de l'appel à Mistral : {e}")
        return None

async def synonym_replacement(sentence, n):
    """
    Mistral génère une augmentation de données supervisées en remplaçant `n` mots par des synonymes.
    La tâche de génération des synonymes est à la charge du modèle LLM.
    """
    prompt = f"""
    Effectue une augmentation de données en remplaçant exactement {n} mots de la phrase suivante par des synonymes.
    Les synonymes doivent être choisis de manière à préserver le sens original de la phrase, sans introduire d'ambiguïté.
    Les mots remplacés doivent être courants et bien compris, en évitant les termes trop techniques ou peu usités.
    Retourne les résultats sous forme de JSON. Exemple de réponse attendu :
    {{"sentences": "phrase modifiée"}}

    Voici la phrase à modifier : {sentence}
    """
    return await call_mistral(prompt, sentence)

async def random_deletion(sentence, p):
    """
    Mistral supprime certains mots de la phrase avec une probabilité de `p`, tout en préservant le sens.
    La tâche de suppression est effectuée par le modèle LLM.
    """
    prompt = f"""
    Effectue une augmentation de données en supprimant certains mots de la phrase suivante avec une probabilité de {p}.
    Les mots supprimés doivent être choisis de manière à ne pas altérer le sens global de la phrase.
    Évite de supprimer des mots essentiels comme des verbes, des noms importants, ou des mots qui sont cruciaux pour la structure grammaticale de la phrase.
    Retourne les résultats sous forme de JSON. Exemple de réponse attendu :
    {{"sentences": "phrase modifiée"}}

    Voici la phrase à modifier : {sentence}
    """
    return await call_mistral(prompt, sentence)

async def random_swap(sentence, n):
    """
    Mistral échange `n` mots dans la phrase tout en préservant le sens global.
    La tâche d'échange de mots est effectuée par le modèle LLM.
    """
    prompt = f"""
    Effectue une augmentation de données en échangeant {n} mots dans la phrase suivante tout en préservant le sens global.
    Assure-toi que les mots échangés sont de nature similaire, sans changer la signification de la phrase.
    Évite de modifier l'ordre des mots qui affectent le sens principal de la phrase.
    Retourne les résultats sous forme de JSON. Exemple de réponse attendu :
    {{"sentences":"phrase modifiée"}}

    Voici la phrase à modifier : {sentence}
    """
    return await call_mistral(prompt, sentence)

async def random_insertion(sentence, n):
    """
    Mistral insère `n` mots contextuellement pertinents dans la phrase tout en préservant son sens.
    La tâche d'insertion est effectuée par le modèle LLM.
    """
    prompt = f"""
    Effectue une augmentation de données en insérant {n} mots contextuellement pertinents dans la phrase suivante.
    Les nouveaux mots doivent être insérés sans altérer le sens de la phrase et doivent être naturellement intégrés.
    Assure-toi que la phrase reste fluide et compréhensible après l'insertion des mots.
    Retourne les résultats sous forme de JSON. Exemple de réponse attendu :
    {{"sentences": "phrase modifiée"}}

    Voici la phrase à modifier : {sentence}
    """
    return await call_mistral(prompt, sentence)

async def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    """
    Effectue une augmentation de données supervisées avec plusieurs techniques, en appelant Mistral pour chaque tâche.
    Mistral est responsable de l'application des différentes techniques d'augmentation des données.
    """
    sentence = get_only_chars(sentence)
    augmented_sentences = []
    num_new_per_technique = (num_aug // 4) + 1

    if alpha_sr > 0:
        for _ in range(num_new_per_technique):
            result = await synonym_replacement(sentence, max(1, int(alpha_sr * len(sentence.split()))))
            if result is not  None:
              augmented_sentences.append(result)  # Récupérer les phrases du JSON

    if alpha_ri > 0:
        for _ in range(num_new_per_technique):
            result = await random_insertion(sentence, max(1, int(alpha_ri * len(sentence.split()))))
            if result is not  None:
              augmented_sentences.append(result)  # Récupérer les phrases du JSON

    if alpha_rs > 0:
        for _ in range(num_new_per_technique):
            result = await random_swap(sentence, max(1, int(alpha_rs * len(sentence.split()))))
            if result is not  None:
              augmented_sentences.append(result)  # Récupérer les phrases du JSON

    if p_rd > 0:
        for _ in range(num_new_per_technique):
            result = await random_deletion(sentence, p_rd)
            if result is not None:
              augmented_sentences.append(result)  # Récupérer les phrases du JSON
    # si augmented sentences est list
    if isinstance(augmented_sentences, list):
        print("augmente list",augmented_sentences )
        augmented_sentences = [s for s in augmented_sentences if not s.isspace()]
    if isinstance(augmented_sentences, str):
        augmented_sentences = [augmented_sentences]

    augmented_sentences.append(sentence)
    return augmented_sentences
