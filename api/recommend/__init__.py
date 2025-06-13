import logging
import json
import azure.functions as func
import pandas as pd
import pickle
from recommend_articles import Recommender

# ========================
# 🔁 Chargement des données à froid (1 fois au démarrage)
# ========================

# 📂 Chemin relatif vers les données (à ajuster si besoin)
DATA_DIR = "data"

# Chargement des clics utilisateur
clicks_sample = pd.read_csv(f"{DATA_DIR}/clicks_sample.csv")

# Chargement des métadonnées
metadata = pd.read_csv(f"{DATA_DIR}/articles_metadata.csv")

# Chargement des embeddings (vecteurs articles)
with open(f"{DATA_DIR}/articles_embeddings.pickle", "rb") as f:
    embeddings = pickle.load(f)

# Mise en forme des embeddings
article_embeddings_df = pd.DataFrame(
    embeddings,
    index=metadata['article_id'],
    columns=[f"embedding_{i}" for i in range(embeddings.shape[1])]
)

# Initialisation du moteur de recommandation
reco = Recommender(clicks_sample, article_embeddings_df)

# ========================
# 🌐 Fonction HTTP Trigger
# ========================

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('🔗 Requête reçue pour générer des recommandations.')

    try:
        # Récupération de l'ID utilisateur depuis les paramètres GET ou JSON
        user_id = req.params.get('user_id')
        if not user_id:
            try:
                req_body = req.get_json()
                user_id = req_body.get('user_id')
            except:
                pass

        if user_id is None:
            return func.HttpResponse(
                json.dumps({"error": "Veuillez fournir un paramètre user_id"}),
                status_code=400,
                mimetype="application/json"
            )

        # Conversion du user_id en entier
        user_id = int(user_id)

        # Génération des recommandations
        article_ids = reco.recommend(user_id)

        # Filtrage des métadonnées si disponibles
        if article_ids:
            results = metadata[metadata['article_id'].isin(article_ids)].copy()
            results['created_at'] = pd.to_datetime(results['created_at_ts'], unit='ms')
            results = results[['article_id', 'category_id', 'words_count', 'created_at']]
            results = results.sort_values(by="created_at", ascending=False)

            # Transformation en JSON
            output = results.to_dict(orient="records")

            return func.HttpResponse(
                json.dumps(output, default=str),  # date → string
                status_code=200,
                mimetype="application/json"
            )
        else:
            return func.HttpResponse(
                json.dumps({"message": "Aucune recommandation trouvée pour cet utilisateur."}),
                status_code=404,
                mimetype="application/json"
            )

    except Exception as e:
        logging.error(f"Erreur : {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
