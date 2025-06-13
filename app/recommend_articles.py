# Logique :
# - Pour un utilisateur donné, on récupère tous les articles cliqués.
# - On fait la moyenne des embeddings de ces articles → profil utilisateur.
# - On compare ce profil à tous les articles non lus via la similarité cosinus.
# - On retourne les Top 5 articles les plus proches.

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, clicks_df, embeddings_df):
        self.clicks_df = clicks_df
        self.embeddings_df = embeddings_df

        # On prépare un cache rapide des articles cliqués par utilisateur
        self.user_clicks = self.clicks_df.groupby('user_id')['click_article_id'].apply(list).to_dict()

    def recommend(self, user_id, top_k=5):
        # 1. Vérification existence utilisateur
        if user_id not in self.user_clicks:
            return []

        clicked_ids = self.user_clicks[user_id]
        clicked_ids = [aid for aid in clicked_ids if aid in self.embeddings_df.index]

        if not clicked_ids:
            return []

        # 2. Calcul du profil utilisateur (moyenne des embeddings des articles cliqués)
        user_profile = self.embeddings_df.loc[clicked_ids].mean(axis=0).values.reshape(1, -1)

        # 3. Sélection des articles non lus
        candidate_ids = self.embeddings_df.index.difference(clicked_ids)
        candidate_vectors = self.embeddings_df.loc[candidate_ids]

        # 4. Similarité cosinus
        similarities = cosine_similarity(user_profile, candidate_vectors.values)[0]

        # 5. Top K recommandations
        top_indices = similarities.argsort()[-top_k:][::-1]
        recommended_ids = candidate_vectors.index[top_indices].tolist()

        return recommended_ids
