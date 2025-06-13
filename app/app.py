import streamlit as st
import pandas as pd
import pickle
from recommend_articles import Recommender  # Moteur de recommandation local

# ================================
# 🔁 Chargement des données (avec cache pour améliorer les perfs)
# ================================

@st.cache_data
def load_data():
    # Chargement des clics utilisateurs (échantillon)
    clicks = pd.read_csv("data/clicks_sample.csv")

    # Chargement des métadonnées des articles (ID, catégorie, longueur, date…)
    metadata = pd.read_csv("data/articles_metadata.csv")

    # Chargement des embeddings vectoriels des articles (250 dimensions)
    with open("data/articles_embeddings.pickle", "rb") as f:
        embeddings = pickle.load(f)

    # On transforme les embeddings en DataFrame, indexé par article_id
    embeddings_df = pd.DataFrame(
        embeddings,
        index=metadata['article_id'],
        columns=[f'embedding_{i}' for i in range(embeddings.shape[1])]
    )

    return clicks, metadata, embeddings_df

# Appel de la fonction pour récupérer les datasets
clicks_sample, metadata, article_embeddings_df = load_data()

# ================================
# 🔧 Instanciation du moteur de recommandation
# ================================

reco = Recommender(clicks_sample, article_embeddings_df)

# ================================
# 🎨 Interface utilisateur Streamlit
# ================================

# 🔖 Barre latérale (menu d’intro)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4727/4727496.png", width=80)
st.sidebar.title("My Content")
st.sidebar.markdown("**Recommandations personnalisées**")

# 🎯 Titre principal
st.title("📚 My Content – Recommandations personnalisées")
st.markdown("Sélectionnez un utilisateur dans la liste pour obtenir les 5 articles recommandés basés sur ses lectures précédentes.")

# 👤 Sélection utilisateur
user_ids = sorted(clicks_sample['user_id'].unique())
selected_user = st.selectbox("👤 Choisissez un utilisateur :", user_ids)

# 🔎 Bouton d'action
if st.button("🔍 Voir les recommandations"):

    article_ids = reco.recommend(selected_user)

    if article_ids:
        results = metadata[metadata['article_id'].isin(article_ids)].copy()
        results['created_at'] = pd.to_datetime(results['created_at_ts'], unit='ms')
        results = results[['article_id', 'category_id', 'words_count', 'created_at']]
        results = results.sort_values(by="created_at", ascending=False)

        st.success(f"Voici les 5 articles recommandés pour l’utilisateur {selected_user}")
        st.table(results)

        # 📥 Bouton de téléchargement CSV
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les recommandations (CSV)",
            data=csv,
            file_name=f"recommandations_user_{selected_user}.csv",
            mime='text/csv'
        )

    else:
        st.warning("Aucune recommandation disponible pour cet utilisateur.")
