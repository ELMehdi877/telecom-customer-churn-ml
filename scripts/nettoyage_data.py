import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Gestion des chemins dynamiques du projet
Base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_data = os.path.join(Base_path, "data/extraction/telecom_data.csv")

# Création du dossier de destination s'il n'existe pas
data_nettoyage = "data/nettoyage"
os.makedirs(data_nettoyage, exist_ok=True)

# 2. Chargement du dataset
df = pd.read_csv(path_data)

# 3. Traitement de TotalCharges (conversion en float et gestion des NaN)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 4. Suppression de l'identifiant inutile
df = df.drop(columns=["customerID"])

# =====================================================================
# FEATURE ENGINEERING (Création de nouvelles colonnes simples)
# =====================================================================

# Variable 3 : Tranches d'ancienneté (Nouveau: 0-12 mois, Moyen: 12-48 mois, Fidèle: >48 mois)
df["Tenure_Group"] = pd.cut(df["tenure"], bins=[-1, 12, 48, 100], labels=["Nouveau", "Moyen", "Fidele"])
# Supprimer tenure pour éviter la redondance avec Tenure_Group
# df = df.drop(columns=["tenure"])

# =====================================================================

# 5. Encodage manuel de la cible (Churn : Yes -> 1, No -> 0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 6. Normalisation / Standardisation des variables numériques
scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df[num_cols] = scaler.fit_transform(df[num_cols])

# 7. Encodage One-Hot des variables catégorielles (texte -> 0 et 1)
df_clean = pd.get_dummies(df, drop_first=True, dtype=int)

# 8. Sauvegarde du fichier nettoyé
path_nettoyage = os.path.join(data_nettoyage, "data_nettoyage.csv")
df_clean.to_csv(path_nettoyage, index=False)

print(df_clean.columns)

print("Nettoyage réussi et fichier sauvegardé avec succès !")
