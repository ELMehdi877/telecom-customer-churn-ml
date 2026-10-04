import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. INITIALISATION DES CHEMINS ET CHARGEMENT DU JEU DE DONNÉES
# ==============================================================================

# Construction du chemin absolu dynamique vers le fichier source CSV
Base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_data = os.path.join(Base_path, "data/extraction/telecom_data.csv")

# Récupération des données brutes depuis le fichier CSV
df = pd.DataFrame(pd.read_csv(path_data))
# print(df.columns)

# ==============================================================================
# 2. INSPECTION STATISTIQUE INITIALE ET DIAGNOSTIC DES DONNÉES
# ==============================================================================

# --- ANALYSE TEXTUELLE (Console) ---

# Affichage des dimensions du DataFrame (Nombre de lignes, Nombre de colonnes)
print(df.shape)

# Diagnostic de la structure : Types de données (int, float, object) et présence de valeurs non-nulles
print("--- INFOS GENERALES ---")
print(df.info())

# Résumé des statistiques descriptives pour les colonnes numériques (moyenne, écart-type, quartiles, min/max)
print(df.describe())

# Comptage du nombre de valeurs manquantes (NaN) par colonne
print("\n--- VALEURS MANQUANTES ---")
print(df.isnull().sum())

# compter les valeurs doublons
print("\n--- DOUBLONS ---")
print(f"Nombre de doublons : {df.duplicated().sum()}")

# changer le type du colone en float au lieu de str
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# remplacer les valeurs manquant avec 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)
# Comptage du nombre de valeurs manquantes (NaN) par colonne apers le rempliçage
print("\n--- VALEURS MANQUANTES APRES REMPLACE PAR 0 0 ---")
print(df.isnull().sum())

print("--- INFOS GENERALES APRES REMPLACE PAR 0 ---")
print(df.info())

print(df.columns)


# Configuration du style des graphiques
sns.set_theme(style="whitegrid")

# -------------------------------------------------------------------
# GRAPHE 1 : Histogrammes (Distribution des variables numériques)
# Rôle : Analyser la concentration de la population (ex: nouveaux vs anciens clients).
# -------------------------------------------------------------------
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df[num_cols].hist(bins=30, figsize=(12, 4), layout=(1, 3))
plt.suptitle("Distribution des variables numériques (Histogrammes)")
plt.show()

# -------------------------------------------------------------------
# GRAPHE 2 : Boxplots (Détection des valeurs aberrantes / Outliers)
# Rôle : Identifier visuellement les valeurs extrêmes ou anormales.
# -------------------------------------------------------------------
plt.figure(figsize=(10, 4))
sns.boxplot(data=df[num_cols])
plt.title("Détection des Outliers (Boxplots)")
plt.show()

# -------------------------------------------------------------------
# GRAPHE 3 : Graphique en barres (Répartition des catégories & Churn)
# Rôle : Identifier les services/contrats les plus demandés et la proportion du Churn.
# -------------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn", hue="Contract")
plt.title("Répartition des Contrats selon le Churn (Bar Plot)")
plt.show()

# -------------------------------------------------------------------
# GRAPHE 4 : Heatmap de corrélation
# Rôle : Mesurer la force de la relation linéaire entre les colonnes numériques.
# -------------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation (Heatmap)")
plt.show()

# -------------------------------------------------------------------
# GRAPHE 5 : Analyse des services et de la facturation selon le Churn
# Rôle : Identifier quelles options ou modes de paiement poussent au Churn.
# -------------------------------------------------------------------
cat_cols = ['InternetService', 'TechSupport', 'PaymentMethod', 'PaperlessBilling']

plt.figure(figsize=(14, 8))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(2, 2, i)
    sns.countplot(data=df, x=col, hue='Churn')
    plt.title(f"Churn par rapport à {col}")
    plt.xticks(rotation=15)

plt.tight_layout()
plt.show()

