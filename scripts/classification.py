import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
print("============================ start ============================")

# --- 1. Chargement des données prétraitées ---
# Récupération du dossier racine du projet (deux niveaux au-dessus du fichier actuel)
Base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construction des chemins absolus vers les fichiers CSV de données
path_clusters = os.path.join(Base_path, "data/clusters/data_clusters.csv")
path_nettoyage = os.path.join(Base_path, "data/nettoyage/data_nettoyage.csv")

# Lecture des fichiers CSV dans des DataFrames Pandas
df_clusters = pd.read_csv(path_clusters)
df_nettoyage = pd.read_csv(path_nettoyage)

# Ajout de la colonne cible 'Churn' issue du fichier nettoyé dans le dataframe des clusters
df_clusters["Churn"] = df_nettoyage["Churn"]

# --- 2. Séparation des caractéristiques (X) et de la cible (y) ---
# X contient toutes les variables explicatives (features) sauf la cible 'Churn'
X = df_clusters.drop(columns=['Churn'])

# y contient uniquement la variable cible à prédire (0 = Fidele, 1 = Churn)
y = df_clusters['Churn'] 

# --- 3. Découpage en jeu d'entraînement (Train) et jeu de test (Test) ---
# test_size=0.2 : 80% des données pour l'entraînement (X_train), 20% pour le test (X_test)
# stratify=y    : Conserve exactement la même proportion de 'Churn' dans Train et Test pour éviter tout biais
# random_state=42 : Fixe la graine d'aléatoire pour garantir des résultats 100% reproductibles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- 4. Normalisation / Standardisation des données ---
# Instanciation du StandardScaler (met la moyenne à 0 et l'écart-type à 1)
scaler = StandardScaler()

# Sur le Train : fit (calcule la moyenne et l'écart-type de X_train) 
# ET transform (applique la formule sur X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Vérification des dimensions des données ---
print(f"Taille de X_train : {X_train_scaled.shape}")
print(f"Taille de X_test : {X_test_scaled.shape}")
print("\nTâche 1 terminée avec succès !")


print("============================ utilisation de class_weight ============================")
# --- 1. Calcul du ratio de déséquilibre des classes ---
# XGBoost gère le déséquilibre différemment de Scikit-Learn :
# Il utilise le paramètre 'scale_pos_weight' qui correspond à : (Nombre de classe 0) / (Nombre de classe 1)
# Ce ratio indique à XGBoost d'accorder plus de poids aux erreurs commises sur la classe minoritaire (Churn)
ratio = (y_train == 0).sum() / (y_train == 1).sum()

# --- 2. Dictionnaire d'initialisation des modèles ---
# On regroupe différents algorithmes pour pouvoir les entraîner et les comparer facilement
models = {
    # Régression Logistique : Modèle linéaire simple, rapide et facile à interpréter.
    # class_weight="balanced" ajuste automatiquement les poids inversement proportionnels aux fréquences des classes.
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", 
        random_state=42
    ),

    # Arbre de Décision : Sépare les données par des règles de décisions successives.
    # Sensible au surapprentissage (overfitting), mais très lisible.
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),

    # Forêt Aléatoire : Ensemble d'arbres de décision qui votent ensemble.
    # Réduit fortement le surapprentissage par rapport à un arbre seul et offre d'excellentes performances.
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),

    # Support Vector Machine (SVM) : Cherche l'hyperplan optimal qui sépare les classes.
    # probability=True est indispensable si l'on veut calculer des probabilités et la courbe ROC-AUC.
    "SVM": CalibratedClassifierCV(SVC(class_weight="balanced",  random_state=42), ensemble=False),

    # XGBoost : Algorithme puissant basé sur le Gradient Boosting (entraînement d'arbres en série qui corrigent les erreurs des précédents).
    # scale_pos_weight=ratio applique le facteur de pondération calculé plus haut pour la classe 1.
    "XGBoost": XGBClassifier(scale_pos_weight=ratio, random_state=42)
}


results = []

# Entraînement et évaluation boucle
for name, model in models.items():

    # 1. Entraînement
    model.fit(X_train_scaled, y_train)

    # 2. Prédictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 3. Calcul des métriques
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    # On extrait les métriques spécifiquement pour la classe Churn (1)
    results.append({
        "Model": name,
        "Recall (Churn = 1)": round(report["1"]["recall"], 3),
        "Precision  (Churn = 1)": round(report["1"]["precision"], 3),
        "F1-Score (Churn = 1)": round(report["1"]["f1-score"], 3),
        "ROC-AUC": round(auc, 3)
    })

# 4. Affichage du tableau comparatif
df_results = pd.DataFrame(results).sort_values(by='Recall (Churn = 1)', ascending=False)
print(df_results.to_string(index=False))

# # 1. Sélection et ré-entraînement du meilleur modèle (Logistic Regression)
# best_model = models["Logistic Regression"]
# best_model.fit(X_train_scaled, y_train)

# # 2. Affichage de la Matrice de Confusion
# y_pred = best_model.predict(X_test_scaled)
# ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")

# # ConfusionMatrixDisplay.from_estimator(models["Logistic Regression"], X_test_scaled, y_test)

# plt.title("Matrice de Confusion - Logistic Regression")
# plt.show()




# # 3. Extraction de l'importance des variables (Coefficients)
# coefficients = pd.DataFrame({
#     'Feature': X.columns,
#     'Coefficient': best_model.coef_[0]
# }).sort_values(by='Coefficient', key=abs, ascending=False)

# # Visualisation des coefficients
# plt.figure(figsize=(10, 6))
# sns.barplot(data=coefficients.head(10), x='Coefficient', y='Feature', palette='vlag')
# plt.title("Top 10 des variables influençant le Churn")
# plt.xlabel("Coefficient (Positif = augmente le Churn, Négatif = réduit le Churn)")
# plt.show()

print("============================ utilisation de smote ============================")
# 1. Application de SMOTE sur le Train set uniquement
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# 2. Définition des modèles (sans class_weight)
models_smote = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
    "XGBoost": XGBClassifier(random_state=42)
}

# 3. Entraînement et comparaison
results_smote = []
for name, model in models_smote.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    results_smote.append({
        "Modele": name,
        "Recall (Churn = 1)": round(report["1"]["recall"], 3),
        "Precision (Churn = 1)": round(report["1"]["precision"], 3),
        "F1-Score (Churn = 1)": round(report["1"]["f1-score"], 3),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 3)
    })

print(pd.DataFrame(results_smote).sort_values(by="Recall (Churn = 1)", ascending=False).to_string(index=False))

# # 1. Sélection et ré-entraînement du meilleur modèle (Logistic Regression)
# best_model = models_smote["Logistic Regression"]
# best_model.fit(X_train_res, y_train_res)

# # 2. Affichage de la Matrice de Confusion
# y_pred_res = best_model.predict(X_test_scaled)
# ConfusionMatrixDisplay.from_predictions(y_test, y_pred_res, cmap="Blues")

# # ConfusionMatrixDisplay.from_estimator(models["Logistic Regression"], X_test_scaled, y_test)

# plt.title("Matrice de Confusion - Logistic Regression")
# plt.show()


# import matplotlib.pyplot as plt
# from sklearn.metrics import ConfusionMatrixDisplay

# Création d'une figure avec 2 sous-graphiques (1 ligne, 2 colonnes)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ==========================================
# 1. MODÈLE SANS SMOTE (À gauche)
# ==========================================
best_model = models["Logistic Regression"]
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_train_scaled)

# On dessine la matrice sur le 1er sous-graphique (ax=axes[0])
ConfusionMatrixDisplay.from_predictions(
    y_train, 
    y_pred, 
    cmap="Blues", 
    ax=axes[0], 
    colorbar=False
)
axes[0].set_title("Logistic Regression (SANS SMOTE)")

# ==========================================
# 2. MODÈLE AVEC SMOTE (À droite)
# ==========================================
best_model_smote = models_smote["Logistic Regression"]
best_model_smote.fit(X_train_res, y_train_res)
y_pred_res = best_model_smote.predict(X_train_res)

# On dessine la matrice sur le 2ème sous-graphique (ax=axes[1])
ConfusionMatrixDisplay.from_predictions(
    y_train_res, 
    y_pred_res, 
    cmap="Blues", 
    ax=axes[1], 
    colorbar=False
)
axes[1].set_title("Logistic Regression (AVEC SMOTE)")

# Ajustement automatique de l'espace et affichage
plt.tight_layout()
plt.show()

print("Entraînement (Équilibré par SMOTE) :\n", y_train_res.value_counts())
print("\nTest (Réalité intacte) :\n", y_test.value_counts())