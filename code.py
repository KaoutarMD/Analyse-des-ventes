import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from pandas import to_datetime

# Charger toutes les feuilles d'un fichier Excel dans un dictionnaire
def read_all_sheets_from_excel(filepath):
    excel_data = pd.ExcelFile(filepath)
    dataframes = {sheet_name: excel_data.parse(sheet_name) for sheet_name in excel_data.sheet_names}
    return dataframes

# Charger le fichier Excel et afficher un aperçu des différentes feuilles
FILEPATH = 'DataSetVentes.xlsx'
dataframes = read_all_sheets_from_excel(FILEPATH)

# Afficher les noms des feuilles et un aperçu des 15 premières lignes de chaque feuille
for k, v in dataframes.items():
    print('dataframe: ' + k)
    print(v.head(3))

# Lire les données
df_pants = pd.read_excel(FILEPATH, sheet_name='Pants sales')
df_dress = pd.read_excel(FILEPATH, sheet_name='Dress sales.')
df_sweater = pd.read_excel(FILEPATH, sheet_name='Sweater sales')

# 1. Échantillonnage aléatoire simple (30% des données)
sample_size = 0.6
pants_sample = df_pants.sample(frac=sample_size, random_state=42)
dress_sample = df_dress.sample(frac=sample_size, random_state=42)
sweater_sample = df_sweater.sample(frac=sample_size, random_state=42)

print("Taille des échantillons (30% des données):")
print("Pantalons:", len(pants_sample), "sur", len(df_pants), "observations")
print("Robes:", len(dress_sample), "sur", len(df_dress), "observations")
print("Pulls:", len(sweater_sample), "sur", len(df_sweater), "observations")


# Suppression des doublons dans chaque dataset
def remove_duplicates(df, name):
    initial_count = len(df)
    df_cleaned = df.drop_duplicates()
    final_count = len(df_cleaned)
    print("Nombre de doublons supprimés dans", name, ":", initial_count - final_count)
    return df_cleaned

# Nettoyage des données
df_pants_cleaned = remove_duplicates(pants_sample, "Pantalons")
df_dress_cleaned = remove_duplicates(dress_sample, "Robes")
df_sweater_cleaned = remove_duplicates(sweater_sample, "Pulls")

# Afficher un aperçu des données nettoyées
print("Aperçu des données nettoyées (Pantalons):")
print(df_pants_cleaned.head())

# Gestion des valeurs manquantes : Imputation par la moyenne
def impute_missing_values(df, name):
    for column in df.select_dtypes(include=["number"]).columns:
        if df[column].isnull().sum() > 0:
            # Calculate the mean value for missing values
            mean_value = df[column].mean()
            # Use .loc to safely update the column in the DataFrame
            df.loc[:, column] = df[column].fillna(mean_value)
            print(f"Valeurs manquantes imputées par la moyenne pour la colonne {column} dans {name}")
    return df


# Appliquer l'imputation sur chaque dataset nettoyé
df_pants_imputed = impute_missing_values(pants_sample, "Pantalons")
df_dress_imputed = impute_missing_values(dress_sample, "Robes")
df_sweater_imputed = impute_missing_values(sweater_sample, "Pulls")

# Vérifier les données après imputation
print("Aperçu des données après imputation (Pantalons):")
print(df_pants_imputed.head())


from scipy.stats import zscore
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Fonction pour détecter les valeurs aberrantes avec des boîtes à moustaches et des scores Z
def display_boxplots(dfs, titles):
    num_datasets = len(dfs)
    plt.figure(figsize=(15, 6))  # Taille de la figure globale

    for i, (df, title) in enumerate(zip(dfs, titles), start=1):
        plt.subplot(1, num_datasets, i)  # Sous-graphe
        numeric_columns = df.select_dtypes(include=["number"]).columns
        sns.boxplot(data=df[numeric_columns])
        plt.title(title)
        plt.xticks(rotation=45)

    plt.tight_layout()  # Ajuste automatiquement les espaces entre les graphiques
    plt.savefig("BoxPlot_All.png")



# Appliquer aux trois datasets
dfs = [df_pants_imputed, df_dress_imputed, df_sweater_imputed]
titles = ["Pantalons", "Robes", "Pulls"]

display_boxplots(dfs, titles)

# Standardisation
from sklearn.preprocessing import StandardScaler


def standardize_data(df, columns_to_standardize):
    # Initialisation du StandardScaler
    scaler = StandardScaler()

    # Appliquer la standardisation aux colonnes spécifiées
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

    return df


# Nettoyer les noms de colonnes en remplaçant les caractères indésirables
df_pants_imputed.columns = df_pants_imputed.columns.str.replace("\n", " ", regex=False)
df_dress_imputed.columns = df_dress_imputed.columns.str.replace("\n", " ", regex=False)
df_sweater_imputed.columns = df_sweater_imputed.columns.str.replace("\n", " ", regex=False)

# Vérification des noms de colonnes après nettoyage
print(df_pants_imputed.columns)

# Liste des colonnes numériques à standardiser (mises à jour avec les bons noms de colonnes)
columns_to_standardize = ["S/ billion Yuan", "X4/ billion", "X8/ 10 million", "X9/ 100 thousand", "X10/ million"]

# Appliquer la standardisation sur les datasets imputés
df_pants_standardized = standardize_data(df_pants_imputed, columns_to_standardize)
df_dress_standardized = standardize_data(df_dress_imputed, columns_to_standardize)
df_sweater_standardized = standardize_data(df_sweater_imputed, columns_to_standardize)

# Vérification des résultats
print("Données standardisées pour Pantalons:")
print(df_pants_standardized.head())

print("\nDonnées standardisées pour Robes:")
print(df_dress_standardized.head())

print("\nDonnées standardisées pour Pulls:")
print(df_sweater_standardized.head())


def plot_all_features_vs_sales_with_lines(datasets, sales_column, titles, output_filename):
    """
    Trace les relations entre les caractéristiques et les ventes pour plusieurs datasets
    dans une seule figure avec des lignes reliant les points.

    :param datasets: Liste des DataFrames
    :param sales_column: Nom de la colonne représentant les ventes
    :param titles: Liste des titres pour chaque DataFrame
    :param output_filename: Nom du fichier de sortie
    """
    num_datasets = len(datasets)
    max_features = max(len(df.select_dtypes(include=["number"]).columns) - 1 for df in datasets)
    plt.figure(figsize=(20, 5 * num_datasets))

    for dataset_idx, (df, title) in enumerate(zip(datasets, titles), start=1):
        numeric_columns = df.select_dtypes(include=["number"]).columns
        numeric_columns = [col for col in numeric_columns if col != sales_column]

        for feature_idx, column in enumerate(numeric_columns, start=1):
            plt.subplot(num_datasets, max_features, (dataset_idx - 1) * max_features + feature_idx)

            # Tracer les points
            sns.scatterplot(data=df, x=column, y=sales_column, color="blue", label="Points")

            # Tracer les lignes reliant les points
            sns.lineplot(data=df.sort_values(by=column), x=column, y=sales_column, color="red", label="Lignes")

            plt.title(f"{title}: {column} vs {sales_column}")
            plt.xlabel(column)
            plt.ylabel(sales_column)
            plt.legend()

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)



# Définir le nom de la colonne représentant les ventes
sales_column = "S/ billion Yuan"

# Appliquer aux datasets
datasets = [df_pants_standardized, df_dress_standardized, df_sweater_standardized]
titles = ["Pantalons", "Robes", "Pulls"]
output_filename = "All_Features_vs_Sales.png"

plot_all_features_vs_sales_with_lines(datasets, sales_column, titles, output_filename)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df, title):
    # Sélectionner uniquement les colonnes numériques
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Matrice de corrélation - {title}')
    plt.tight_layout()

    # Générer un nom de fichier unique
    filename = f'HeatMap_{title}.png'.replace(' ', '_')
    plt.savefig(filename)
    plt.close()  # Fermer la figure pour éviter des chevauchements entre les figures

# Créer les heatmaps pour chaque type de vêtement
plot_correlation_heatmap(df_pants_standardized, 'Pantalons')
plot_correlation_heatmap(df_dress_standardized, 'Robes')
plot_correlation_heatmap(df_sweater_standardized, 'Pulls')



import scipy.stats as stats

pants_sales = df_pants_cleaned['S/ billion Yuan']
dress_sales = df_dress_cleaned['S/ billion Yuan']
sweater_sales = df_sweater_cleaned['S/ billion Yuan']

# Test de Student entre les pantalons et les robes
t_stat_1, p_value_1 = stats.ttest_ind(pants_sales, dress_sales, nan_policy='omit')
print(f"Test de Student (Pantalons vs Robes) :")
print(f"t-statistique : {t_stat_1}, p-value : {p_value_1}")

# Test de Student entre les pantalons et les pulls
t_stat_2, p_value_2 = stats.ttest_ind(pants_sales, sweater_sales, nan_policy='omit')
print(f"\nTest de Student (Pantalons vs Pulls) :")
print(f"t-statistique : {t_stat_2}, p-value : {p_value_2}")

# Test de Student entre les robes et les pulls
t_stat_3, p_value_3 = stats.ttest_ind(dress_sales, sweater_sales, nan_policy='omit')
print(f"\nTest de Student (Robes vs Pulls) :")
print(f"t-statistique : {t_stat_3}, p-value : {p_value_3:.10f}")



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Fonction pour appliquer l'ACP
def apply_pca(df, name):
    # Sélectionner uniquement les colonnes numériques
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    data = df[numeric_cols]

    # Standardisation des données
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Appliquer l'ACP
    pca = PCA()
    principal_components = pca.fit_transform(data_scaled)

    # Variance expliquée
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Retourner les composantes principales et la variance expliquée cumulative
    return cumulative_variance, explained_variance

# Appliquer l'ACP sur chaque dataset nettoyé
pants_cumulative, pants_variance = apply_pca(df_pants_standardized, "Pantalons")
dress_cumulative, dress_variance = apply_pca(df_dress_standardized, "Robes")
sweater_cumulative, sweater_variance = apply_pca(df_sweater_standardized, "Pulls")

# Créer une figure avec des sous-graphiques
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Ajouter chaque graphique
axes[0].plot(range(1, len(pants_cumulative) + 1), pants_cumulative, marker='o', linestyle='--')
axes[0].set_title("Pantalons")
axes[0].set_xlabel("Nombre de composantes principales")
axes[0].set_ylabel("Variance expliquée cumulative")
axes[0].grid()

axes[1].plot(range(1, len(dress_cumulative) + 1), dress_cumulative, marker='o', linestyle='--')
axes[1].set_title("Robes")
axes[1].set_xlabel("Nombre de composantes principales")
axes[1].grid()

axes[2].plot(range(1, len(sweater_cumulative) + 1), sweater_cumulative, marker='o', linestyle='--')
axes[2].set_title("Pulls")
axes[2].set_xlabel("Nombre de composantes principales")
axes[2].grid()

# Sauvegarder l'image
plt.tight_layout()  # Ajuste les espaces entre les sous-graphiques
plt.savefig('ACP.png')


# Importer les bibliothèques nécessaires
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Sélectionner les colonnes pertinentes pour l'entraînement
# Supposons que "S/ billion Yuan" soit la colonne cible (les ventes), et les autres colonnes sont les caractéristiques
# Ici, nous considérons des caractéristiques numériques uniquement
df_pants_features = df_pants_cleaned.drop(columns=["Date", "S/ billion Yuan"])
df_pants_target = df_pants_cleaned["S/ billion Yuan"]

# Étape 1 : Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df_pants_features, df_pants_target, test_size=0.2, random_state=42)

# Étape 2 : Entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)
# Étape 3 : Tester le modèle sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer la performance du modèle
# Calculer l'erreur quadratique moyenne (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Erreur quadratique moyenne (RMSE) : ", rmse)

# Afficher les coefficients du modèle pour chaque caractéristique
print("Coefficients du modèle : ", model.coef_)

# Afficher l'ordonnée à l'origine (intercept)
print("Ordonnée à l'origine (intercept) : ", model.intercept_)

#  Comparer les valeurs réelles et prédites pour un échantillon
comparison = pd.DataFrame({"Ventes réelles": y_test, "Ventes prédites": y_pred})
print(comparison.head())

# Importer les bibliothèques nécessaires
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Créer une figure avec plusieurs sous-graphiques
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2 lignes, 2 colonnes

# Étape 1 : Tracer les valeurs réelles vs prédites (graphique de dispersion)
axes[0, 0].scatter(y_test, y_pred, color='blue', alpha=0.6)
axes[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ligne idéale
axes[0, 0].set_title("Ventes réelles vs Ventes prédites")
axes[0, 0].set_xlabel("Ventes réelles")
axes[0, 0].set_ylabel("Ventes prédites")
axes[0, 0].grid(True)

# Étape 2 : Tracer la distribution des erreurs
errors = y_test - y_pred
sns.histplot(errors, kde=True, color='green', bins=30, ax=axes[0, 1])
axes[0, 1].set_title("Distribution des erreurs")
axes[0, 1].set_xlabel("Erreur (Ventes réelles - Ventes prédites)")
axes[0, 1].set_ylabel("Fréquence")
axes[0, 1].grid(True)

# Étape 3 : Comparer les valeurs réelles et prédites dans un tableau
comparison = pd.DataFrame({"Ventes réelles": y_test, "Ventes prédites": y_pred})
axes[1, 0].axis('off')  # Masquer ce sous-graphe pour ne pas afficher un graphique vide
axes[1, 0].table(cellText=comparison.head().values, colLabels=comparison.columns, loc='center')

# Étape 4 : Visualisation des coefficients du modèle
sns.barplot(x=df_pants_features.columns, y=model.coef_, ax=axes[1, 1])
axes[1, 1].set_title("Coefficients du modèle de régression linéaire")
axes[1, 1].set_xlabel("Caractéristiques")
axes[1, 1].set_ylabel("Coefficient")
axes[1, 1].grid(True)

# Ajuster l'espacement entre les sous-graphiques
plt.tight_layout()

# Sauvegarder l'image avec tous les graphiques
plt.savefig("pants_graphs.png")



# Importer les bibliothèques nécessaires
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Sélectionner les colonnes pertinentes pour l'entraînement
# Supposons que "S/ billion Yuan" soit la colonne cible (les ventes), et les autres colonnes sont les caractéristiques
df_robes_features = df_dress_cleaned.drop(columns=["Date", "S/ billion Yuan"])  # Remplacer df_pants_cleaned par df_robes_cleaned
df_robes_target = df_dress_cleaned["S/ billion Yuan"]  # Remplacer df_pants_cleaned par df_robes_cleaned

# Étape 1 : Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df_robes_features, df_robes_target, test_size=0.2, random_state=42)

# Étape 2 : Entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Étape 3 : Tester le modèle sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer la performance du modèle
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Erreur quadratique moyenne (RMSE) pour les robes : ", rmse)

# Afficher les coefficients du modèle pour chaque caractéristique
print("Coefficients du modèle pour les robes : ", model.coef_)

# Afficher l'ordonnée à l'origine (intercept)
print("Ordonnée à l'origine (intercept) pour les robes : ", model.intercept_)

# Comparer les valeurs réelles et prédites pour un échantillon
comparison = pd.DataFrame({"Ventes réelles": y_test, "Ventes prédites": y_pred})
print(comparison.head())

# Importer les bibliothèques nécessaires pour la visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Créer une figure avec plusieurs sous-graphiques
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Tracer les valeurs réelles vs prédites (graphique de dispersion)
axes[0, 0].scatter(y_test, y_pred, color='blue', alpha=0.6)
axes[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ligne idéale
axes[0, 0].set_title("Ventes réelles vs Ventes prédites pour les robes")
axes[0, 0].set_xlabel("Ventes réelles")
axes[0, 0].set_ylabel("Ventes prédites")
axes[0, 0].grid(True)

# Tracer la distribution des erreurs
errors = y_test - y_pred
sns.histplot(errors, kde=True, color='green', bins=30, ax=axes[0, 1])
axes[0, 1].set_title("Distribution des erreurs pour les robes")
axes[0, 1].set_xlabel("Erreur (Ventes réelles - Ventes prédites)")
axes[0, 1].set_ylabel("Fréquence")
axes[0, 1].grid(True)

# Comparer les valeurs réelles et prédites dans un tableau
comparison = pd.DataFrame({"Ventes réelles": y_test, "Ventes prédites": y_pred})
axes[1, 0].axis('off')  # Masquer ce sous-graphe pour ne pas afficher un graphique vide
axes[1, 0].table(cellText=comparison.head().values, colLabels=comparison.columns, loc='center')

# Visualisation des coefficients du modèle
sns.barplot(x=df_robes_features.columns, y=model.coef_, ax=axes[1, 1])
axes[1, 1].set_title("Coefficients du modèle de régression linéaire pour les robes")
axes[1, 1].set_xlabel("Caractéristiques")
axes[1, 1].set_ylabel("Coefficient")
axes[1, 1].grid(True)

# Ajuster l'espacement entre les sous-graphiques
plt.tight_layout()

# Sauvegarder l'image avec tous les graphiques
plt.savefig("graphs_robes.png")



# Sélectionner les colonnes pertinentes pour l'entraînement
# Supposons que "S/ billion Yuan" soit la colonne cible (les ventes), et les autres colonnes sont les caractéristiques
# Ici, nous considérons des caractéristiques numériques uniquement pour les pulls
df_sweaters_features = df_sweater_cleaned.drop(columns=["Date", "S/ billion Yuan"])
df_sweaters_target = df_sweater_cleaned["S/ billion Yuan"]

# Étape 1 : Diviser les données en ensembles d'entraînement et de test
X_train_sweaters, X_test_sweaters, y_train_sweaters, y_test_sweaters = train_test_split(df_sweaters_features, df_sweaters_target, test_size=0.2, random_state=42)

# Étape 2 : Entraîner le modèle de régression linéaire
model_sweaters = LinearRegression()
model_sweaters.fit(X_train_sweaters, y_train_sweaters)

# Étape 3 : Tester le modèle sur l'ensemble de test
y_pred_sweaters = model_sweaters.predict(X_test_sweaters)

# Évaluer la performance du modèle pour les pulls
# Calculer l'erreur quadratique moyenne (RMSE)
rmse_sweaters = np.sqrt(mean_squared_error(y_test_sweaters, y_pred_sweaters))
print("Erreur quadratique moyenne (RMSE) pour les pulls : ", rmse_sweaters)

# Afficher les coefficients du modèle pour chaque caractéristique
print("Coefficients du modèle pour les pulls : ", model_sweaters.coef_)

# Afficher l'ordonnée à l'origine (intercept)
print("Ordonnée à l'origine (intercept) pour les pulls : ", model_sweaters.intercept_)

# Comparer les valeurs réelles et prédites pour les pulls
comparison_sweaters = pd.DataFrame({"Ventes réelles": y_test_sweaters, "Ventes prédites": y_pred_sweaters})
print(comparison_sweaters.head())

# Créer une figure avec plusieurs sous-graphiques pour les pulls
fig_sweaters, axes_sweaters = plt.subplots(2, 2, figsize=(14, 10))  # 2 lignes, 2 colonnes

# Étape 1 : Tracer les valeurs réelles vs prédites (graphique de dispersion)
axes_sweaters[0, 0].scatter(y_test_sweaters, y_pred_sweaters, color='blue', alpha=0.6)
axes_sweaters[0, 0].plot([min(y_test_sweaters), max(y_test_sweaters)], [min(y_test_sweaters), max(y_test_sweaters)], color='red', linestyle='--')  # Ligne idéale
axes_sweaters[0, 0].set_title("Ventes réelles vs Ventes prédites pour les pulls")
axes_sweaters[0, 0].set_xlabel("Ventes réelles")
axes_sweaters[0, 0].set_ylabel("Ventes prédites")
axes_sweaters[0, 0].grid(True)

# Étape 2 : Tracer la distribution des erreurs pour les pulls
errors_sweaters = y_test_sweaters - y_pred_sweaters
sns.histplot(errors_sweaters, kde=True, color='green', bins=30, ax=axes_sweaters[0, 1])
axes_sweaters[0, 1].set_title("Distribution des erreurs pour les pulls")
axes_sweaters[0, 1].set_xlabel("Erreur (Ventes réelles - Ventes prédites)")
axes_sweaters[0, 1].set_ylabel("Fréquence")
axes_sweaters[0, 1].grid(True)

# Étape 3 : Comparer les valeurs réelles et prédites dans un tableau
comparison_sweaters = pd.DataFrame({"Ventes réelles": y_test_sweaters, "Ventes prédites": y_pred_sweaters})
axes_sweaters[1, 0].axis('off')  # Masquer ce sous-graphe pour ne pas afficher un graphique vide
axes_sweaters[1, 0].table(cellText=comparison_sweaters.head().values, colLabels=comparison_sweaters.columns, loc='center')

# Étape 4 : Visualisation des coefficients du modèle pour les pulls
sns.barplot(x=df_sweaters_features.columns, y=model_sweaters.coef_, ax=axes_sweaters[1, 1])
axes_sweaters[1, 1].set_title("Coefficients du modèle de régression linéaire pour les pulls")
axes_sweaters[1, 1].set_xlabel("Caractéristiques")
axes_sweaters[1, 1].set_ylabel("Coefficient")
axes_sweaters[1, 1].grid(True)

# Ajuster l'espacement entre les sous-graphiques
plt.tight_layout()

# Sauvegarder l'image avec tous les graphiques pour les pulls
plt.savefig("graphs_Pull.png")
