import pandas as pd
import re

# Charger le fichier CSV
df = pd.read_csv("SPY_returns.csv", parse_dates=[0]) 

# Fonction pour nettoyer les noms de colonnes
def clean_column_names(columns):
    new_columns = []
    seen_names = set()
    for i, col in enumerate(columns):
        try:
            float(col)
            new_col_name = f"feature_{i+1}"
        except ValueError:
            new_col_name = col

        new_col_name = new_col_name.lower().strip()
        new_col_name = re.sub(r'\W+', '_', new_col_name)

        while new_col_name in seen_names:
            new_col_name += "_dup"
        seen_names.add(new_col_name)

        new_columns.append(new_col_name)
    
    return new_columns

# Appliquer le nettoyage des colonnes
df.columns = clean_column_names(df.columns)

# Supprimer les colonnes en double
df = df.loc[:, ~df.columns.duplicated()]
print("Colonnes apres suppression des doublons :", df.columns)

# Analyser les valeurs manquantes
missing_summary = df.isnull().sum() / len(df)  # Proportion de valeurs manquantes
print("Pourcentage de valeurs manquantes par colonne :\n", missing_summary)

# Strategie : Supprimer les colonnes avec plus de 80% de valeurs manquantes
threshold = 0.8
cols_to_drop = missing_summary[missing_summary > threshold].index
df = df.drop(columns=cols_to_drop)
print("Colonnes supprimees car trop de valeurs manquantes :", cols_to_drop)

# Imputation des valeurs manquantes (median pour les variables numeriques)
for col in df.select_dtypes(include=['number']).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Verification de la validite des donnees
def check_dataframe(df):
    errors = []
    
    if df.columns.duplicated().any():
        errors.append("Il y a des colonnes en double apres le renommage.")
    
    if any(col.strip() == "" for col in df.columns):
        errors.append("Il y a des colonnes vides ou avec un nom incorrect.")

    if df.isnull().all().any():
        errors.append("Certaines colonnes contiennent uniquement des valeurs manquantes.")
    
    if not pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
        errors.append("Attention : La premiere colonne ne semble pas etre une date.")

    if errors:
        print("\nProblemes detectes :")
        for error in errors:
            print(error)
        print("\nCorrigez ces erreurs avant d executer le script principal.")
    else:
        print("\nTout est correct. Vous pouvez executer votre script principal.")

# Executer la verification
check_dataframe(df)
