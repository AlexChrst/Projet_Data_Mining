"""
Projet Data Mining
M2 MoSEF

main.py: Script principal
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import functions_EDA as f_eda
import functions_pretraitement as f_pre
import functions_modelisation as f_m

# Importation des donnees
df = pd.read_csv("data/train.csv", index_col='id')
test = pd.read_csv("data/test.csv", index_col='id')
submission = pd.read_csv("data/sample_submission.csv")


# EDA =============================================================================================================

df.head()
df.shape
df.info()
df.describe()

# Observations des variables
df.columns

# Exploration univariee
f_eda.afficher_df_head(df)
f_eda.afficher_cat_vars_univarie_graph(df, 'Exited', palette=["purple"])
f_eda.afficher_cat_vars_univarie_tableau(df, 'Exited')
f_eda.afficher_num_vars_univarie_graph(df, palette=["#3336ff"])

len(df['Surname'].unique()) # 755 noms différents

# Traitement des variables categorielles

vars_cat = df.select_dtypes(include=['object', 'category']).columns
print(vars_cat) # vars cat sans 'Exited'

vars_cat = list(vars_cat)
vars_cat.remove('Surname')

for var in vars_cat:
    print(df[var].nunique(),' modalités issues de la variable', str(var) + " : ", df[var].unique())


# Traitement des variables numériques

vars_num = df.select_dtypes(include=['float', 'int']).columns
vars_num = list(vars_num)
vars_num.remove('Exited'); vars_num.remove('CustomerId')
print(vars_num)


# Exploration bivariee
ma_palette = ["#ff0000", "#3511ca"]

f_eda.afficher_num_vars_bivarie_graph(df, 'Exited', palette = ma_palette)
f_eda.afficher_cat_vars_bivarie_tableau(df, 'Exited')

# Exploration multivarie

f_eda.afficher_matrice_correlation_num(df)


# Data preprocessing =============================================================================================================

# Traitement des valeurs manquantes
f_pre.afficher_pourcentage_valeurs_manquantes(df)

# Feature Engineering / Pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), vars_num),
    (OneHotEncoder(drop='first'), vars_cat),
    remainder='passthrough'  # pour garder les autres colonnes
)

# pipeline
pipeline = make_pipeline(preprocessor)

# Traitement de df (train.csv)
df.drop(columns = ['CustomerId', 'Surname'], inplace=True)
df_transformed = pipeline.fit_transform(df)
encoded_columns = pipeline.named_steps['columntransformer'].named_transformers_['onehotencoder'].get_feature_names_out(vars_cat)
all_columns = vars_num + list(encoded_columns) + ['Exited']

df_clean = pd.DataFrame(df_transformed, columns=all_columns)

df_clean.columns

# Traitement de test (test.csv)
test.drop(columns = ['CustomerId', 'Surname'], inplace=True)
test_transformed = pipeline.fit_transform(test)
all_columns.remove('Exited')
test_clean = pd.DataFrame(test_transformed, columns=all_columns)
test_clean['Exited'] = pd.NA # variable target vide pour l'instant

test_clean.columns

# Modelization =============================================================================================================

features = list(df_clean.columns)
features.remove('Exited')

# Random forest / Gridsearch K-fold
# best_model, best_params = f_m.random_forest_kfold_gridsearch(df_clean, features, 'Exited')
# y_pred_proba = best_model.predict_proba(test_clean[features])[:,1]

# XGBoost fine-tuned
# best_model = f_m.optuna_optimization_xgb(df_clean[features], df_clean['Exited'])
# y_pred_proba = best_model.predict_proba(test_clean[features])[:,1]

# LGBM fine-tuned
best_model = f_m.optuna_optimization_lgbm(df_clean[features], df_clean['Exited'])
y_pred_proba = best_model.predict_proba(test_clean[features])[:,1]

submission['Exited'] = y_pred_proba


date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Export des best hyperparameters
best_params = best_model.get_params()
hyperparam_name = f"hyperparameters/hyperparameters_{date_time_str}.txt"

with open(hyperparam_name, 'w') as file:
    json.dump(best_params, file, indent=4)

# Export de la soumission
submission_name = f"submissions/submission_{date_time_str}.csv"
submission.to_csv(submission_name, sep=',', index=False)




