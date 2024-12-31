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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import joblib

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

len(df['CustomerId'].unique())

# CustomerId
moyenne_exited_customerid = df.groupby('CustomerId').mean('Exited')
moyenne_exited_customerid['Exited'].value_counts()

df_shuffled = df.copy()
df_shuffled['CustomerId'] = df['CustomerId'].sample(frac=1).reset_index(drop=True)
moyenne_exited_customerid_shuffled = df_shuffled.groupby('CustomerId').mean('Exited')
moyenne_exited_customerid_shuffled['Exited'].value_counts()
# Même distribution entre CustomerId normal et CustomerId mélangé donc pas de lien entre
# CustomerId et Exited

len(df['Surname'].unique()) # 755 noms différents
moyenne_exited_surname = df.groupby('Surname').mean('Exited')
moyenne_exited_surname['Exited'].value_counts()

df_shuffled = df.copy()
df_shuffled['Surname'] = df['Surname'].sample(frac=1).reset_index(drop=True)
moyenne_exited_surname_shuffled = df_shuffled.groupby('Surname').mean('Exited')
moyenne_exited_surname_shuffled['Exited'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(15,8))

# Histogramme pour 'moyenne_exited_surname'
axes[0].hist(moyenne_exited_surname['Exited'], bins=20, edgecolor='black')
axes[0].set_title('Normal')
axes[1].hist(moyenne_exited_surname_shuffled['Exited'], bins=20, color='red',edgecolor='black')
axes[1].set_title('Shuffled')

plt.tight_layout()
plt.show()
# Même distribution entre Surname normal et Surname mélangé donc pas de lien entre
# Surname et Exited


# Traitement des variables categorielles
vars_cat = df.select_dtypes(include=['object', 'category']).columns

vars_cat = list(vars_cat)
vars_cat.remove('Surname')
print(vars_cat) # vars cat sans 'Exited'

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
f_pre.afficher_pourcentage_valeurs_manquantes(df) # pas de valeurs manquantes


# Feature Engineering / Pipeline
def create_new_features_scaled(X):
    """
    createur de nouvelles features
    """
    X = X.copy()
    # ** 2
    X['CreditScore^2'] = X['CreditScore'] ** 2
    X['Age^2'] = X['Age'] ** 2
    X['Tenure^2'] = X['Tenure'] ** 2
    X['Balance^2'] = X['Balance'] ** 2
    X['EstimatedSalary^2'] = X['EstimatedSalary'] ** 2

    # interactions
    X['Age_Balance'] = X['Age'] * X['Balance']
    X['CreditScore_IsActiveMember'] = X['CreditScore'] * X['IsActiveMember']
    X['NumOfProducts_HasCrCard'] = X['NumOfProducts'] * X['HasCrCard']

    X.drop(columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'NumOfProducts', 'IsActiveMember', 'HasCrCard'], inplace=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled

# pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), vars_num),
    (OneHotEncoder(drop='first'), vars_cat),
    (FunctionTransformer(create_new_features_scaled), ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'NumOfProducts', 
                                                'IsActiveMember', 'HasCrCard']),
    remainder='passthrough'
)

pipeline = make_pipeline(preprocessor)

# Traitement de df (train.csv)
df.drop(columns = ['CustomerId', 'Surname'], inplace=True)
df_transformed = pipeline.fit_transform(df)
encoded_columns = pipeline.named_steps['columntransformer'].named_transformers_['onehotencoder'].get_feature_names_out(vars_cat)
new_features = ['CreditScore^2', 'Age^2', 'Tenure^2', 'Balance^2', 'EstimatedSalary^2', 'Age_Balance', 'CreditScore_IsActiveMember', 'NumOfProducts_HasCrCard']
all_columns = vars_num + list(encoded_columns) + new_features + ['Exited']

df_clean = pd.DataFrame(df_transformed, columns= all_columns)


# Traitement de test (test.csv)
test.drop(columns = ['CustomerId', 'Surname'], inplace=True)
test_transformed = pipeline.fit_transform(test)
all_columns.remove('Exited')
test_clean = pd.DataFrame(test_transformed, columns= all_columns)
test_clean['Exited'] = pd.NA # variable target vide pour l'instant

test_clean.columns

# exportation des dataframes preprocessés
df_clean.to_csv('data/df_clean.csv', sep=';', index=False)
test_clean.to_csv('data/test_clean.csv', sep=';', index=False)

# Modelization ==============================================================================================================

# Imporation des dataframes déjà preprocessés
df_clean = pd.read_csv("data/df_clean.csv", sep=';')
test_clean = pd.read_csv("data/test_clean.csv", sep=';')

features = list(df_clean.columns) # liste de toutes les features
features.remove('Exited')

# Random forest / Gridsearch K-fold ===============
# best_model, best_params = f_m.random_forest_kfold_gridsearch(df_clean, features, 'Exited')

# XGBoost fine-tuned / Optuna ===============
#best_model = f_m.optuna_optimization_xgb(df_clean[features], df_clean['Exited'])


# LGBM fine-tuned / Optuna ===============
#best_model = f_m.optuna_optimization_lgbm(df_clean[features], df_clean['Exited'])

#Catboost fine-tuned /Optuna 
best_model=f_m.optuna_optimization_catboost(df_clean[features], df_clean['Exited'])

# prédictions mises dans submission
y_pred_proba = best_model.predict_proba(test_clean[features])[:,1]
submission['Exited'] = y_pred_proba

date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nom_modele = 'CatBoost_V3'

# export des best hyperparamètres
best_params = best_model.get_params()
hyperparam_name = f"artifacts/hyperparameters/{nom_modele}_hyperparameters_{date_time_str}.txt"

with open(hyperparam_name, 'w') as file:
    json.dump(best_params, file, indent=4)

# export du modèle en .joblib
model_name = f"artifacts/models/{nom_modele}_model_{date_time_str}.joblib"
joblib.dump(best_model, model_name)

# export de la soumission
submission_name = f"submissions/{nom_modele}_submission_{date_time_str}.csv"
submission.to_csv(submission_name, sep=',', index=False)

print(f"modèle sauvegardé sous: {model_name}")
print(f"hyperparamètres sauvegardés sous: {hyperparam_name}")
print(f"soumission sauvegardée sous: {submission_name}")