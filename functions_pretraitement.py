"""
Projet Data Mining

Fonctions pour EDA
"""
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor

# Traitement des valeurs manquantes

def afficher_pourcentage_valeurs_manquantes(df):
    pourcentage_manquantes = df.isna().mean() * 100
    pourcentage_manquantes = pourcentage_manquantes.sort_values(ascending=False)
    print("Pourcentage de valeurs manquantes par variable (en %) :\n")
    print(round(pourcentage_manquantes), 2)

def afficher_nb_valeurs_manquantes(df):
    lignes_avec_n_manquantes = df.isna().sum(axis=1) # nb de valeurs manquantes par ligne
    return lignes_avec_n_manquantes

def afficher_valeurs_manquantes_barplot(df):
   
    missing_values = (df.isnull().sum() / len(df)) * 100
    missing_values = missing_values.sort_values(ascending=False)
    # barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values.index, y=missing_values, palette='viridis')
    plt.xlabel('Variable')
    plt.ylabel('Pourcentage de valeurs manquantes (%)')
    plt.title('Pourcentage de valeurs manquantes par variable')
    plt.show()


def random_forest_imputation(df, colonne_a_imputer):
    df_complet = df[df[colonne_a_imputer].notna()]
    df_manquant = df[df[colonne_a_imputer].isna()]
    
    features = df_complet.select_dtypes(include=[np.number]).drop(columns=[colonne_a_imputer, 'BAD']).columns.tolist()

    X = df_complet[features]
    y = df_complet[colonne_a_imputer]
    
    model = RandomForestRegressor(random_state=999)
    model.fit(X, y)
    
    df_manquant[colonne_a_imputer] = model.predict(df_manquant[features])
    
    # concatenation du df complet avec le df manquant imputé
    df_impute = pd.concat([df_complet, df_manquant]).sort_index()
    
    return df_impute

# Normalisation

def normaliser_variables(df, y, scaler = "standard" ):

    vars_num = df.select_dtypes(include=['float', 'int']).columns
    vars_num = list(vars_num)
    vars_num.remove(y)

    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    elif scaler == 'robust':
        scaler = RobustScaler()

    df[vars_num] = scaler.fit_transform(df[vars_num])
    
    return df

def creation_variables_interaction(df, cols, polynomial = True):
    """
    Cette fonction permet de créer des variables d'intéraction à partir de colonnes données
    d'un df d'entrée
    """

    # boucle sur toutes les combinaisons avec remplacement
    for col1, col2 in combinations_with_replacement(cols, 2):
        
        if col1 == col2:
            if polynomial:
                df[f'{col1}**2'] = df[col1] ** 2
        else:
            # Si les deux colonnes sont différentes -> interaction
            df[f'{col1}*{col2}'] = df[col1] * df[col2]
    
    return df