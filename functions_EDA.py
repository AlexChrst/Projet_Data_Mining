"""
Projet Data Mining

Fonctions pour EDA
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def afficher_cat_vars_univarie_graph(df, var_y, palette):

    vars_cat = df.select_dtypes(include=['object', 'category']).columns
    vars_cat = list(vars_cat)
    vars_cat.append(var_y)

    for var in vars_cat:
        plt.figure(figsize=(12, 5))
        
        total_counts = df[var].value_counts()
        pourcentages = total_counts / total_counts.sum() * 100
        
        ax = sns.barplot(x=pourcentages.index, y=pourcentages.values, palette=palette)
        
        ax.set_ylabel("Pourcentage (%)")
        plt.title(f"Distribution de {var}")
        
        # pourcentage sur chaque barre
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 8), textcoords='offset points')
        
        plt.show()

def afficher_cat_vars_univarie_tableau(df, var_y):
    vars_cat = df.select_dtypes(include=['object', 'category']).columns
    vars_cat = list(vars_cat)
    vars_cat.append(var_y)

    for var in vars_cat:
        tableau_contingence = df[var].value_counts(normalize=True) * 100  # pourcentage
        print(f"Distribution en % de {var}:\n")
        print(tableau_contingence)
        print("\n" + "-"*50 + "\n")


def afficher_num_vars_univarie_graph(df, palette):
    vars_num = df.select_dtypes(include=['float', 'int']).columns

    for var in vars_num:
        plt.figure(figsize=(12, 5))
        
        # histo
        plt.subplot(1, 2, 1)
        if df[var].nunique()>30:
            kde_bool = True
        else :
            kde_bool = False
        sns.histplot(df[var], kde=kde_bool, color=palette[0])

        plt.title(f"Histogramme de {var}")
        plt.xlabel(var)
        plt.ylabel("Nombre d'observations")

        # boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[var], palette=palette)
        plt.title(f"Boxplot de {var}")
        plt.xlabel(var)

        plt.tight_layout()
        plt.show()



# Bivarie

def afficher_cat_vars_bivarie_graph(df, y, palette):
    vars_cat = df.select_dtypes(include=['object', 'category']).columns
    for var in vars_cat:
        plt.figure(figsize=(10, 5))
        ax = sns.countplot(data=df, x=var, hue=y, stat="percent", palette=palette)
        ax.set_ylabel("Pourcent (%)")
        plt.title(f"Distribution de {var} x {y}")
        plt.show()

def afficher_cat_vars_bivarie_tableau(df, y):
    vars_cat = df.select_dtypes(include=['object', 'category']).columns
    for var in vars_cat:
        tableau_freq_rows= pd.crosstab(df[var], df[y], normalize='index') * 100  # pourcentage
        tableau_freq_columns = pd.crosstab(df[var], df[y], normalize='columns') * 100  # pourcentage
        tableau_freq_all = pd.crosstab(df[var], df[y], normalize='all') * 100  # pourcentage
        print(f"Tableaux de fréquence /lignes pour {var} x {y}:\n")
        print(tableau_freq_rows, "\n")
        print(f"Tableaux de fréquence /colonnes pour {var} x {y}:\n")
        print(tableau_freq_columns, "\n")
        print(f"Tableaux de fréquence /tous pour {var} x {y}:\n")
        print(tableau_freq_all, "\n")
        print("\n" + "-"*50 + "\n")

def afficher_num_vars_bivarie_graph(df, y, palette):
    vars_num = df.select_dtypes(include=['float', 'int']).columns
    vars_num = list(vars_num)
    vars_num.remove(y)

    for var in vars_num:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x=y, y=var, palette=palette)
        plt.title(f"Boxplots de {var} x {y}")
        plt.show()

def v_cramer_test_khi2(var1, var2):
    """Calcule le V de Cramer entre 2 variables catégorielles"""
    matrice_confusion = pd.crosstab(var1, var2)
    # test Chi-2
    chi2_stat, p_val, _ , _ = chi2_contingency(matrice_confusion)
    n = matrice_confusion.sum().sum()
    r, k = matrice_confusion.shape
    return round(np.sqrt(chi2_stat / (n*(min(r,k)-1))),3), round(p_val,3)

def correlation_ratio(var_cat, var_num):
    """Calcule le eta entre une var cat et une var num"""
    fcat, _ = pd.factorize(var_cat)
    cat_means = np.array([var_num[fcat == i].mean() for i in range(len(np.unique(fcat)))])
    overall_mean = var_num.mean()
    numerateur = np.sum([len(var_num[fcat == i]) * (cat_mean - overall_mean) ** 2 for i, cat_mean in enumerate(cat_means)])
    denominateur = np.sum((var_num - overall_mean) ** 2)
    return np.sqrt(numerateur / denominateur)

# Multivarie

def afficher_df_head(df):
    for var in df.columns:
        print(f"{var}:\n{df[var].head()}\n")

def afficher_matrice_correlation_num(df, methode='pearson'):
    # methode: pearson, kendall ou spearman
    df_num = df.select_dtypes(include=['float64', 'int64'])
    matrice_correlation = df_num.corr(method = methode)
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(matrice_correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matrice de corrélation')
    plt.show()
