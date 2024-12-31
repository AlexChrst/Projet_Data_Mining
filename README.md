                                                                ####### Challenge Kaggle: churn prediction #######

L'objectif de ce projet est de développer un modèle permettant de prédire la probabilité qu'un client 
quitte la banque (variable "Exited) à l'aide des variables fournies (son score de crédit, son âge, s'il détient
ou non une carte de crédit...)

###### Structure du code ######

## functions_EDA.py

Script contenant les fonctions pour l'analyse exploratoire des données (EDA).

## functions_pretraitement.py

Script contenant les fonction pour le pré-traitement des données

## functions_modelisation.py

Script contenant les fonctions pour la modélisation

## main.py

Script permettant de lancer toutes les étapes de la pipeline du projet. Il importe ainsi
les fonctions décrites précédemment. Il enregistre également les hyper-paramètres des modèles lancés et 
les modèles eux-mêmes en format Joblib

###### Structure du code ######

## Se placer dans le dossier et ouvrir un terminal
## Créer un environnement virtuel : python -m venv <nom_de_l_environnement>
## Activer l'environnement virtual : <nom_de_l_environnement>\Scripts\activate
## Installer les dépendances nécessaire au projet : pip install -r requirements.txt
## Lancer le Scrip main.py : python main.py

