# 📊 Projet d’Analyse des Ventes

## 📌 Description

Ce projet a pour objectif d’analyser les ventes de produits vestimentaires (pantalons, robes et pulls) sur une plateforme d’E-Commerce entre **mai 2016 et avril 2019**.
L’analyse repose sur différentes étapes de préparation et de modélisation des données afin de :

* Détecter les tendances et opportunités commerciales
* Anticiper les ventes futures
* Identifier les facteurs influençant les résultats
* Optimiser les décisions stratégiques en matière de marketing, tarification et gestion des stocks

---

## 🎯 Objectifs

* **Optimiser les performances commerciales** : identifier les segments les plus rentables
* **Prédire les ventes futures** : planification des ressources et des stocks
* **Analyser les facteurs influents** : caractéristiques des produits, conditions de marché, promotions
* **Améliorer la rentabilité** : concentrer les efforts sur les leviers à fort impact
* **Atteindre les objectifs commerciaux** : suivi et ajustement des stratégies

---

## 📂 Jeu de Données

* Source : plateforme d’E-Commerce
* Période : **01/05/2016 → 01/04/2019**
* Catégories :

  * Ventes de pantalons
  * Ventes de robes
  * Ventes de pulls

### Variables principales

* **S** : Ventes (milliards de Yuan)
* **X4** : Pages vues (milliards)
* **X8** : Achats supplémentaires (dizaines de millions)
* **X9** : Indice de groupe de clients (centaines de milliers)
* **X10** : Indice de transaction (millions)

---

## 🔎 Processus d’Analyse

1. Collecte des données
2. Échantillonnage (aléatoire simple)
3. Suppression des doublons
4. Gestion des valeurs manquantes
5. Détection des valeurs aberrantes (IQR & boxplots)
6. Mise à l’échelle des caractéristiques (standardisation)
7. Analyse des relations entre variables et ventes
8. Étude des corrélations (heatmaps)
9. Tests d’hypothèses (Test de Student)
10. Réduction de dimension (ACP)
11. Prédiction des ventes (régression linéaire)

---

## 📈 Résultats

* **Relations fortes** :

  * X8 (achats supplémentaires) et X10 (indice de transaction) sont les meilleurs indicateurs des ventes.
* **Tests de Student** :

  * Différences significatives entre les moyennes des ventes (pantalons vs robes, pantalons vs pulls, robes vs pulls).
* **ACP** :

  * 3 composantes principales suffisent à expliquer la quasi-totalité de la variance.
* **Prédiction des ventes** :

  * Bonne précision obtenue (RMSE faible : \~0.08 à 0.13 selon la catégorie).

---

## 🛠️ Technologies Utilisées

* **Python**
* **Pandas, NumPy** (manipulation de données)
* **Matplotlib, Seaborn** (visualisations)
* **Scikit-learn** (ACP, régression linéaire, métriques)

---

## ✅ Conclusion

Ce projet démontre comment l’analyse statistique et la modélisation prédictive peuvent aider une entreprise à :

* Comprendre ses ventes
* Identifier les leviers de croissance
* Prendre de meilleures décisions stratégiques

Les modèles développés fournissent des prédictions fiables qui peuvent servir à la gestion des stocks et à l’optimisation des campagnes marketing.

---


