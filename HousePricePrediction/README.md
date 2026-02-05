# House Price Prediction

## Description
Ce projet utilise le **Machine Learning** pour prédire le prix des maisons à partir de différentes caractéristiques du quartier et de la maison.  
Le projet est réalisé en **Python** avec les librairies suivantes : `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`.

L’objectif est de comparer le **prix réel** et le **prix prédit** par les modèles pour voir leurs performances.

---

## Dataset
Le dataset utilisé contient les colonnes suivantes (features) :  

| Feature | Description |
|---------|-------------|
| CRIM    | Taux de criminalité par habitant par ville |
| ZN      | Proportion de terrains résidentiels zonés pour de grandes parcelles |
| INDUS   | Proportion de terres non commerciales |
| CHAS    | Variable binaire si la maison borde la rivière (1) ou non (0) |
| NOX     | Concentration de NOx (pollution) |
| RM      | Nombre moyen de pièces par logement |
| AGE     | Proportion de maisons construites avant 1940 |
| DIS     | Distances pondérées aux centres d’emploi de Boston |
| RAD     | Indice d’accessibilité aux autoroutes radiales |
| TAX     | Taux d’imposition foncière |
| PTRATIO | Ratio élèves/professeur par ville |
| B       | Proportion de la population noire |
| LSTAT   | Pourcentage de la population à bas statut économique |
| price   | **Prix de la maison (target)** |

Le modèle va prédire la colonne **`price`** à partir des autres features.

---

## Étapes du projet

1. **Exploration des données (EDA)**  
   - Analyse statistique des features (`.describe()`)  
   - Visualisation des distributions et corrélations avec `seaborn`  
   - Visualisation des outliers  

2. **Préparation des données**  
   - Séparation en train/test (`train_test_split`)  
   - Standardisation ou scaling des features si nécessaire  

3. **Modélisation**  
   - Test de modèle de régression :  
     - **XGBoost Regressor**  

4. **Évaluation du modèle**  
   - Calcul du **Mean Squared Error (MSE)** et du **R² score**  
   - Comparaison des performances des modèles  

5. **Visualisation des résultats**  
   - Graphique comparatif entre **prix réel vs prix prédit**  

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Prix réel")
plt.ylabel("Prix prédit")
plt.title("Comparaison prix réel vs prix prédit")
plt.show()

Installation

Pour exécuter le projet, installer les dépendances :

pip install -r requirements.txt


Cloner le repo :

git clone https://github.com/NEIMEEE/ML.git
cd ML/HousePricePrediction
Le notebook .ipynb contient tout le code pour l’entraînement et la prédiction.
