import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

## 1. Données manquantes 

# 1.1 Détection des données manquantes

def donnees_manquantes(data):
    """
    Calcule le nombre de valeurs manquantes par colonne dans le DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        Le DataFrame contenant les données.

    Returns:
    --------
    pandas.Series:
        Une série contenant le nombre de valeurs manquantes pour chaque colonne.
    """
    missing_values_per_column = data.isnull().sum()
    return missing_values_per_column

## 1.2 Traitement des données manquantes 

def traiter_donnees_manquantes(data):
    """
    Remplace les valeurs manquantes dans les colonnes spécifiques du DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        Le DataFrame contenant les données.

    Returns:
    --------
    pandas.DataFrame:
        Le DataFrame avec les valeurs manquantes traitées.
    """
    # Remplacez les valeurs manquantes par la valeur "no_info" pour les variables "workclass" et "occupation"
    data['workclass'].fillna("no_info", inplace=True)
    data['occupation'].fillna("no_info", inplace=True)

    # Remplacez les valeurs manquantes dans la colonne "native-country" par le mode
    mode_native_country = data['native-country'].mode()[0]
    data['native-country'].fillna(mode_native_country, inplace=True)

    return data

## 2. Détection des données aberrantes par boxplot pour les variables numériques

def detecter_donnees_aberrantes_par_boxplot(data):
    """
    Crée un boxplot pour chaque colonne numérique du DataFrame afin de détecter les valeurs aberrantes.

    Parameters:
    -----------
    data : pandas.DataFrame
        Le DataFrame contenant les données.

    Returns:
    --------
    None
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    # Sélectionnez les colonnes numériques
    colonnes_numeriques = data.select_dtypes(include=['float64', 'int64'])

    # Tracez un boxplot pour chaque colonne numérique
    for colonne in colonnes_numeriques:
        plt.figure()  # Crée une nouvelle figure pour chaque boxplot
        sns.boxplot(data=data[colonne])
        plt.title(f'Boxplot de {colonne}')  # Utilisation du nom de la colonne comme titre
        plt.show()  # Affiche le boxplot

## 3. Détection des données aberrantes par boxplot pour les variables numériques

def regroupement(data):
    """
    Regroupe certaines variables du DataFrame en créant de nouvelles catégories.

    Parameters:
    -----------
    data : pandas.DataFrame
        Le DataFrame contenant les données.

    Returns:
    --------
    pandas.DataFrame:
        Le DataFrame avec les nouvelles catégories créées.
    """
    import numpy as np
    # 3.1 Variable 'age'
    def recoder_age(data):
        conditions = [
            (data['age'] <= 28),
            (data['age'] > 28) & (data['age'] <= 39),
            (data['age'] > 39) & (data['age'] <= 49),
            (data['age'] >= 50)
        ]

        choices = ['<=28 ans', '29_39 ans', '40_49 ans', '>=50 ans']

        data['Age'] = np.select(conditions, choices, default='')

        return data

    recoder_age(data)

    # 3.2 Variable 'hours-per-week'
    def recoder_heures_semaine(data):
        conditions_hours = [
            (data['hours-per-week'] <= 40),
            (data['hours-per-week'] > 40) & (data['hours-per-week'] <= 46),
            (data['hours-per-week'] > 46)
        ]

        choices_hours = ['<=40H', '40_46H', '>46H']

        data['Heures_semaine'] = np.select(conditions_hours, choices_hours, default='')

        return data

    recoder_heures_semaine(data)

    # 3.3 Variable 'native-country'
    def recoder_native_country(data):
        data['native-country'] = data['native-country'].str.strip()

        def group_countries(country):
            if country == 'United-States':
                return 'United_States'
            elif pd.notnull(country):
                return 'autres_pays'
            else:
                return country

        data['country_group'] = data['native-country'].apply(group_countries)

        return data 

    recoder_native_country(data)

    # 3.4 Variable 'workclass'
    def recoder_emploi_secteur(data):
        def emploi_secteur(workclass):
            if workclass in ('Federal-gov', 'Local-gov', 'State-gov'):
                return 'Gouvernement'
            elif workclass in ('Never-worked', 'Without-pay'):
                return 'Sans_emploi'
            else:
                return 'Prive'

        data['Emploi_Secteur'] = data['workclass'].apply(emploi_secteur)

        return data

    recoder_emploi_secteur(data)

    # 3.5 Variable 'marital-status'
    def recoder_marital_status(data):
        def recoder_status(status):
            if status in ('Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent'):
                return 'Marie'
            elif status == 'Never-married':
                return 'Jamais_marie'
            else:
                return 'Ex_Marie'

        data['Marital_status'] = data['marital-status'].apply(recoder_status)

        return data

    recoder_marital_status(data)

    # 3.6 Variable 'education'
    def recoder_education(data):
        conditions_education = [
            data['education'].isin(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th']),
            data['education'].isin(['10th', '11th', '12th', 'HS-grad']),
            data['education'].isin(['Some-college', 'Assoc-acdm', 'Assoc-voc']),
        ]

        choices_education = ['Primaire_Second', 'Lycee', 'Sup']

        data['Education'] = np.select(conditions_education, choices_education, default='Sup_plus')

        return data

    recoder_education(data)

    # 3.7 Variable 'occupation'
    def recoder_occupation(data):
        conditions = [
            data['occupation'].isin(['Adm-clerical', 'Exec-managerial', 'Prof-specialty', 'Tech-support']),
            data['occupation'].isin(['Armed-Forces', 'Protective-serv']),
            data['occupation'].isin(['Craft-repair', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct']),
            data['occupation'].isin(['No_information', 'Other-service']),
            data['occupation'].isin(['Priv-house-serv', 'Sales']),
            data['occupation'] == 'Transport-moving'
        ]

        choices = ['Administration', 'Armee_defense', 'Artisan_Reparation', 'Autre_service', 'Ventes_Services', 'Transport']

        data['Occupation'] = np.select(conditions, choices, default='Autre')

        return data

    recoder_occupation(data)

    # 3.8 Variable 'relationship'
    def recoder_relationship(data):
        conditions_relationship = [
            (data['relationship'] == 'Husband') | (data['relationship'] == 'Wife')
        ]

        choices_relationship = ['Conjoint']

        data['Relationship'] = np.where(conditions_relationship[0], choices_relationship[0], 'Autre')

        return data

    recoder_relationship(data)

    return data

# 4. Centre et réduit les variables "capital-loss" et "capital-gain" en une variable "capital"

def traitement_capital(data):
    """
    Traite la variable 'capital' en créant une nouvelle colonne et standardise les valeurs.

    Parameters:
    -----------
    data : pandas.DataFrame
        Le DataFrame contenant les données.

    Returns:
    --------
    pandas.DataFrame:
        Le DataFrame avec la variable 'capital' traitée.
    """
    from sklearn.preprocessing import StandardScaler

    # Création de la nouvelle colonne 'capital'
    data['capital'] = data['capital-gain'] - data['capital-loss']

    # Remplacement de la valeur 99999 par NaN
    data['capital'].replace(99999, np.nan, inplace=True)

    # Suppression des lignes avec des valeurs manquantes dans la colonne 'capital'
    data.dropna(subset=['capital'], inplace=True)

    # Standardisation de la colonne 'capital' avec StandardScaler
    scaler = StandardScaler()
    data_to_scale = data[['capital']]
    data['capital'] = scaler.fit_transform(data_to_scale)

    return data

# 5. Supprimer des colonnes ou on a fait des regroupements

def supprimer_colonnes(data):
    """
    Supprime les colonnes spécifiées du DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        Le DataFrame contenant les données.

    Returns:
    --------
    pandas.DataFrame:
        Le DataFrame avec les colonnes spécifiées supprimées.
    """
    # Colonnes à supprimer
    colonnes_a_supprimer = ['occupation', 
    'hours-per-week',
    'relationship',
    'native-country',
    'marital-status',
    'education',
    'age',
    'workclass',
    'capital-gain',
    'capital-loss',
    'country_group',
    'Emploi_Secteur',
    'race',
    'fnlwgt',
    'educational-num']

    # Suppression des colonnes spécifiées
    data.drop(columns=colonnes_a_supprimer, inplace=True)

    return data



