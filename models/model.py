from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np


#Entrainement du modèle DecisionTreeClassifier

def fit_tree(data):
    """
    Entraine un classificateur d'arbre de décision sur les données fournies.

    Parameters:
    - data (DataFrame): Le DataFrame contenant les données d'entraînement.

    Returns:
    - best_estimator (DecisionTreeClassifier): Le meilleur estimateur après la recherche sur grille.
    """
    x = data.drop(['income'], axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        random_state=0, 
                                                        shuffle=True, 
                                                        stratify=y)
    
    cat_columns = ['Age','Heures_semaine','Education','Marital_status','Relationship','gender','Occupation']
    num_columns = ['capital']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_columns),
            ('num', 'passthrough', num_columns)
        ],
        remainder='drop'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.fit_transform(X_test)
    
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier()
    params={
     "max_depth" : [1, 2, 3, 4],
     "min_samples_leaf" : [3, 5, 7, 9, 11],
     "min_samples_split" : [8, 9, 10, 11, 12 ,13, 14]
     }

    tree = GridSearchCV(tree, params,refit=True, n_jobs=4, cv=5)
    tree.fit(X_train_encoded, y_train)
    print("DecisionTreeClassifier a été entrainé")
    return tree.best_estimator_

#Entrainement du modèle RandomForestClassifier

def fit_forest(data):
    """
    Entraine un classificateur de forêt aléatoire sur les données fournies.

    Parameters:
    - data (DataFrame): Le DataFrame contenant les données d'entraînement.

    Returns:
    - forest (RandomForestClassifier): Le classificateur de forêt aléatoire entraîné.
    """
    x = data.drop(['income'], axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        random_state=0, 
                                                        shuffle=True, 
                                                        stratify=y)
    
    cat_columns = ['Age','Heures_semaine','Education','Marital_status','Relationship','gender','Occupation']
    num_columns = ['capital']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_columns),
            ('num', 'passthrough', num_columns)
        ],
        remainder='drop'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.fit_transform(X_test)

    forest=RandomForestClassifier()
    forest.fit(X_train_encoded, y_train)
    print("RandomForestClassifier a été entrainé")
    return forest


#Entrainement du modèle AdaBoostClassifier

def fit_ada(data):
    """
    Entraine un classificateur AdaBoost sur les données fournies.

    Parameters:
    - data (DataFrame): Le DataFrame contenant les données d'entraînement.

    Returns:
    - ada (AdaBoostClassifier): Le classificateur AdaBoost entraîné.
    """
    x = data.drop(['income'], axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        random_state=0, 
                                                        shuffle=True, 
                                                        stratify=y)
    
    cat_columns = ['Age','Heures_semaine','Education','Marital_status','Relationship','gender','Occupation']
    num_columns = ['capital']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_columns),
            ('num', 'passthrough', num_columns)
        ],
        remainder='drop'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.fit_transform(X_test)

    ada=AdaBoostClassifier()
    ada.fit(X_train_encoded, y_train)
    print("AdaBoostClassifier a été entrainé")
    return ada


#Entrainement du modèle KNeighborsClassifier

def fit_knn(data):
    """
    Entraîne un classificateur k-NN (k plus proches voisins) sur les données fournies.

    Parameters:
    - data (DataFrame): Le DataFrame contenant les données d'entraînement.

    Returns:
    - knn (KNeighborsClassifier): Le classificateur k-NN entraîné.
    """
    x = data.drop(['income'], axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        random_state=0, 
                                                        shuffle=True, 
                                                        stratify=y)
    
    cat_columns = ['Age','Heures_semaine','Education','Marital_status','Relationship','gender','Occupation']
    num_columns = ['capital']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_columns),
            ('num', 'passthrough', num_columns)
        ],
        remainder='drop'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.fit_transform(X_test)

    knn=KNeighborsClassifier()
    knn.fit(X_train_encoded, y_train)
    print("KNeighborsClassifier a été entrainé")
    return knn


#Entrainement du modèle MLPClassifier

def fit_mlp(data):
    """
    Entraîne un classificateur MLP (Multi-Layer Perceptron) sur les données fournies.

    Parameters:
    - data (DataFrame): Le DataFrame contenant les données d'entraînement.

    Returns:
    - mlp (MLPClassifier): Le classificateur MLP entraîné.
    """
    x = data.drop(['income'], axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        random_state=0, 
                                                        shuffle=True, 
                                                        stratify=y)
    
    cat_columns = ['Age','Heures_semaine','Education','Marital_status','Relationship','gender','Occupation']
    num_columns = ['capital']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_columns),
            ('num', 'passthrough', num_columns)
        ],
        remainder='drop'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.fit_transform(X_test)

    mlp=MLPClassifier()
    mlp.fit(X_train_encoded, y_train)
    print("MLPClassifier a été entrainé")
    return mlp


#Entrainement du modèle SVC

def fit_svc(data):
    """
    Entraîne un classificateur SVM (Support Vector Machine) sur les données fournies.

    Parameters:
    - data (DataFrame): Le DataFrame contenant les données d'entraînement.

    Returns:
    - svc (SVC): Le classificateur SVM entraîné.
    """
    x = data.drop(['income'], axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        random_state=0, 
                                                        shuffle=True, 
                                                        stratify=y)
    
    cat_columns = ['Age','Heures_semaine','Education','Marital_status','Relationship','gender','Occupation']
    num_columns = ['capital']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_columns),
            ('num', 'passthrough', num_columns)
        ],
        remainder='drop'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.fit_transform(X_test)

    svc=SVC()
    svc.fit(X_train_encoded, y_train)
    print("SVC a été entrainé")
    return svc


#Entrainement du modèle LogisticRegression

def fit_log(data):
    """
    Entraîne un classificateur de régression logistique sur les données fournies.

    Parameters:
    - data (DataFrame): Le DataFrame contenant les données d'entraînement.

    Returns:
    - log (LogisticRegression): Le classificateur de régression logistique entraîné.
    """
    x = data.drop(['income'], axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        random_state=0, 
                                                        shuffle=True, 
                                                        stratify=y)
    
    cat_columns = ['Age','Heures_semaine','Education','Marital_status','Relationship','gender','Occupation']
    num_columns = ['capital']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_columns),
            ('num', 'passthrough', num_columns)
        ],
        remainder='drop'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.fit_transform(X_test)

    log=LogisticRegression()
    log.fit(X_train_encoded, y_train)
    print("LogisticRegression a été entrainé")
    return log


#Score du modèle

def model_score(model, data):
    """
    Évalue les performances d'un modèle sur les ensembles d'apprentissage et de test.

    Parameters:
    - model: Le modèle entraîné à évaluer.
    - data (DataFrame): Le DataFrame contenant les données.

    Returns:
    None
    """
    x = data.drop(['income'], axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        random_state=0, 
                                                        shuffle=True, 
                                                        stratify=y)
    
    cat_columns = ['Age','Heures_semaine','Education','Marital_status','Relationship','gender','Occupation']
    num_columns = ['capital']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_columns),
            ('num', 'passthrough', num_columns)
        ],
        remainder='drop'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.fit_transform(X_test)

    print("Accuracy apprentissage: ",model.score(X_train_encoded, y_train))

    y_pred_model = model.predict(X_test_encoded)

    print("Accuracy test: ",accuracy_score(y_test, y_pred_model))
    print("Precision:", precision_score(y_test, y_pred_model, pos_label='>50K'))
    print("Recall:", recall_score(y_test, y_pred_model, pos_label='>50K'))
    print("F1:", f1_score(y_test, y_pred_model, pos_label='>50K'))

    le = LabelEncoder()
    y_test_numeric = le.fit_transform(y_test)
    y_pred_numeric = le.transform(y_pred_model)

    auc_score = roc_auc_score(y_test_numeric, y_pred_numeric)
    print("AUC:", auc_score)

    return None


#Matrice de confusion

def matrice_confusion(model, data):
    """
    Affiche la matrice de confusion du modèle sur l'ensemble de test.

    Parameters:
    - model: Le modèle entraîné.
    - data (DataFrame): Le DataFrame contenant les données.

    Returns:
    - cm_display: L'objet ConfusionMatrixDisplay prêt à être affiché.
    """
    x = data.drop(['income'], axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        random_state=0, 
                                                        shuffle=True, 
                                                        stratify=y)
    
    cat_columns = ['Age','Heures_semaine','Education','Marital_status','Relationship','gender','Occupation']
    num_columns = ['capital']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_columns),
            ('num', 'passthrough', num_columns)
        ],
        remainder='drop'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.fit_transform(X_test)

    y_pred_model = model.predict(X_test_encoded)

    return print("Arbre", ConfusionMatrixDisplay.from_predictions(y_test,y_pred_model,
                                                                  cmap=plt.cm.Blues,
                                                                  display_labels=["<=50K",">50K"]))
    

#Courbe de roc

def courbe_roc(model, data):
    """
    Affiche la courbe ROC du modèle sur l'ensemble de test.

    Parameters:
    - model: Le modèle entraîné.
    - data (DataFrame): Le DataFrame contenant les données.

    Returns:
    - roc_display: L'objet RocCurveDisplay prêt à être affiché.
    """
    x = data.drop(['income'], axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        random_state=0, 
                                                        shuffle=True, 
                                                        stratify=y)
    
    cat_columns = ['Age','Heures_semaine','Education','Marital_status','Relationship','gender','Occupation']
    num_columns = ['capital']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_columns),
            ('num', 'passthrough', num_columns)
        ],
        remainder='drop'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.fit_transform(X_test)

    y_pred_model = model.predict(X_test_encoded)
    
    return RocCurveDisplay.from_estimator(model, X_test_encoded, y_test, name=model)


#Features importances de RandomForest

def features_importances(forest, data):
    """
    Affiche les importances des caractéristiques d'un modèle sur l'ensemble de test.

    Parameters:
    - model: Le modèle entraîné.
    - data (DataFrame): Le DataFrame contenant les données.

    Returns:
    None
    """
    x = data.drop(['income'], axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        random_state=0, 
                                                        shuffle=True, 
                                                        stratify=y)
    
    cat_columns = ['Age','Heures_semaine','Education','Marital_status','Relationship','gender','Occupation']
    num_columns = ['capital']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_columns),
            ('num', 'passthrough', num_columns)
        ],
        remainder='drop'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.fit_transform(X_test)

    forest=RandomForestClassifier()
    forest.fit(X_train_encoded, y_train)

    feature_importances = forest.feature_importances_

    feature_names = np.array(preprocessor.get_feature_names_out())

    indices = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), feature_importances[indices], align='center')
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.show()

    return None