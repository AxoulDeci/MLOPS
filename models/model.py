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
    Entraîne un classifieur d'arbre de décision en utilisant une recherche sur grille pour ajuster les hyperparamètres.

    Returns:
    Modèle Tree

    Cette fonction utilise le classifieur d'arbre de décision de scikit-learn avec une recherche sur grille
    pour ajuster les hyperparamètres tels que 'max_depth', 'min_samples_leaf' et 'min_samples_split'.
    Les résultats du meilleur modèle sont stockés dans l'objet GridSearchCV.

    Assurez-vous que les données d'entraînement et les étiquettes sont préalablement définies dans les variables
    globales X_train_encoded et y_train.

    Exemple :
    fit_tree()

    Note :
    Cette fonction utilise scikit-learn, donc assurez-vous que le module est installé dans votre environnement.
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
    Entraîne un classifieur de forêt aléatoire (Random Forest) sur les données d'entraînement.

    Returns:
    None

    Cette fonction utilise le classifieur de forêt aléatoire de scikit-learn pour entraîner un modèle
    sur les données d'entraînement préalablement encodées. Assurez-vous que les données d'entraînement et
    les étiquettes sont définies dans les variables globales X_train_encoded et y_train.

    Exemple :
    fit_forest()

    Note :
    Cette fonction nécessite que le module scikit-learn soit installé dans votre environnement.
    Assurez-vous d'avoir préalablement effectué l'encodage des données et défini les variables globales.
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
    Entraîne un classifieur AdaBoost sur les données d'entraînement.

    Returns:
    None

    Cette fonction utilise le classifieur AdaBoost de scikit-learn pour entraîner un modèle
    sur les données d'entraînement préalablement encodées. Assurez-vous que les données d'entraînement et
    les étiquettes sont définies dans les variables globales X_train_encoded et y_train.

    Exemple :
    fit_ada()

    Note :
    Cette fonction nécessite que le module scikit-learn soit installé dans votre environnement.
    Assurez-vous d'avoir préalablement effectué l'encodage des données et défini les variables globales.
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
    Entraîne un classifieur k-NN (k plus proches voisins) sur les données d'entraînement.

    Returns:
    None

    Cette fonction utilise le classifieur k-NN de scikit-learn pour entraîner un modèle
    sur les données d'entraînement préalablement encodées. Assurez-vous que les données d'entraînement et
    les étiquettes sont définies dans les variables globales X_train_encoded et y_train.

    Exemple :
    fit_knn()

    Note :
    Cette fonction nécessite que le module scikit-learn soit installé dans votre environnement.
    Assurez-vous d'avoir préalablement effectué l'encodage des données et défini les variables globales.
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
    Entraîne un classifieur MLP (Perceptron multi-couches) sur les données d'entraînement.

    Returns:
    None

    Cette fonction utilise le classifieur MLP de scikit-learn pour entraîner un modèle
    sur les données d'entraînement préalablement encodées. Assurez-vous que les données d'entraînement et
    les étiquettes sont définies dans les variables globales X_train_encoded et y_train.

    Exemple :
    fit_mlp()

    Note :
    Cette fonction nécessite que le module scikit-learn soit installé dans votre environnement.
    Assurez-vous d'avoir préalablement effectué l'encodage des données et défini les variables globales.
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
    Entraîne un classifieur SVM (Support Vector Machine) sur les données d'entraînement.

    Returns:
    None

    Cette fonction utilise le classifieur SVM de scikit-learn pour entraîner un modèle
    sur les données d'entraînement préalablement encodées. Assurez-vous que les données d'entraînement et
    les étiquettes sont définies dans les variables globales X_train_encoded et y_train.

    Exemple :
    fit_svc()

    Note :
    Cette fonction nécessite que le module scikit-learn soit installé dans votre environnement.
    Assurez-vous d'avoir préalablement effectué l'encodage des données et défini les variables globales.
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
    Entraîne un classifieur de régression logistique sur les données d'entraînement.

    Returns:
    None

    Cette fonction utilise le classifieur de régression logistique de scikit-learn pour entraîner un modèle
    sur les données d'entraînement préalablement encodées. Assurez-vous que les données d'entraînement et
    les étiquettes sont définies dans les variables globales X_train_encoded et y_train.

    Exemple :
    fit_log()

    Note :
    Cette fonction nécessite que le module scikit-learn soit installé dans votre environnement.
    Assurez-vous d'avoir préalablement effectué l'encodage des données et défini les variables globales.
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
    Évalue les performances d'un modèle de classification.

    Parameters:
    - model: Le modèle préalablement entraîné à évaluer.

    Returns:
    None

    Cette fonction évalue les performances du modèle sur les ensembles d'apprentissage et de test.
    Elle affiche l'accuracy, la précision, le rappel, le score F1, et l'aire sous la courbe ROC (AUC).

    Assurez-vous que les données d'apprentissage et de test ainsi que les étiquettes sont définies dans
    les variables globales X_train_encoded, y_train, X_test_encoded et y_test.

    Exemple :
    model_score(model)

    Note :
    Cette fonction nécessite que le module scikit-learn soit installé dans votre environnement.
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


def matrice_confusion(model, data):
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
    

def courbe_roc(model, data):
    
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


def features_importances(forest, data):

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