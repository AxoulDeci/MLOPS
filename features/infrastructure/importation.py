def importation():
    """
    Importe les données du fichier CSV 'adult.csv' dans un DataFrame Pandas.

    Returns:
    --------
    pandas.DataFrame:
        Le DataFrame contenant les données du fichier CSV. Les valeurs '?' sont traitées comme valeurs manquantes.
    """
    import pandas as pd
    df = pd.read_csv("../data/raw/adult.csv", na_values='?')
    return df

df=importation()