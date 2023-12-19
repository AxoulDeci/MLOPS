def importation():
    """
    Importe les données du fichier CSV 'adult.csv' dans un DataFrame Pandas.

    Returns:
    --------
    pandas.DataFrame:
        Le DataFrame contenant les données du fichier CSV. Les valeurs '?' sont traitées comme valeurs manquantes.
    """
    df = pd.read_csv("adult.csv", na_values='?')
    return df
