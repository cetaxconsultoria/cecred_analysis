import pandas as pd
from collections import Counter
import pickle


def get_data(file, sep="|", low_memory=False):
    return pd.read_csv(file, sep=sep, low_memory=low_memory)


def clean_data(data,
               remove_anormal=True,
               useless_columns=["AGENCIA_NMBAIRRO", "NRENDERE", "QTFUNCIO", "DSSEXO",
                                "DSPROFTL", "NMBAIRRO", "AGENCIA_DSENDCOP", "DSENDERE",
                                "AGENCIA_NMRESAGE", "AGENCIA_NMCIDADE"],
               infiltrated_labels=['CARTAOCRED', 'COBRANCARIA', 'CONSORCIO', 'EMPRESFINAN',
                                   'CESSINTERNET', 'DEBITOAUT', 'LMTDSTTIT', 'LMTCHESPECIAL',
                                   'DDA', 'LMTDSRCHEQ', 'POUPPROG', 'DOMICBANCARIO', 'APLICACAO',
                                   'PLANOCOTAS', 'UTILCOBRANCA', 'LMTTRANSACAO', 'CONVFOLHAPAGTO',
                                   'LMTTRANSACAO'],
               remove_redundant=True,
               remove_product_value=True,
               dummify_categorical=True,
               booleanfy=True,
               remove_total_amount=True,
               load=True,
               ):

    count_vector = {}
    for column in data.columns.values:
        count_vector[column] = Counter(data[column])

    # Colunas com apenas 1 tipo de dado
    redundant_columns = []

    # Colunas que representam o valor de um produto
    product_value_columns = []

    # Colunas que devem ser dummified
    dummy_columns = []

    # Para cada coluna no count_vector
    for k, v in count_vector.items():

        # ignora colunas inúteis e labels infiltradas
        if k in useless_columns or k in infiltrated_labels:
            continue
        # Encontra redundantes
        elif len(v) == 1 and remove_redundant:
            redundant_columns.append(k)
            # Encontra colunas de valores de produto
        elif k.startswith("VL") and remove_product_value:
            product_value_columns.append(k)
            # dummify
        elif len(v) > 2 and (not k.startswith("VL") and not k.startswith("QTDE")) and dummify_categorical:
            dummy_columns.append(k)
            #  booleanfy boolean columns (1 == most common)
        elif len(v) == 2 and booleanfy:
            if not (v.most_common()[0][0] == 0 or v.most_common()[0][0] == 1) or not (v.most_common()[1][0] == 0 or v.most_common()[1][0] == 1):
                data[k] = data[k].map(
                    {v.most_common()[0][0]: 1, v.most_common()[1][0]: 0})

    # Remove as colunas indicadas
    data = data.drop(redundant_columns, axis=1)

    data = data.drop(product_value_columns, axis=1)

    data = data.drop(useless_columns, axis=1)

    data = data.drop(infiltrated_labels, axis=1)

    data = pd.get_dummies(data, columns=dummy_columns)

    try:
        data = data.drop(["QTD_TOTAL_PROD"],
                         axis=1) if remove_total_amount else data
    except:
        pass

    return data


def get_cleaned_data(data='data/datasets/raw.csv', cached_data='data/datasets/cached/raw.pkl', cache=True):
    if cache:
        try:
            with open(cached_data, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            pass
    data = get_data(data)
    data = clean_data(data)
    if cache:
        with open(cached_data, 'wb') as file:
            pickle.dump(data, file)
    return data


def get_produtos(data):
    return list(filter(lambda x: x.startswith("QTDE"), data.columns.values))
