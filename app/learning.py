from sklearn.model_selection import train_test_split


def select_product(data, product_name, produtos, include_other_products=False):
    produtos = produtos
    produto = data[product_name]
    if include_other_products:
        data = data.drop(product_name, axis=1)
    else:
        data = data.drop(produtos, axis=1)
    return data, produto


# Helper pra criar test e train set pra um produto
def train_test_sets(df, product, produtos, test_size=0.2, include_other_products=False):
    data, p = select_product(df, product, produtos, include_other_products)
    return train_test_split(data, p, test_size=test_size)


def get_sets(data, product_name, produtos, test_size=0.2, include_other_products=False):
    """X_train, X_test, y_train, y_test"""
    return train_test_sets(data, product_name, produtos, test_size, include_other_products)
