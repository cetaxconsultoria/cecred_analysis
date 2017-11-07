import numpy as np
import pandas as pd
from math import sqrt
import scipy.spatial as spsp


class SVDRecommender(object):
    """
    Singular Value Decomposition is an important technique used in recommendation systems.
    Using SVD, the complete utility matrix is decomposed into user and item features.
    Thus the dimensionality of the matrix is reduced and we get the most important features
    neglecting the weaker ones.

    The utility matrix is initially sparse having a lot of missing values. The missing values
    are filled in using the mean for that item.

    n_features: the number of the biggest features that are to be taken for each user and
                    item. default value is 15.

    method: 1. default: The mean for the item is deducted from the user-item pair value in
                        the utility matrix. SVDRecommender is computed. With the computed values, the
                        mean for the item is added back to get the final result.
    """

    def __init__(self,
                 n_features=15,
                 method='default',
                 ):
        self.parameters = {"n_features", "method"}
        self.method = method
        self.no_of_features = n_features

    def get_params(self, deep=False):
        out = {}
        for param in self.parameters:
            out[param] = getattr(self, param)

        return out

    def set_params(self, **params):

        for a in params:
            if a in self.parameters:
                setattr(self, a, params[a])
            else:
                raise AttributeError("No such attribute exists to be set")

    def fit(self, user_item_matrix):
        self.users = user_item_matrix.index.values
        self.items = user_item_matrix.columns.values

        self.user_index = {k: i for i, k in enumerate(self.users)}
        self.item_index = {k: i for i, k in enumerate(self.items)}

        mask = pd.isnull(user_item_matrix)
        masked_arr = np.ma.masked_array(user_item_matrix, mask)

        self.predMask = ~mask
        self.item_means = np.mean(masked_arr, axis=0)
        self.user_means = np.mean(masked_arr, axis=1)
        self.item_means_tiled = np.tile(
            self.item_means, (user_item_matrix.shape[0], 1))

        # utility matrix or ratings matrix that can be fed to svd
        self.utilMat = masked_arr.filled(self.item_means)

        # for the default method
        if self.method == 'default':
            self.utilMat = self.utilMat - self.item_means_tiled

        # Singular Value Decomposition starts
        # k denotes the number of features of each user and item
        # the top matrices are cropped to take the greatest k rows or
        # columns. U, V, s are already sorted descending.

        k = self.no_of_features
        U, s, V = np.linalg.svd(self.utilMat, full_matrices=False)

        U = U[:, 0:k]
        V = V[0:k, :]
        s_root = np.diag([sqrt(s[i]) for i in range(0, k)])

        self.Usk = np.dot(U, s_root)
        self.skV = np.dot(s_root, V)
        self.UsV = np.dot(self.Usk, self.skV)

        self.UsV = self.UsV + self.item_means_tiled
        return self

    def predict(self, X, formatizer={'user': 0, 'item': 1}):
        """Takes an array with 2 columns. The first
        represents user names and the second item names
        """

        users = X.ix[:, formatizer['user']].tolist()
        items = X.ix[:, formatizer['item']].tolist()

        if self.method == 'default':

            values = []
            for i in range(len(users)):
                user = users[i]
                item = items[i]

                # user and item in the test set may not always occur in the train set. In these cases
                # we can not find those values from the utility matrix.
                # That is why a check is necessary.
                # 1. both user and item in train
                # 2. only user in train
                # 3. only item in train
                # 4. none in train

                if user in self.user_index:
                    if item in self.item_index:
                        values.append(
                            self.UsV[self.user_index[user], self.item_index[item]])
                    else:
                        values.append(self.user_means[self.user_index[user]])

                elif item in self.item_index and user not in self.user_index:
                    values.append(self.item_means[self.item_index[item]])

                else:
                    values.append(np.mean(self.item_means) *
                                  0.6 + np.mean(self.user_means) * 0.4)

        return values

    def topN_similar(self, x, column='item', N=10):
        """
        Gives out the most similar contents compared to the input content given. For an user input gives out similar
        users. For an item input, gives out the most similar items.

        :param x: the identifier string for the user or item.
        :param column: either 'user' or 'item'
        :param N: The number of best matching similar content to output

        :return: A list of tuples.
        """
        out = list()

        if column == 'user':
            if x not in self.user_index:
                raise Exception("Invalid user")
            else:
                for user in self.user_index:
                    if user != x:
                        temp = spsp.distance.euclidean(
                            self.Usk[self.user_index[user], :], self.Usk[self.user_index[x], :])
                        out.append((self.users[self.user_index[user]], temp))
        if column == 'item':
            if x not in self.item_index:
                raise Exception("Invalid item")
            else:
                for item in self.item_index:
                    if item != x:
                        temp = spsp.distance.euclidean(
                            self.skV[:, self.item_index[item]], self.skV[:, self.item_index[x]])
                        out.append((self.items[self.item_index[item]], temp))

        out.sort(key=lambda x: x[1])
        out = out[:N]
        return out

    def recommend(self, content_list, content='user', N=10, values=False):

        # utilMat element not zero means that element has already been
        # discovered by the user and can not be recommended
        predMat = np.ma.masked_where(
            self.predMask, self.UsV).filled(fill_value=-999)
        out = []

        if content == 'user':
            if values is True:
                for user in content_list:
                    try:
                        j = self.user_index[user]
                    except KeyError:
                        raise Exception("Invalid user:", user)
                    max_indices = predMat[j, :].argsort()[-N:][::-1]
                    out.append([(self.items[index], predMat[j, index])
                                for index in max_indices])

            else:
                for user in content_list:
                    try:
                        j = self.user_index[user]
                    except KeyError:
                        raise Exception("Invalid user:", user)
                    max_indices = predMat[j, :].argsort()[-N:][::-1]
                    out.append([self.items[index] for index in max_indices])

        elif content == 'item':
            if values is True:
                for item in content_list:
                    try:
                        j = self.item_index[item]
                    except KeyError:
                        raise Exception("Invalid item:", item)
                    max_indices = predMat[:, j].argsort()[-N:][::-1]
                    out.append([(self.users[index], predMat[index, j])
                                for index in max_indices])

            else:
                for item in content_list:
                    try:
                        j = self.item_index[item]
                    except KeyError:
                        raise Exception("Invalid item:", item)
                    max_indices = predMat[:, j].argsort()[-N:][::-1]
                    out.append([self.users[index] for index in max_indices])

        return out

    def __str__(self):
        return "SVDRecommender. features: {}, method:{}".format(self.no_of_features, self.method)

    def __repr__(self):
        return str(self)
