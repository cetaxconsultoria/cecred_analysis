from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib


class ColumnsSelector(BaseEstimator, TransformerMixin):
    """Selects a column from the data passed (as a list)"""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        return dataframe[self.columns]


class CecredModel:

    dummy_columns = ['COOP_NMRESCOP',
                     'COOP_SKCOOPERATIVA',
                     'COOP_NRENDCOP',
                     'DSDRISGP',
                     'NMCIDADE',
                     'DSGRUPORENDAFAT',
                     'DSMOTIDEM',
                     'DSSITDCT',
                     'DSSITDTL',
                     'DSTIPCTA',
                     'CDDSCNAE',
                     'QTD_TOTAL_PROD',
                     'DSTIPOVINCULACAO'
                     ]

    not_dummy_columns = ['DSINADIMPL', 'DSRESTR',
                         'DSSPCOUTRASINST', 'FLGCIDATU', 'CONTRATOS_PREJUIZO']

    def __init__(self, product=None, file=None):

        self.pipeline = Pipeline([
            # Use FeatureUnion to combine the processed
            ('union', FeatureUnion(
                transformer_list=[

                    # Pipeline for scaling and preprocessing not dummified features
                    ('not_dummy', Pipeline([
                        ('selector', ColumnsSelector(columns=self.not_dummy_columns)),
                        ('scaler', StandardScaler()),
                    ])),

                    # Pipeline for scaling and preprocessing dummified features
                    ('dummy', Pipeline([
                        ('selector', ColumnsSelector(columns=self.dummy_columns)),
                        ('scaler', StandardScaler(with_mean=False, with_std=False)),
                    ])),
                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'not_dummy': 1,
                    'dummy': 1,
                },
            )),
            # Apply a voting classifier as ensemble method with 3 classifiers
            ('ensemble', VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(C=1, fit_intercept=True,
                                              intercept_scaling=1, solver='saga', tol=0.01)),
                    ('mlp', MLPClassifier(early_stopping=True, hidden_layer_sizes=(
                        40, 30, 5), learning_rate='adaptive')),
                    ('lsv', LinearSVC(C=1, fit_intercept=True,
                                      intercept_scaling=1e-05, tol=0.01)),
                ],
                voting='hard'
            ))
        ])

        self.product = product
        self.file = file or "models/{}.pkl".format(product)
        self.trained = False

    def fit(self, X, y, save=True):
        self.trained = True
        self.pipeline.fit(X, y)
        if save and self.product:
            self.save_model()
        return self

    def predict(self, vector):
        if not self.trained and self.product:
            self.load_model()
            self.trained = True
        return self.pipeline.predict(vector)

    def predict_proba(self, vector):
        if not self.trained and self.product:
            self.load_model()
            self.trained = True
        return self.pipeline.predict_proba(vector)

    def transform(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def fit_transform(self, fit_vector, transform_vector):
        self.fit(fit_vector)
        return self.transform(transform_vector)

    def save_model(self):
        joblib.dump(self.pipeline, self.file)

    def load_model(self):
        self.pipeline = joblib.load(self.file)
