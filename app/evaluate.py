import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from app.learning import select_product


def confusion_test(data_test, label_test, fitted_pipeline):
    predicted = fitted_pipeline.predict(data_test)

    print()
    print(classification_report(label_test, predicted))
    plt.matshow(confusion_matrix(predicted, label_test), cmap=plt.cm.binary, interpolation='nearest')
    plt.colorbar()
    plt.ylabel('expected label')
    plt.xlabel('predicted label')
    plt.show()
    return


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=10, train_sizes=np.linspace(.1, 1.0, 5)):
    print("Creating figure")
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    print("Creating learning curve")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes)
    print("Processing results")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print("Creating grid")
    plt.grid()
    print("Filling grid")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    print("Plotting")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def product_learning_curves(data, product, produtos, title, pipeline, include_other_products=False, cv=10):
    data_vector, label_vector = select_product(data, product, produtos, include_other_products=False)
    plot_learning_curve(pipeline, "accuracy vs. training set size ({})".format(title), data_vector, label_vector, cv=cv).show()
