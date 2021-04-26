import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


def main():
    # the goal is to predict the category of iris based on it's features

    # reading data to a dataframe
    iris = pd.read_csv('iris.csv')

    # pairplot of the data set
    sns.pairplot(iris, hue = 'species', palette = 'Dark2')
    # setosa seems to be the most separable

    # kde plot of sepal_length versus sepalwidth for setosa species of flower
    plt.figure()
    setosa = iris[iris['species'] == 'setosa']
    sns.kdeplot(y = setosa['sepal_length'], x = setosa['sepal_width'], cmap = 'plasma', shade = True)

    # splitting the data into a training set and a testing set
    X = iris.drop('species', axis = 1)
    y = iris['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

    # SVC() model and fitting the model to the training data
    model = SVC()
    model.fit(X_train, y_train)

    # predictions from the model, a confusion matrix and a classification report
    predictions = model.predict(X_test)
    print(confusion_matrix(y_test, predictions), '\n')
    print(classification_report(y_test, predictions))

    # dictionary filled with some parameters for C and gamma
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

    # GridSearchCV object, fitted to the training data.
    grid = GridSearchCV(SVC(), param_grid, verbose = 3)
    grid.fit(X_train, y_train)

    # predictions using the test set, classification report and confusion matrix
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions), '\n')
    print(confusion_matrix(y_test, grid_predictions))
    # the results are similar

    plt.show()


if __name__ == '__main__':
    main()