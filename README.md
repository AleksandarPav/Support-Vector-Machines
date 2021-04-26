# Support Vector Machines

Famous Iris flower data set is used. The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). The iris dataset contains measurements for 150 iris flowers from three different species.
The three classes in the Iris dataset:
Iris-setosa (n=50)
Iris-versicolor (n=50)
Iris-virginica (n=50)
The four features of the Iris dataset:
sepal length in cm
sepal width in cm
petal length in cm
petal width in cm

Based on initial visualizing the data, setosa seems to be the most separable. Data is splitted into a training and a testing set, with 7:3 ratio. Training set is fitted to a SVM model and predictions are made based on the testing set. For evaluating the model, classification report and confusion matrix are used. GridSearchCV is used to find values for C and gamma parameter that would potentially give better results. Again, classification report and confusion matrix are used for model evaluation, but it showed similar results as the original SVM model.
