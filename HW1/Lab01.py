import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# read in the data using pandas
df = pd.read_csv('heart.csv')

# create a dataframe with all training data except the target column
X, y = df.drop(columns=['target']), df['target'].values

# split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# check accuracy using holdout
knn= KNeighborsClassifier(n_neighbaors=2)
knn.fit(X_train, y_train)
print("holdout : {}".format(knn.score(X_test, y_test)))

# check accuracy using k-fold cross valudation
cv_score = cross_val_score(knn, X, y, cv=10)
print("cross validation : {}".format(np.mean(cv_score)))
