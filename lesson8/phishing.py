import pandas as pd
import numpy as np

data = pd.read_csv("phishing.csv")
print(data.head())

print(data.columns)
data.info()

X = data.drop(columns="class")
print (X.head())

Y = pd.DataFrame(data["class"])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
model = dt.fit(X_train, y_train)

dt_predict = model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(dt_predict, y_test))

#

# from sklearn.naive_bayes import MultinomialNB
#
# Y = pd.DataFrame(data["class"].apply(lambda x: 1 if x == 1 else -1))
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
#
# from sklearn.tree import DecisionTreeClassifier
#
# dt = MultinomialNB()
# model = dt.fit(X_train, y_train)
#
# dt_predict = model.predict(X_test)
#
# from sklearn.metrics import accuracy_score
#
# print(accuracy_score(dt_predict, y_test))