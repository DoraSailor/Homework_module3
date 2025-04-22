# ДЗ. Убрать из данных iris часть точек (на которых мы обучаемся) и убедиться, что на предсказание влияют только опорные вектора.
from operator import index

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

# 1. Оригинальный набор точек
iris = sns.load_dataset("iris")

data = iris[["sepal_length", "petal_length", "species"]]
data_df = data[(data["species"] == "setosa") |(data["species"] == "versicolor")]
X = data_df[["sepal_length", "petal_length"]]
y = data_df["species"]

data_df_seposa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

plt.scatter(data_df_seposa["sepal_length"],data_df_seposa["petal_length"])
plt.scatter(data_df_versicolor["sepal_length"],data_df_versicolor["petal_length"])

model = SVC(kernel='linear', C=10000)
model.fit(X, y)
print(model.support_vectors_)


plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=400,facecolor='none',edgecolors='black')

x1_p = np.linspace(min(data_df["sepal_length"]),max(data_df["sepal_length"]), 200)
x2_p = np.linspace(min(data_df["petal_length"]),max(data_df["petal_length"]), 200)
X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T,columns=["sepal_length", "petal_length"])

y_p = model.predict(X_p)

X_p["species"] = y_p

X_p_setosa = X_p[X_p["species"] == "setosa"]
X_p_versicolor = X_p[X_p["species"] == "versicolor"]

plt.scatter(X_p_setosa["sepal_length"], X_p_setosa["petal_length"], alpha=0.4)
plt.scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha=0.4)

data_df_original=data_df
original_support_vectors=model.support_vectors_

# 2) уберем пару точек только неопорных веторов
plt.figure()
data_df=data_df_original[[(s, p) in original_support_vectors or i%2==0 for i, s, p in zip(data_df_original.index, data_df_original.sepal_length,data_df_original.petal_length)]]

X = data_df[["sepal_length", "petal_length"]]
y = data_df["species"]

data_df_seposa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

plt.scatter(data_df_seposa["sepal_length"],data_df_seposa["petal_length"])
plt.scatter(data_df_versicolor["sepal_length"],data_df_versicolor["petal_length"])

model = SVC(kernel='linear', C=10000)
model.fit(X, y)

print(model.support_vectors_)

plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=400,facecolor='none',edgecolors='black')

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T,columns=["sepal_length", "petal_length"])

y_p = model.predict(X_p)

X_p["species"] = y_p

X_p_setosa = X_p[X_p["species"] == "setosa"]
X_p_versicolor = X_p[X_p["species"] == "versicolor"]

plt.scatter(X_p_setosa["sepal_length"], X_p_setosa["petal_length"], alpha=0.4)
plt.scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha=0.4)

# 3) теперь уберем точки включая все опорные вектора
plt.figure()
data_df=data_df_original[[(s, p) not in original_support_vectors and i%2==0 for i, s, p in zip(data_df_original.index, data_df_original.sepal_length,data_df_original.petal_length)]]

X = data_df[["sepal_length", "petal_length"]]
y = data_df["species"]

data_df_seposa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

plt.scatter(data_df_seposa["sepal_length"],data_df_seposa["petal_length"])
plt.scatter(data_df_versicolor["sepal_length"],data_df_versicolor["petal_length"])

model = SVC(kernel='linear', C=10000)
model.fit(X, y)

print(model.support_vectors_)

plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=400,facecolor='none',edgecolors='black')

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T,columns=["sepal_length", "petal_length"])

y_p = model.predict(X_p)

X_p["species"] = y_p

X_p_setosa = X_p[X_p["species"] == "setosa"]
X_p_versicolor = X_p[X_p["species"] == "versicolor"]

plt.scatter(X_p_setosa["sepal_length"], X_p_setosa["petal_length"], alpha=0.4)
plt.scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha=0.4)

# 4) теперь уберем точки включая какой-нибудь один опорный вектор
plt.figure()
data_df=data_df_original[[any(np.array((s, p)) != original_support_vectors[1]) for i, s, p in zip(data_df_original.index, data_df_original.sepal_length,data_df_original.petal_length)]]
X = data_df[["sepal_length", "petal_length"]]
y = data_df["species"]

data_df_seposa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

plt.scatter(data_df_seposa["sepal_length"],data_df_seposa["petal_length"])
plt.scatter(data_df_versicolor["sepal_length"],data_df_versicolor["petal_length"])

model = SVC(kernel='linear', C=10000)
model.fit(X, y)

print(model.support_vectors_)

plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=400,facecolor='none',edgecolors='black')

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T,columns=["sepal_length", "petal_length"])

y_p = model.predict(X_p)

X_p["species"] = y_p

X_p_setosa = X_p[X_p["species"] == "setosa"]
X_p_versicolor = X_p[X_p["species"] == "versicolor"]

plt.scatter(X_p_setosa["sepal_length"], X_p_setosa["petal_length"], alpha=0.4)
plt.scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha=0.4)
plt.show()
