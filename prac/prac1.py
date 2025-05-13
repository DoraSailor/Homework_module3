import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC

df = px.data.iris()
df= df.drop(columns=["petal_length"])
print(df)
# fig = px.scatter_3d(df, x="sepal_length", y="sepal_width", z="petal_width", color="species")
# fig.show()
# Это конечно здорово, но plotly.express мы не проходили,
# а изучать её как то не хочется учитывая что все графики можно сделать в matplotlib
### ЗАДАЧА НА ПРАКТИКУ № 1
# Обучение с учителем (классификация). Выбрать ДВА ЛЮБЫХ СОРТА и для них реализовать.
# 1. Метод опорных векторов
print("1. Метод опорных векторов")
#setosa and versicolor

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_xlabel("sepal_length")
ax.set_ylabel("sepal_width")
ax.set_zlabel("petal_width")

ax.scatter(df[df["species"] == "setosa"]["sepal_length"],
           df[df["species"] == "setosa"]["sepal_width"],
           df[df["species"] == "setosa"]["petal_width" ],
           label = 'setosa')
ax.scatter(df[df["species"] == "versicolor"]["sepal_length"],
           df[df["species"] == "versicolor"]["sepal_width"],
           df[df["species"] == "versicolor"]["petal_width" ],
           label = 'versicolor')
df = df[(df["species"] == "setosa") | (df["species"] == "versicolor")]

X = df[["sepal_length","sepal_width","petal_width"] ]
Y = df["species"]
model = SVC(kernel='linear', C=10000)
model.fit(X, Y)
print(model.support_vectors_)
ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],model.support_vectors_[:,2],s=400,facecolor='none',edgecolors='black')

x1_p = np.linspace(min(df["sepal_length"]),max(df["sepal_length"]), 8)
x2_p = np.linspace(min(df["sepal_width"]), max(df["sepal_width"]), 8)
x3_p = np.linspace(min(df["petal_width"]), max(df["petal_width"]), 8)

X1_p, X2_p ,X3_p= np.meshgrid(x1_p, x2_p,x3_p)
X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel(),X3_p.ravel()]).T,columns=["sepal_length","sepal_width","petal_width"])
y_p = model.predict(X_p)

X_p["species"] = y_p
X_p_setosa = X_p[X_p["species"] == "setosa"]
X_p_versicolor = X_p[X_p["species"] == "versicolor"]

print(X_p_setosa)
print(X_p_versicolor)

ax.scatter(X_p_setosa["sepal_length"], X_p_setosa["sepal_width"],X_p_setosa["petal_width"], alpha=0.4,label = 'predicted setosa')
ax.scatter(X_p_versicolor["sepal_length"], X_p_versicolor["sepal_width"], X_p_versicolor["petal_width"],alpha=0.4,label = 'predicted versicolor')

plt.legend()
plt.show()
