# Обучение без учителя (классификация).
# 3. Метод k средних
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

df = px.data.iris()
df= df.drop(columns=["petal_length"])
print(df)
# fig = px.scatter_3d(df, x="sepal_length", y="sepal_width", z="petal_width", color="species")
# fig.show()
# Это конечно здорово, но plotly.express мы не проходили,
# а изучать её как то не хочется учитывая что все графики можно сделать в matplotlib



fig = plt.figure()
ax = plt.axes(projection = '3d')
df = df[(df["species"] == "setosa") | (df["species"] == "versicolor")]
ax.scatter(df["sepal_length"],
           df["sepal_width"],
           df["petal_width" ],
           label = 'setosa and versicolor')

X = df[["sepal_length","sepal_width","petal_width"] ]


model = KMeans(n_clusters=2, random_state=0, n_init="auto")
model.fit(X)
print(model.cluster_centers_)

x1_p = np.linspace(min(df["sepal_length"]),max(df["sepal_length"]), 8)
x2_p = np.linspace(min(df["sepal_width"]), max(df["sepal_width"]), 8)
x3_p = np.linspace(min(df["petal_width"]), max(df["petal_width"]), 8)
X1_p, X2_p ,X3_p= np.meshgrid(x1_p, x2_p,x3_p)

X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel(),X3_p.ravel()]).T,columns=["sepal_length","sepal_width","petal_width"])

Y_p=model.predict(X_p)
X_p["species_id"] = Y_p
X_p0 = X_p[X_p["species_id"] == 0]
X_p1 = X_p[X_p["species_id"] == 1]

ax.scatter(X_p0["sepal_length"], X_p0["sepal_width"],X_p0["petal_width"], alpha=0.4,label = 'cluster 0')
ax.scatter(X_p1["sepal_length"], X_p1["sepal_width"], X_p1["petal_width"],alpha=0.4,label = 'cluster 1')


ax.legend()
plt.show()