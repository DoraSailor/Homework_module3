# 2. Метод главных компонент
print('2. Метод главных компонент')

import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

df = px.data.iris()
df= df.drop(columns=["petal_length"])
print(df)
# fig = px.scatter_3d(df, x="sepal_length", y="sepal_width", z="petal_width", color="species")
# fig.show()
# Это конечно здорово, но plotly.express мы не проходили,
# а изучать её как то не хочется учитывая что все графики можно сделать в matplotlib

# setosa
data_v = df[df["species"] == "setosa"]
data_v = data_v.drop(columns=["species"])
data_v = data_v.drop(columns=["species_id"])
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_xlabel("sepal_length")
ax.set_ylabel("sepal_width")
ax.set_zlabel("petal_width")
# print(data_v)
X = data_v["sepal_length"]
Y = data_v["sepal_width"]
Z = data_v["petal_width"]
ax.scatter(X, Y,Z,label = 'setosa')
p = PCA(n_components=3)
p.fit(data_v)


print(p.components_)
print(p.explained_variance_)
print(p.mean_)
ax.scatter(p.mean_[0], p.mean_[1], p.mean_[2],c='red',label='p_mean')

ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[0][0] * np.sqrt(p.explained_variance_[0])],
    [p.mean_[1], p.mean_[1] + p.components_[0][1] * np.sqrt(p.explained_variance_[0])],
    [p.mean_[2], p.mean_[2] + p.components_[0][2] * np.sqrt(p.explained_variance_[0])],)
ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[1][0] * np.sqrt(p.explained_variance_[1])],
    [p.mean_[1], p.mean_[1] + p.components_[1][1] *np.sqrt(p.explained_variance_[1])],
    [p.mean_[2], p.mean_[2] + p.components_[1][2] * np.sqrt(p.explained_variance_[1])],)
ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[2][0] * np.sqrt(p.explained_variance_[2])],
    [p.mean_[1], p.mean_[1] + p.components_[2][1] * np.sqrt(p.explained_variance_[2])],
    [p.mean_[2], p.mean_[2] + p.components_[2][2] * np.sqrt(p.explained_variance_[2])],)

X_p = p.transform(data_v)
p1 = PCA(n_components=1)
p1.fit(data_v)
X_p = p1.transform(data_v)
print(X_p)
X_p_new = p1.inverse_transform(X_p)
print(X_p_new)
ax.scatter(X_p_new[:, 0], X_p_new[:, 1],X_p_new[:, 2],s=20,label='trasformed')
ax.legend()
plt.show()

# versicolor
data_v = df[df["species"] == "versicolor"]
data_v = data_v.drop(columns=["species"])
data_v = data_v.drop(columns=["species_id"])
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_xlabel("sepal_length")
ax.set_ylabel("sepal_width")
ax.set_zlabel("petal_width")
# print(data_v)
X = data_v["sepal_length"]
Y = data_v["sepal_width"]
Z = data_v["petal_width"]
ax.scatter(X, Y,Z,label = 'versicolor')
p = PCA(n_components=3)
p.fit(data_v)
# X_p = p.transform(data_v)

print(p.components_)
print(p.explained_variance_)
print(p.mean_)
ax.scatter(p.mean_[0], p.mean_[1], p.mean_[2],c='red',label='p_mean')

ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[0][0] * np.sqrt(p.explained_variance_[0])],
    [p.mean_[1], p.mean_[1] + p.components_[0][1] * np.sqrt(p.explained_variance_[0])],
    [p.mean_[2], p.mean_[2] + p.components_[0][2] * np.sqrt(p.explained_variance_[0])],)
ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[1][0] * np.sqrt(p.explained_variance_[1])],
    [p.mean_[1], p.mean_[1] + p.components_[1][1] *np.sqrt(p.explained_variance_[1])],
    [p.mean_[2], p.mean_[2] + p.components_[1][2] * np.sqrt(p.explained_variance_[1])],)
ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[2][0] * np.sqrt(p.explained_variance_[2])],
    [p.mean_[1], p.mean_[1] + p.components_[2][1] * np.sqrt(p.explained_variance_[2])],
    [p.mean_[2], p.mean_[2] + p.components_[2][2] * np.sqrt(p.explained_variance_[2])],)

X_p = p.transform(data_v)
p1 = PCA(n_components=1)
p1.fit(data_v)
X_p = p1.transform(data_v)
print(X_p)
X_p_new = p1.inverse_transform(X_p)
print(X_p_new)
ax.scatter(X_p_new[:, 0], X_p_new[:, 1],X_p_new[:, 2],s=20,label='trasformed')
ax.legend()
plt.show()