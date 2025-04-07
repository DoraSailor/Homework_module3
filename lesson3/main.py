import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt

data = np.array(
    [
        [1, 5],
        [2, 7],
        [3, 7],
        [4, 10],
        [5, 11],
        [6, 14],
        [7, 17],
        [8, 19],
        [9, 22],
        [10, 28],
    ]
)

# Градиентный спуск — пакетный градиентный спуск:
# для работы используются ВСЕ доступные обучающие данные.
# На практике используется стохастический градиентный спуск,
# т.е. на каждой итерации обучение происходит только на одной выборке
# с данными.
# В этом случае сокращается число вычислений и вносится смещение,
# что позволяет бороться с переобучением.
# Мини-пакетный градиентный спуск: на каждой итерации используется
# несколько выборок.

x = data[:, 0]
y = data[:, 1]

n = len(x)
w1 = 0.0
w0 = 0.0

L = 0.001
# Размер выборки:
sample_size = 2

iterations = 100_000

for i in range(iterations):
    idx = np.random.choice(n, sample_size, replace=False)
    D_w0 = 2 * sum(-y[idx] + w0 + w1 * x[idx])
    D_w1 = 2 * sum ((x[idx] * (-y[idx] + w0 + w1 * x[idx])))
    w1 -= L * D_w1
    w0 -= L * D_w0

print (w1, w0)
# Как оценить насколько сильно "промахиваются" прогнозы при использовании
# линейной регрессии?

data_df = pd.DataFrame(data)
print (data_df.corr(method = "pearson"))

data_df[1] = data_df[1].values[::-1]
print (data_df.corr(method = "pearson"))

# Коэффициент корреляции помогает понять наличие связи меж двумя переменными.

# Обучающие и тестовые выборки
# Основной метод борьбы с переобучением заключается в том,
# что набор данных делится на обучающую и тестовую выборки.

# Во всех видах машинного обучения с учителем это встречается.

# Обычная пропорция — это 2/3 на обучение и 1/3 на тест.
# Иногда встречается 4/5 к 1/5 и 9/10 к 1/10. Всё зависит от данных, модели.

X = data_df.values[:,:-1]
Y = data_df.values[:,-1]

print (X)
print (Y)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3)
# print(X_train)
# print(Y_train)

# print(X_test)
# print(Y_test)

kfold = KFold(n_splits=3, random_state=1, shuffle=True)

model = LinearRegression()
# model.fit(X_train, Y_train)
# Среднеквадратичные ошибки:
results = cross_val_score(model, X, Y, cv=kfold)

print(results)
print(results.mean(), results.std())
# r = model.score(X_test, Y_test) # Корень из коэффициента детерминации
# print(r)

# Метрики показывают насколько ЕДИНООБРАЗНО ведёт себя модель
# на разных выборках.
# Возможно использование поэлементной перекрёстной валидации (мало данных).

# Случайная валидация — позволяет перемешивать и разделять данные многократно.
# Валидационная выборка — имеет смысл при сравнении
# нескольких моделей или конфигураций.

data_df = pd.read_csv('multiple_independent_variable_linear.csv')
print(data_df.head())

X = data_df.values[:,:-1]
Y = data_df.values[:,-1]

model = LinearRegression().fit(X, Y)
print(model.coef_, model.intercept_)

x1 = X[:,0]
x2 = X[:,1]
y = Y

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(x1, x2, y)

x1_ = np.linspace(min(x1), max(x1), 100)
x2_ = np.linspace(min(x2), max(x2), 100)

X1_, X2_ = np.meshgrid(x1_, x2_)
Y_ = model.intercept_ + model.coef_[0]*X1_ + model.coef_[1]*X2_

ax.plot_surface(X1_, X2_, Y_, cmap="Greys", alpha=0.1)

plt.show()