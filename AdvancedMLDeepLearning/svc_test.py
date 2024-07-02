from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
print(y)
clf.fit(X, y)
print(clf.predict([[2., 2.]]))

import pandas as pd
flights = pd.read_csv('formatted_flights.csv')
print(flights.head(10))

import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np

# Создадим словарь, в котором ключами будут имена людей, а значениями - списки векторов, отождествленных с этими людьми.
people = {
    "Alice": [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.array([19, 0, 8])
    ],
    "Bob": [
        np.array([10, 11, 12]),
        np.array([13, 14, 15]),
        np.array([16, 17, 18])
    ],
    "Carol": [
        np.array([19, 20, 21]),
        np.array([22, 23, 24]),
        np.array([25, 26, 27])
    ]
}

# Создадим словарь, в котором ключами будут имена людей, а значениями - средние векторы, соответствующие этим людям.
average_vectors = {}

# Для каждого человека вычислим средний вектор и сохраним его в словаре.
for person, vectors in people.items():
    average_vectors[person] = np.mean(vectors, axis=0)

# Выведем на печать словарь со средними векторами.
print(average_vectors)

print('########################################################################################')

import pandas as pd

# Создаем пример данных
data = {
    'Person': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C'],
    'Vector': [(1, 2), (3, 4), (5, 6), (2, 3), (4, 5), (0, 1), (2, 3), (4, 5)]
}

df = pd.DataFrame(data)
print(df)

# Группируем данные по человеку и вычисляем средний вектор
average_vectors = df.groupby('Person')['Vector'].apply(lambda x: tuple(map(lambda i: sum(i) / len(i), zip(*x)))).reset_index()

print(average_vectors)



print('########################################################################################')
import pandas as pd

# Создаем пример данных в виде значений
data = {
    'Person': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C'],
    'X': [1, 3, 5, 2, 4, 0, 2, 4],
    'Y': [2, 4, 6, 3, 5, 1, 3, 5]
}

df = pd.DataFrame(data)
print(df)

# Группируем данные по человеку и вычисляем средний вектор
average_vectors = df.groupby('Person').mean().reset_index()

print(average_vectors)

print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Пример данных с усредненными векторами для каждого человека
data = {
    'Person': ['A', 'B', 'C'],
    'X': [3, 3, 2],
    'Y': [4, 4, 3]
}

average_vectors = pd.DataFrame(data)
print(average_vectors)

# Вычисляем косинусное сходство между парами усредненных векторов
# Предполагаем, что первые два строки в DataFrame представляют два усредненных вектора
cosine_sim = cosine_similarity(average_vectors.iloc[0:2, 1:].values.reshape(1, -1), average_vectors.iloc[1:3, 1:].values.reshape(1, -1))

print(cosine_sim)

print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')


df = pd.read_csv('persons_pics_train.csv')
df.sort_values(by=['label'], inplace=True)
average_vectors = df.groupby('label').mean().reset_index()



data = {
    'Person': ['A', 'B', 'C'],
    'X': [3, 3, 2],
    'Y': [4, 4, 3]
}

data_vectors = pd.DataFrame(data)
print(data_vectors)

print(average_vectors)
cosine_sim = cosine_similarity(average_vectors.iloc[:, 1:].values)
print(cosine_sim[6])

sns.heatmap(cosine_sim, annot=True, xticklabels=average_vectors['label'], yticklabels=average_vectors['label'])
plt.title('Косинусное сходство между усредненными векторами')
plt.show()