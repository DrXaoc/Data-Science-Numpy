#Задание 1
#Импортируйте библиотеки pandas и numpy.
#Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn. Создайте датафреймы X и Y из этих данных.
#Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test) с помощью функции train_test_split так, чтобы размер тестовой выборки составлял 30% от всех данных, при этом аргумент random_state должен быть равен 42.
#Создайте модель линейной регрессии под названием lr с помощью класса LinearRegression из модуля sklearn.linear_model.
#Обучите модель на тренировочных данных (используйте все признаки) и сделайте предсказание на тестовых.
#Вычислите R2 полученных предказаний с помощью r2_score из модуля sklearn.metrics.

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()
data = boston["data"]
feature_names = boston["feature_names"]

X = pd.DataFrame(data, columns=feature_names)
X.head()

target = boston["target"]

Y = pd.DataFrame(target, columns=["price"])
Y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)

y_pred_lr = lr.predict(X_test)
check_test_lr = pd.DataFrame({
    "Y_test": Y_test["price"],
    "Y_pred_lr": y_pred_lr.flatten()})

check_test_lr.head()

from sklearn.metrics import mean_squared_error

mean_squared_error_lr = mean_squared_error(check_test_lr["Y_pred_lr"], check_test_lr["Y_test"])
print(mean_squared_error_lr)

#Задание 2
#Создайте модель под названием model с помощью RandomForestRegressor из модуля sklearn.ensemble.
#Сделайте агрумент n_estimators равным 1000, max_depth должен быть равен 12 и random_state сделайте равным 42.
#Обучите модель на тренировочных данных аналогично тому, как вы обучали модель LinearRegression, но при этом в метод fit вместо датафрейма y_train поставьте y_train.values[:, 0], чтобы получить из датафрейма одномерный массив Numpy, так как для класса RandomForestRegressor в данном методе для аргумента y предпочтительно применение массивов вместо датафрейма.
#Сделайте предсказание на тестовых данных и посчитайте R2. Сравните с результатом из предыдущего задания.
#Напишите в комментариях к коду, какая модель в данном случае работает лучше.
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)
clf.fit(X_train, Y_train.values[:, 0])
y_pred_clf = clf.predict(X_test)
check_test_clf = pd.DataFrame({
    "Y_test": Y_test["price"],
    "Y_pred_clf": y_pred_clf.flatten()})

check_test_clf.head()

mean_squared_error_clf = mean_squared_error(check_test_clf["Y_pred_clf"], check_test_clf["Y_test"])
print(mean_squared_error_clf)

print(mean_squared_error_lr, mean_squared_error_clf)

#* Задание 3
#Вызовите документацию для класса , найдите информацию об атрибуте featureimportances.
#С помощью этого атрибута найдите сумму всех показателей важности, установите, какие два признака показывают наибольшую важность.
print(clf.feature_importances_)
feature_importance = pd.DataFrame({'name':X.columns,
                                   'feature_importance':clf.feature_importances_},
                                  columns=['feature_importance', 'name'])
print(feature_importance)
feature_importance.nlargest(2, 'feature_importance')

#* Задание 4
#В этом задании мы будем работать с датасетом, с которым мы уже знакомы по домашнему заданию по библиотеке Matplotlib, это датасет Credit Card Fraud Detection.
#Для этого датасета мы будем решать задачу классификации - будем определять, какие из транзакциции по кредитной карте являются мошенническими.
#Данный датасет сильно несбалансирован (так как случаи мошенничества относительно редки), так что применение метрики accuracy не принесет пользы и не поможет выбрать лучшую модель.
#Мы будем вычислять AUC, то есть площадь под кривой ROC.
#Импортируйте из соответствующих модулей RandomForestClassifier, GridSearchCV и train_test_split.
#Загрузите датасет creditcard.csv и создайте датафрейм df.
#С помощью метода value_counts с аргументом normalize=True убедитесь в том, что выборка несбалансирована.
#Используя метод info, проверьте, все ли столбцы содержат числовые данные и нет ли в них пропусков.
#Примените следующую настройку, чтобы можно было просматривать все столбцы датафрейма:
#pd.options.display.max_columns = 100.
#Просмотрите первые 10 строк датафрейма df.
#Создайте датафрейм X из датафрейма df, исключив столбец Class.
#Создайте объект Series под названием y из столбца Class.
#Разбейте X и y на тренировочный и тестовый наборы данных при помощи функции train_test_split, используя аргументы: test_size=0.3, random_state=100, stratify=y.
#У вас должны получиться объекты X_train, X_test, y_train и y_test.
#Просмотрите информацию о их форме.
#Для поиска по сетке параметров задайте такие параметры:
#parameters = [{'n_estimators': [10, 15],
#'max_features': np.arange(3, 5),
#'max_depth': np.arange(4, 7)}]
#Создайте модель GridSearchCV со следующими аргументами:
#estimator=RandomForestClassifier(random_state=100),
#param_grid=parameters,
#scoring='roc_auc',
#cv=3.
#Обучите модель на тренировочном наборе данных (может занять несколько минут).
#Просмотрите параметры лучшей модели с помощью атрибута bestparams.
#Предскажите вероятности классов с помощью полученнной модели и метода predict_proba.
#Из полученного результата (массив Numpy) выберите столбец с индексом 1 (вероятность класса 1) и запишите в массив y_pred_proba.
#Из модуля sklearn.metrics импортируйте метрику roc_auc_score.
#Вычислите AUC на тестовых данных и сравните с результатом, полученным на тренировочных данных, используя в качестве аргументов массивы y_test и y_pred_proba.
df = pd.read_csv('../Lesson04/creditcard.csv.zip', compression='zip')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

df['Class'].value_counts(normalize=True)
df.info()
pd.options.display.max_columns=100
df.head(10)
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

print('X_train ', X_train.shape)
print('X_test ', X_test.shape)
print('y_train ', y_train.shape)
print('y_test ', y_test.shape)

parameters = [{
    'n_estimators': [10, 15],
    'max_features': np.arange(3, 5),
    'max_depth': np.arange(4, 7)
}]
clf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=100),
    param_grid=parameters,
    scoring='roc_auc',
    cv=3,
)
clf.fit(X_train, y_train)
clf.best_params_
clf = RandomForestClassifier(max_depth=6, max_features=3, n_estimators=15)

clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
y_pred_proba = y_pred[:, 1]
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_proba)

#*Дополнительные задания:
# 1.Загрузите датасет Wine из встроенных датасетов sklearn.datasets с помощью функции load_wine в переменную data.
from sklearn.datasets import load_wine
data = load_wine()

#2.Полученный датасет не является датафреймом. Это структура данных, имеющая ключи аналогично словарю. Просмотрите тип данных этой структуры данных и создайте список data_keys, содержащий ее ключи.
data_keys = data.keys()
print(data_keys)

#3.Просмотрите данные, описание и названия признаков в датасете. Описание нужно вывести в виде привычного, аккуратно оформленного текста, без обозначений переноса строки, но с самими переносами и т.д.
data.data
print(data.DESCR)
data.feature_names

#4.Сколько классов содержит целевая переменная датасета? Выведите названия классов.
print(set(data.target))
print(len(set(data.target)))
data.target_names

#5.На основе данных датасета (они содержатся в двумерном массиве Numpy) и названий признаков создайте датафрейм под названием X.
X = pd.DataFrame(data.data, columns=data.feature_names)
X.head()

#6.Выясните размер датафрейма X и установите, имеются ли в нем пропущенные значения.
X.shape
X.info()

#7.Добавьте в датафрейм поле с классами вин в виде чисел, имеющих тип данных numpy.int64. Название поля - 'target'.
X['target'] = data.target
X.head()

#8.Постройте матрицу корреляций для всех полей X. Дайте полученному датафрейму название X_corr.
X_corr = X.corr()
X_corr

#9.Создайте список high_corr из признаков, корреляция которых с полем target по абсолютному значению превышает 0.5 (причем, само поле target не должно входить в этот список).
high_corr = X_corr.loc[(abs(X_corr['target']) > 0.5) & (X_corr.index != 'target'), X_corr.columns != 'target'].index
high_corr

#10.Удалите из датафрейма X поле с целевой переменной. Для всех признаков, названия которых содержатся в списке high_corr, вычислите квадрат их значений и добавьте в датафрейм X соответствующие поля с суффиксом '_2', добавленного к первоначальному названию признака. Итоговый датафрейм должен содержать все поля, которые, были в нем изначально, а также поля с признаками из списка high_corr, возведенными в квадрат. Выведите описание полей датафрейма X с помощью метода describe.
X = X.drop('target', axis=1)
X.head()
for feature_name in high_corr:
    X[f'{feature_name}_2'] = X.apply(lambda row: row[feature_name] ** 2, axis=1)
X.describe()

