#Задание 1
#Загрузите модуль pyplot библиотеки matplotlib с псевдонимом plt, а также библиотеку numpy с псевдонимом np.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Примените магическую функцию %matplotlib inline для отображения графиков в Jupyter Notebook и настройки конфигурации ноутбука со значением 'svg' для более четкого отображения графиков.
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
#Создайте список под названием x с числами 1, 2, 3, 4, 5, 6, 7 и список y с числами 3.5, 3.8, 4.2, 4.5, 5, 5.5, 7.
x=[1, 2, 3, 4, 5, 6, 7]
y=[3.5, 3.8, 4.2, 4.5, 5, 5.5, 7.]
print(x)
print(y)
#С помощью функции plot постройте график, соединяющий линиями точки с горизонтальными координатами из списка x и вертикальными - из списка y.
plt.plot(x,y)
plt.show()
#Затем в следующей ячейке постройте диаграмму рассеяния (другие названия - диаграмма разброса, scatter plot).
plt.scatter(x, y)
plt.show()


#Задание 2
#С помощью функции linspace из библиотеки Numpy создайте массив t из 51 числа от 0 до 10 включительно.
t = np.linspace(0, 10, 51)
t
#Создайте массив Numpy под названием f, содержащий косинусы элементов массива t.
f=np.cos(t)
f
#Постройте линейную диаграмму, используя массив t для координат по горизонтали,а массив f - для координат по вертикали. Линия графика должна быть зеленого цвета.
plt.plot(t,f,color="green")
plt.show()
#Выведите название диаграммы - 'График f(t)'. Также добавьте названия для горизонтальной оси - 'Значения t' и для вертикальной - 'Значения f'.
#Ограничьте график по оси x значениями 0.5 и 9.5, а по оси y - значениями -2.5 и 2.5.
plt.axis([0.5, 9.5, -2.5, 2.5])
plt.plot(t,f,color="green")
plt.title('График f(t)')
plt.xlabel('Значения t')
plt.ylabel('Значения f')
plt.show()


#*Задание 3
#С помощью функции linspace библиотеки Numpy создайте массив x из 51 числа от -3 до 3 включительно.
x = np.linspace(-3, 3, 51)
x
#Создайте массивы y1, y2, y3, y4 по следующим формулам:
#y1 = x**2
#y2 = 2 * x + 0.5
#y3 = -3 * x - 1.5
#y4 = sin(x)
y1 = x**2
y2 = 2 * x + 0.5
y3 = -3 * x - 1.5
y4 = np.sin(x)
print(y1)
print(y2)
print(y3)
print(y4)
#Используя функцию subplots модуля matplotlib.pyplot, создайте объект matplotlib.figure.Figure с названием fig и массив объектов Axes под названием ax,причем так, чтобы у вас было 4 отдельных графика в сетке, состоящей из двух строк и двух столбцов. В каждом графике массив x используется для координат по горизонтали.В левом верхнем графике для координат по вертикали используйте y1,в правом верхнем - y2, в левом нижнем - y3, в правом нижнем - y4.Дайте название графикам: 'График y1', 'График y2' и т.д.
#Для графика в левом верхнем углу установите границы по оси x от -5 до 5.
#Установите размеры фигуры 8 дюймов по горизонтали и 6 дюймов по вертикали.
#Вертикальные и горизонтальные зазоры между графиками должны составлять 0.3.
fig, ax = plt.subplots(nrows=2, ncols=2)
ax1, ax2, ax3, ax4 = ax.flatten()
ax1.plot(x, y1)
ax1.set_title('График y1')
ax1.set_xlim([-5, 5])
ax2.plot(x, y2)
ax2.set_title('График y2')
ax3.plot(x, y3)
ax3.set_title('График y3')
ax4.plot(x, y4)
ax4.set_title('График y4')
fig.set_size_inches(8, 6)
plt.subplots_adjust(wspace=0.3, hspace=0.3)


#*Задание 4
#В этом задании мы будем работать с датасетом, в котором приведены данные по мошенничеству с кредитными данными: Credit Card Fraud Detection (информация об авторах: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015).
#Ознакомьтесь с описанием и скачайте датасет creditcard.csv с сайта Kaggle.com по ссылке:
#Credit Card Fraud Detection
#Данный датасет является примером несбалансированных данных, так как мошеннические операции с картами встречаются реже обычных.
#Импортируйте библиотеку Pandas, а также используйте для графиков стиль “fivethirtyeight”.
#Посчитайте с помощью метода value_counts количество наблюдений для каждого значения целевой переменной Class и примените к полученным данным метод plot, чтобы построить столбчатую диаграмму. Затем постройте такую же диаграмму, используя логарифмический масштаб.
#На следующем графике постройте две гистограммы по значениям признака V1 - одну для мошеннических транзакций (Class равен 1) и другую - для обычных (Class равен 0). Подберите значение аргумента density так, чтобы по вертикали графика было расположено не число наблюдений, а плотность распределения. Число бинов должно равняться 20 для обеих гистограмм, а коэффициент alpha сделайте равным 0.5, чтобы гистограммы были полупрозрачными и не загораживали друг друга. Создайте легенду с двумя значениями: “Class 0” и “Class 1”. Гистограмма обычных транзакций должна быть серого цвета, а мошеннических - красного. Горизонтальной оси дайте название “V1”.
df = pd.read_csv('creditcard.csv')
df.head()
plt.style.use('fivethirtyeight')
t=df['Class'].value_counts()
t
df_class_info = pd.Series(t)
df_class_info.plot('bar')
plt.show()
df_class_info.plot(kind='bar', logy=True)
plt.show()
v1_class1=df.set_index('Class')['V1'].filter(like='1', axis=0)
v1_class1=v1_class1.reset_index()
v1_class1=v1_class1.drop('Class', axis=1)
v1_class1.head(), v1_class1.count()
v1_class0=df.set_index('Class')['V1'].filter(like='0', axis=0)
v1_class0=v1_class0.reset_index()
v1_class0=v1_class0.drop('Class', axis=1)
v1_class0.head(), v1_class0.count()
plt.hist(v1_class0['V1'], bins=20, color='grey', edgecolor='black', density = True, orienta
plt.hist(v1_class1['V1'], bins=20, color='red', edgecolor='black', density = True, orientat
plt.plot()
plt.xlabel('Class')
plt.legend(labels=['Class 0', 'Class 1'])


#**Задание на повторение материала
#Создать одномерный массив Numpy под названием a из 12 последовательных целых чисел чисел от 12 до 24 невключительно
a=np.arange(12, 24)
print(a)

#Создать 5 двумерных массивов разной формы из массива a. Не использовать в аргументах метода reshape число -1.
a1=a.reshape(2, 6)
a2=a.reshape(3, 4)
a3=a.reshape(6, 2)
a4=a.reshape(4, 3)
a5=a.reshape(12, 1)
print(a1)
print(a2)
print(a3)
print(a4)
print(a5)

#Создать 5 двумерных массивов разной формы из массива a. Использовать в аргументах метода reshape число -1 (в трех примерах - для обозначения числа столбцов, в двух - для строк).
a1=a.reshape(2, -1)
a2=a.reshape(3, -1)
a3=a.reshape(-1, 2)
a4=a.reshape(-1, 3)
a5=a.reshape(12, -1)
print(a1)
print(a2)
print(a3)
print(a4)
print(a5)

#Можно ли массив Numpy, состоящий из одного столбца и 12 строк, назвать одномерным?
x=np.arange(12, 24)
matr=x.reshape(3,4)
y=matr.reshape(12,1)
print(x)
print(y) #Таким образом, массив Numpy, состоящий из одного столбца и 12 строк, НЕЛЬЗЯ назвать одномерным.
#Создать массив из 3 строк и 4 столбцов, состоящий из случайных чисел с плавающей запятой из нормального распределения со средним, равным 0 и среднеквадратичным отклонением, равным 1.0. Получить из этого массива одномерный массив с таким же атрибутом size, как и исходный массив.
b=np.random.randn(3,4)
print(b.size)
b=b.flatten()
print(b.size)
print(b)

#Создать массив a, состоящий из целых чисел, убывающих от 20 до 0 невключительно с интервалом 2.
a=np.arange(20, 0, -2)
print(a)

#Создать массив b, состоящий из 1 строки и 10 столбцов: целых чисел, убывающих от 20 до 1 невключительно с интервалом 2. В чем разница между массивами a и b?
b=np.arange(20, 1, -2).reshape(1,10)
print(b)

#Вертикально соединить массивы a и b. a - двумерный массив из нулей, число строк которого больше 1 и на 1 меньше, чем число строк двумерного массива b, состоящего из единиц. Итоговый массив v должен иметь атрибут size, равный 10.
a = np.zeros((2, 2))
b = np.zeros((3, 2))+1
v = np.vstack((a, b))
a.shape, b.shape, v.shape
print(v.size)

#Создать одномерный массив а, состоящий из последовательности целых чисел от 0 до 12. Поменять форму этого массива, чтобы получилась матрица A (двумерный массив Numpy), состоящая из 4 строк и 3 столбцов. Получить матрицу At путем транспонирования матрицы A. Получить матрицу B, умножив матрицу A на матрицу At с помощью матричного умножения. Какой размер имеет матрица B? Получится ли вычислить обратную матрицу для матрицы B и почему?
a=np.arange(0, 12)
A=a.reshape(4,3)

#Инициализируйте генератор случайных числе с помощью объекта seed, равного 42.
x = np.random.seed(42)
print(x)

#Создайте одномерный массив c, составленный из последовательности 16-ти случайных равномерно распределенных целых чисел от 0 до 16 невключительно.
c=np.random.randint(0, 16,16)
print(c)

#Поменяйте его форму так, чтобы получилась квадратная матрица C. Получите матрицу D, поэлементно прибавив матрицу B из предыдущего вопроса к матрице C, умноженной на 10. Вычислите определитель, ранг и обратную матрицу D_inv для D.
C=c.reshape(4,4)
D=B+C*10
print(D)
d=np.linalg.det(D)
print(d)
r=np.linalg.matrix_rank(D)
print(r)
D_inv=np.linalg.inv(D)
print(D_inv)

#Приравняйте к нулю отрицательные числа в матрице D_inv, а положительные - к единице. Убедитесь, что в матрице D_inv остались только нули и единицы. С помощью функции numpy.where, используя матрицу D_inv в качестве маски, а матрицы B и C - в качестве источников данных, получите матрицу E размером 4x4.  Элементы матрицы E, для которых соответствующий элемент матрицы D_inv равен 1, должны быть равны соответствующему элементу матрицы B, а элементы матрицы E, для которых соответствующий элемент матрицы D_inv равен 0, должны быть равны соответствующему элементу матрицы C.
D_inv=np.where(D_inv<0, 0,1)
print(D_inv)
print(B)
print(C)
E=np.where(D_inv==1, B, C)
print(E)