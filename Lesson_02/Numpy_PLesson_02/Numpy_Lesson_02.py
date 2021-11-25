# Практическое задание 2 урока. “Вычисления с помощью Numpy”. Задание 1
import numpy as np # Подключаем библиотеку numpy и задаем ей псевдоним np
# создаем двумерный массив
a = np.array([[1, 6],
              [2, 8],
              [3, 11],
              [3, 10],
              [1, 7]])

print('a =', a) # Выводим на экран массив
mean_a = a.mean(axis=0) # В mean_a записываем среднее значение всех элементов каждого столбца массива my_array
print('mean_a =', mean_a) # Выводим на экран среднее значение всех элементов каждого столбца массива

# Практическое задание 2 урока. “Вычисления с помощью Numpy”. Задание 2
#Вычислите массив a_centered, отняв от значений массива “а” средние значения соответствующих признаков, содержащиеся в массиве mean_a.
# Вычисление должно производиться в одно действие. Получившийся массив должен иметь размер 5x2.
a_centered = np.subtract(a, mean_a)
print('a_centered =', a_centered)

# Практическое задание 2 урока. “Вычисления с помощью Numpy”. Задание 3
#Найдите скалярное произведение столбцов массива a_centered.
# В результате должна получиться величина a_centered_sp.
# Затем поделите a_centered_sp на N-1, где N - число наблюдений.
a1 = a_centered[:,0]
a2 = a_centered[:,1]
a_centered_sp = np.dot(a1,a2)
print('a_centered_sp =', a_centered_sp)

N = a.shape[0]
my_cov = a_centered_sp / (N - 1)
print('my_cov =', my_cov)

# Практическое задание 2 урока. “Вычисления с помощью Numpy”. Задание 4**
print('a_cov =', np.cov(a.T)[0,1])