import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

def compute_cost(X, y, m, b, lasso_lambda=0, ridge_lambda=0):
    n = len(y)
    cost = 0.0
    for i in range(n):
        y_pred = m[0] * X[i][0] + m[1] * X[i][1] + b
        cost += (y[i] - y_pred) ** 2
    cost = cost / n

    # Додавання регуляризації Lasso та Ridge до функції вартості
    cost += lasso_lambda * (np.abs(m[0]) + np.abs(m[1]))
    cost += ridge_lambda * (m[0]**2 + m[1]**2)

    return cost

def gradient_descent(m_now, b_now, X, y, L, lasso_lambda=0, ridge_lambda=0, epochs=1000):
    n = len(y)
    cost_history = []

    itr = 0
    for epoch in range(epochs):
        itr += 1
        m_gradient = np.zeros(2)
        b_gradient = 0

        if epoch > 2 and abs(cost_history[epoch-1] - cost_history[epoch-2]) <= 1e-4: break

        for i in range(n):
            error = y[i] - (m_now[0] * X[i][0] + m_now[1] * X[i][1] + b_now)
            m_gradient[0] += -(2/n) * X[i][0] * error
            m_gradient[1] += -(2/n) * X[i][1] * error
            b_gradient += -(2/n) * error

        if lasso_lambda != 0:
            m_gradient[0] += lasso_lambda * np.sign(m_now[0])
            m_gradient[1] += lasso_lambda * np.sign(m_now[1])

        if ridge_lambda != 0:
            m_gradient[0] += 2 * ridge_lambda * m_now[0]
            m_gradient[1] += 2 * ridge_lambda * m_now[1]

        m_now[0] -= L * m_gradient[0]
        m_now[1] -= L * m_gradient[1]
        b_now -= L * b_gradient

        cost = compute_cost(X, y, m_now, b_now, lasso_lambda, ridge_lambda)
        cost_history.append(cost)

    print(f"Кількість ітерацій: {itr}")
    return m_now, b_now, cost_history

def normalize_data(X):
    # Обчислюємо середнє значення кожної ознаки
    mean = np.mean(X, axis=0)
    # Обчислюємо стандартне відхилення кожної ознаки
    std = np.std(X, axis=0)
    # Віднімаємо середнє значення від кожного значення ознаки
    X_normalized = X - mean
    # Поділяємо значення на стандартне відхилення
    X_normalized /= std
    return X_normalized, mean, std

def mean_squared_error_m(y_true, y_pred):
    n = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / n
    return mse

def r2_score_m(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def plotChart(iterations, cost_num):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost_num, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Iterations')
    plt.style.use('fivethirtyeight')
    plt.show()

def run():
    # Зчитуємо датасет
    data = pd.read_csv("good_reads_top_1000_books.csv")

    # Вибираємо лише потрібні колонки
    X = data[["Number of Ratings", "Average Rating"]].values
    y = data["Score on Goodreads"].values

    X_normalized, mean, std = normalize_data(X)

    m_initial = np.zeros(2)
    b_initial = 0
    learning_rate = 0.001
    lasso_lambda = 0
    ridge_lambda = 0.8
    epochs = 100000

    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    start_time = time.time()
    m, b, cost_history = gradient_descent(m_initial, b_initial, X_train, y_train, learning_rate, lasso_lambda, ridge_lambda, epochs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Витрачено часу на навчання: {elapsed_time} сек.")

    print("Коефіцієнти (a):", m)
    print("Зміщення (b):", b)
    #print("Історія вартості:", cost_history[-1])

    # Прогнозування та оцінка
    y_pred = X_test @ m + b

    mse_m = mean_squared_error_m(y_test, y_pred)
    r2_m = r2_score_m(y_test, y_pred)

    print(f'MSE: {mse_m}')
    print(f'R-squared: {r2_m}')

    # Візуалізація
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Actual vs Predicted')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Лінія y=x
    plt.show()

    plotChart(len(cost_history), cost_history)



if __name__ == '__main__':
    run()


