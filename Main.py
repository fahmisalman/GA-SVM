import numpy as np
import matplotlib.pyplot as plt
import random
import math


def plot(x, y):
    x = np.array(x)
    y = np.array(y)

    plt.scatter(x[:, 0], x[:, 1], c=y[:], s=30, alpha=0.5)
    plt.show()


def gen_populasi(n_pop, n_krom):
    pop = [[random.random() * 4 - 2 for i in range(n_krom)] for j in range(n_pop)]
    return pop


def randomParent(n_krom):
    return int(round(random.uniform(0, n_krom)))


def hitung_fitness(x, y, p):
    for i in range(len(x)):
        f = []
        for i in range(len(p)):
            temp = []
            for j in range(len(x)):
                temp.append(((p[i][0] * x[j][0] + p[i][1] * x[j][1] + p[i][2]) * y[j]) / math.sqrt(p[i][0] ** 2 + p[i][1] ** 2))
            f.append(temp[temp.index(min(temp))])
        return f


if __name__ == '__main__':

    x_train = [[1, 0], [0, 1], [2, 2], [-1, 0], [0, -1], [2, 1], [-2, 1], [0, -2], [1, 2], [1, 1], [-1, -1], [-2, -2]]
    y_train = [1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1]

    # plot(x_train, y_train)

    n_pop = 5
    n_krom = 3
    n_gen = 5
    pCross = 0.8
    pMutasi = 0.1

    pop = gen_populasi(n_pop, n_krom)

    for i in range(n_gen):

        fitness = []

        anak = []

        for j in range(n_pop // 2):

            # Seleksi orang tua
            parent1 = randomParent(n_pop - 1)
            parent2 = randomParent(n_pop - 1)

            anak1 = pop[parent1][:]
            anak2 = pop[parent2][:]

            # Crossover
            rand = random.random()
            titik = int(round(random.uniform(1, n_krom - 1)))
            if rand <= pCross:
                for k in range(titik):
                    anak1[k], anak2[k] = anak2[k], anak1[k]

            # mutasi
            rand = random.random()
            titik = int(round(random.uniform(0, n_krom - 1)))
            if rand <= pMutasi:
                anak1[titik] += random.random() * 4 - 2
            rand = random.random()
            titik = int(round(random.uniform(0, n_krom - 1)))
            if rand <= pMutasi:
                anak2[titik] += random.random() * 4 - 2

            anak.append(anak1)
            anak.append(anak2)

        gab = pop + anak
        fitness = hitung_fitness(x_train, y_train, gab)
        steadyState = sorted(range(len(fitness)), key=lambda k: fitness[k], reverse=True)
        pop = []
        for j in range(n_pop):
            pop.append(gab[steadyState[j]])

    print("\nJumlah nilai :", fitness[steadyState[0]])
    print("Kromosom Terbaik :", pop[0])

    temp = []

    for i in range(len(x_train)):
        if pop[0][0] * x_train[i][0] + pop[0][1] * x_train[i][1] + pop[0][2] > 0:
            temp.append(1)
        else:
            temp.append(-1)
    plot(x_train, temp)
