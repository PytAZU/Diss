import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from scipy.integrate import dblquad

# ==== Границы переменных ====
AREA_MIN, AREA_MAX = 0.001, 0.1      # площадь, м^2
ALPHA_MIN, ALPHA_MAX = 0, np.pi/2    # угол падения, радиан
BETA_MIN, BETA_MAX = 0, np.pi/2      # угол beta, радиан
GAMMA_MIN, GAMMA_MAX = 0, np.pi/2    # угол gamma, радиан
THETA_MIN, THETA_MAX = 0, np.pi/2    # угол поворота, радиан
I0_MIN, I0_MAX = 0.5, 2.0
OMEGA_MIN, OMEGA_MAX = 0.5, 2.0
P0_MIN, P0_MAX = 0.5, 2.0
C_MIN, C_MAX = 0.0, 1.0
B_MIN, B_MAX = -1.0, 1.0

# ==== ВАШИ ФУНКЦИИ ====
def my_efficiency(area, alpha, beta, gamma, theta, I0, omega, P0, C, B_vec):
    """Расчет полного КПД по вашей формуле с учетом матрицы R и A"""
    # Коэффициенты отражения (замените на свои формулы при необходимости)
    Rs_theta = 0.1
    Rp_theta = 0.05
    # Аналитическая часть
    part1 = np.cos(theta) * ((1 - Rs_theta) * np.cos(alpha)**2 + (1 - Rp_theta) * np.sin(alpha)**2)
    # Формируем матрицу R (3x3)
    R = np.array([
        [np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)],
        [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)],
        [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]
    ])
    # Формируем матрицу A (2x2) по вашей формуле
    R11, R12, R21, R22 = R[0,0], R[0,1], R[1,0], R[1,1]
    A_mat = np.array([
        [R11**2 + R12**2, R11*R21 + R12*R22],
        [R11*R21 + R12*R22, R21**2 + R22**2]
    ])
    try:
        A_inv = np.linalg.inv(A_mat)
    except np.linalg.LinAlgError:
        A_inv = np.eye(2)
    detA = np.linalg.det(A_mat)
    BTAinvB = np.dot(B_vec.T, np.dot(A_inv, B_vec))
    geom_factor = (I0 * omega**2) / (2 * P0 * np.sqrt(detA)) * np.exp(-2 * (C - BTAinvB) / omega**2)
    # Пределы интегрирования по площади: квадрат со стороной a = sqrt(area)
    a = np.sqrt(area)
    u1_min, u1_max = -a/2, a/2
    u2_min, u2_max = -a/2, a/2
    # Численный интеграл
    def integrand(u1, u2):
        return np.exp(-u1**2 - u2**2)
    integral, _ = dblquad(
        lambda u2, u1: integrand(u1, u2),
        u1_min, u1_max,
        lambda u1: u2_min,
        lambda u1: u2_max
    )
    # Итоговый КПД
    eta = part1 * geom_factor * integral
    return eta

def my_area(area):
    """Замените на свою функцию площади, если она зависит от других переменных"""
    return area

def my_cost(area):
    """Замените на свою функцию стоимости"""
    base_cost = 500
    return base_cost + 2000 * area

# ==== Целевая функция ====
def evaluate(ind):
    area = ind[0]
    alpha = ind[1]
    beta = ind[2]
    gamma = ind[3]
    theta = ind[4]
    I0 = ind[5]
    omega = ind[6]
    P0 = ind[7]
    C = ind[8]
    B1 = ind[9]
    B2 = ind[10]
    B_vec = np.array([B1, B2])
    eff = my_efficiency(area, alpha, beta, gamma, theta, I0, omega, P0, C, B_vec)
    area_val = my_area(area)
    cost_val = my_cost(area)
    # КПД максимизируем, остальные минимизируем
    return (-eff, area_val, cost_val)

# ==== Настройка DEAP ====
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_area", np.random.uniform, AREA_MIN, AREA_MAX)
toolbox.register("attr_alpha", np.random.uniform, ALPHA_MIN, ALPHA_MAX)
toolbox.register("attr_beta", np.random.uniform, BETA_MIN, BETA_MAX)
toolbox.register("attr_gamma", np.random.uniform, GAMMA_MIN, GAMMA_MAX)
toolbox.register("attr_theta", np.random.uniform, THETA_MIN, THETA_MAX)
toolbox.register("attr_I0", np.random.uniform, I0_MIN, I0_MAX)
toolbox.register("attr_omega", np.random.uniform, OMEGA_MIN, OMEGA_MAX)
toolbox.register("attr_P0", np.random.uniform, P0_MIN, P0_MAX)
toolbox.register("attr_C", np.random.uniform, C_MIN, C_MAX)
toolbox.register("attr_B1", np.random.uniform, B_MIN, B_MAX)
toolbox.register("attr_B2", np.random.uniform, B_MIN, B_MAX)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_area, toolbox.attr_alpha, toolbox.attr_beta, toolbox.attr_gamma, toolbox.attr_theta,
                  toolbox.attr_I0, toolbox.attr_omega, toolbox.attr_P0, toolbox.attr_C,
                  toolbox.attr_B1, toolbox.attr_B2), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.3)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

# ==== Основной цикл ====
def main():
    pop = toolbox.population(n=100)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, mu=200, lambda_=400, cxpb=0.7, mutpb=0.3, ngen=100,
                             stats=stats, halloffame=hof, verbose=True)

    # ==== Сбор данных для графика ====
    effs = []
    areas = []
    costs = []
    for ind in hof:
        eff, area, cost = ind.fitness.values
        effs.append(-eff)  # Минус, чтобы КПД был положительным
        areas.append(area)
        costs.append(cost)

    # ==== График Парето-множества ====
    # 2d: КПД vs площадь
    plt.figure(figsize=(8,6))
    plt.scatter(areas, effs, c=costs, cmap='viridis', s=40)
    plt.xlabel("Площадь фотоприёмника, м²")
    plt.ylabel("КПД")
    plt.title("Парето-множество: КПД vs Площадь (цвет = стоимость)")
    cbar = plt.colorbar()
    cbar.set_label('Стоимость')
    plt.tight_layout()
    plt.show()

    # 3d: КПД vs площадь vs стоимость
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(areas, effs, costs, c=effs, cmap='plasma')
    ax.set_xlabel('Площадь, м²')
    ax.set_ylabel('КПД')
    ax.set_zlabel('Стоимость')
    plt.title("Парето-множество (3D)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
