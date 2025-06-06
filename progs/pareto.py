import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from scipy.integrate import dblquad

# ==== Границы переменных ====
W_MIN, W_MAX = 0.01, 0.5         # ширина, м
H_MIN, H_MAX = 0.01, 0.25        # высота, м
P_MIN, P_MAX = 0.005, 0.05       # шаг, м
AREA_MIN, AREA_MAX = 0.001, 0.1  # площадь, м^2
ALPHA_MIN, ALPHA_MAX = 0, np.pi/2
BETA_MIN, BETA_MAX = 0, np.pi/2
GAMMA_MIN, GAMMA_MAX = 0, np.pi/2
THETA_MIN, THETA_MAX = 0, np.pi/2
I0_MIN, I0_MAX = 200, 1274.0
OMEGA_MIN, OMEGA_MAX = 0.5, 2.0
P0_MIN, P0_MAX = 0.5, 2000.0
LAMBDA_MIN, LAMBDA_MAX = 0.4e-6, 2e-6  # длина волны, м
Z_MIN, Z_MAX = 0.01, 1.0               # расстояние до фотоприёмника, м
W0_MIN, W0_MAX = 0.001, 0.02           # минимальная ширина пучка, м
X0_MIN, X0_MAX = -0.05, 0.05           # координата центра по x, м
Y0_MIN, Y0_MAX = -0.05, 0.05           # координата центра по y, м

def my_area(W, H, p):
    if p <= 0 or W <= 0 or H <= 0:
        return 0.0
    Nw = int(W // p)
    Nh = int(H // p)
    if Nw < 1 or Nh < 1:
        return 0.0
    N = Nw * Nh
    area = N * p * p
    return area

def my_efficiency(area, alpha, beta, gamma, theta, I0, omega, P0, 
                  x0, y0, w0, z, lambda_):
    if area < 1e-7 or not np.isfinite(area):
        return 0.0

    Rs_theta = 0.1
    Rp_theta = 0.05

    part1 = np.cos(theta) * ((1 - Rs_theta) * np.cos(alpha)**2 + (1 - Rp_theta) * np.sin(alpha)**2)

    R = np.array([
        [np.cos(alpha)*np.cos(beta),
         np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma),
         np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)],
        [np.sin(alpha)*np.cos(beta),
         np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma),
         np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)],
        [-np.sin(beta),
         np.cos(beta)*np.sin(gamma),
         np.cos(beta)*np.cos(gamma)]
    ])
    R11, R12, R21, R22 = R[0,0], R[0,1], R[1,0], R[1,1]
    B_vec = np.array([
        R11 * x0 + R12 * y0,
        R21 * x0 + R22 * y0
    ])
    C_val = x0**2 + y0**2

    A_mat = np.array([
        [R11**2 + R12**2, R11*R21 + R12*R22],
        [R11*R21 + R12*R22, R21**2 + R22**2]
    ])
    try:
        detA = np.linalg.det(A_mat)
    except:
        return 0.0
    if detA < 1e-8 or not np.isfinite(detA):
        return 0.0
    try:
        A_inv = np.linalg.inv(A_mat)
    except np.linalg.LinAlgError:
        return 0.0
    try:
        BTAinvB = np.dot(B_vec.T, np.dot(A_inv, B_vec))
    except:
        return 0.0

    if omega == 0 or P0 == 0:
        return 0.0

    zr = np.pi * w0 ** 2 / lambda_
    w_z = w0 * np.sqrt(1 + (z / zr) ** 2)

    try:
        val_exp = -2 * (C_val - BTAinvB) / (omega**2)
        if val_exp < -700:
            exp_val = 0.0
        elif val_exp > 700:
            return 0.0
        else:
            exp_val = np.exp(val_exp)
        geom_factor = (I0 * omega**2) / (2 * P0 * np.sqrt(detA)) * exp_val
    except:
        return 0.0

    a = np.sqrt(area)
    u1_min, u1_max = -a/2, a/2
    u2_min, u2_max = -a/2, a/2

    try:
        integral, _ = dblquad(
            lambda u2, u1: np.exp(-2 * ((u1**2 + u2**2)/(w_z ** 2))),
            u1_min, u1_max,
            lambda u1: u2_min,
            lambda u1: u2_max
        )
    except Exception:
        return 0.0

    eta = part1 * geom_factor * integral
    if not np.isfinite(eta) or eta < 0 or eta > 1:
        return 0.0
    return eta

def evaluate(ind):
    W      = ind[0]
    H      = ind[1]
    p      = ind[2]
    alpha  = ind[3]
    beta   = ind[4]
    gamma  = ind[5]
    theta  = ind[6]
    I0     = ind[7]
    omega  = ind[8]
    P0     = ind[9]
    lambda_ = ind[10]
    z       = ind[11]
    w0      = ind[12]
    x0      = ind[13]
    y0      = ind[14]

    if (W > W_MAX or W < W_MIN or H > H_MAX or H < H_MIN or p > P_MAX or p < P_MIN or
        lambda_ < LAMBDA_MIN or lambda_ > LAMBDA_MAX or
        w0 < W0_MIN or w0 > W0_MAX or z < Z_MIN or z > Z_MAX or
        x0 < X0_MIN or x0 > X0_MAX or y0 < Y0_MIN or y0 > Y0_MAX):
        return (1e6, 1e6)

    area_val = my_area(W, H, p)
    if area_val < AREA_MIN or area_val > AREA_MAX or not np.isfinite(area_val):
        return (1e6, 1e6)

    eff = my_efficiency(area_val, alpha, beta, gamma, theta, I0, omega, P0,
                        x0, y0, w0, z, lambda_)
    if not np.isfinite(eff) or eff <= 0 or eff > 1:
        return (1e6, 1e6)

    return (-eff, area_val)

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_W", np.random.uniform, W_MIN, W_MAX)
toolbox.register("attr_H", np.random.uniform, H_MIN, H_MAX)
toolbox.register("attr_p", np.random.uniform, P_MIN, P_MAX)
toolbox.register("attr_alpha", np.random.uniform, ALPHA_MIN, ALPHA_MAX)
toolbox.register("attr_beta", np.random.uniform, BETA_MIN, BETA_MAX)
toolbox.register("attr_gamma", np.random.uniform, GAMMA_MIN, GAMMA_MAX)
toolbox.register("attr_theta", np.random.uniform, THETA_MIN, THETA_MAX)
toolbox.register("attr_I0", np.random.uniform, I0_MIN, I0_MAX)
toolbox.register("attr_omega", np.random.uniform, OMEGA_MIN, OMEGA_MAX)
toolbox.register("attr_P0", np.random.uniform, P0_MIN, P0_MAX)
toolbox.register("attr_lambda", np.random.uniform, LAMBDA_MIN, LAMBDA_MAX)
toolbox.register("attr_z", np.random.uniform, Z_MIN, Z_MAX)
toolbox.register("attr_w0", np.random.uniform, W0_MIN, W0_MAX)
toolbox.register("attr_x0", np.random.uniform, X0_MIN, X0_MAX)
toolbox.register("attr_y0", np.random.uniform, Y0_MIN, Y0_MAX)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_W, toolbox.attr_H, toolbox.attr_p, toolbox.attr_alpha, toolbox.attr_beta, toolbox.attr_gamma,
                  toolbox.attr_theta, toolbox.attr_I0, toolbox.attr_omega, toolbox.attr_P0, toolbox.attr_lambda,
                  toolbox.attr_z, toolbox.attr_w0, toolbox.attr_x0, toolbox.attr_y0), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.3)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

def main():
    pop = toolbox.population(n=1000)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, mu=1000, lambda_=2000, cxpb=0.7, mutpb=0.3, ngen=50,
                              stats=stats, halloffame=hof, verbose=True)

    effs = []
    areas = []
    for ind in hof:
        eff, area = ind.fitness.values
        effs.append(-eff)
        areas.append(area)

    plt.figure(figsize=(8,6))
    plt.scatter(areas, effs, s=40)
    plt.xlabel("Площадь фотоприёмника, м²")
    plt.ylabel("КПД")
    plt.title("Парето-множество: КПД vs Площадь")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
