import numpy as np

dt = 1 / 252
r_f = 0.05
u_s = 0.1
eta_x = -0.2
theta = 4
gamma_x = 2
rho = -0.9
x_bar = np.log(0.1)

S_0 = 1

x_0 = x_bar
T = 126

n_paths = 100000
K = 0.0001


def advance(S, x):
    assert S.shape == x.shape
    v_t = np.exp(x)
    es = np.random.normal(shape=S.shape)
    ev = np.random.normal(shape=S.shape)

    new_S = S * np.exp((r_f - 0.5 * v_t**2) * dt + v_t * np.sqrt(dt) * es)
    eta_st = (u_s - r_f) / v_t
    new_x = x - theta * (x - x_bar) * dt - (
        rho * eta_st +
        np.sqrt(1 - rho**2) * eta_x) * gamma_x * dt + gamma_x * np.sqrt(dt) * (
            rho * es + np.sqrt(1 - rho**2) * ev)
    return new_S, new_x


def one_hot(array, depth):
    return np.eye(depth)[array]


def compute_price():
    S = []
    S.append(np.ones(shape=n_paths) * S_0)
    x = []
    x.append(np.ones(shape=n_paths) * x_bar)

    for i in range(T):
        new_S, new_x = advance(S[-1], x[-1])
        S.append(new_S)
        x.append(new_x)

    rv = {}

    def RV(t):
        s = 0
        c = 0
        for u in range(max(1, t - 10), t):
            c += 1
            s += (np.log(S[u]) - np.log(S[u - 1]))**2
        return s / c

    for i in range(T + 1):
        rv[i] = RV(i)

    underlying = {}

    for i in range(T + 1):
        underlying[i] = rv[i]

    discount = np.exp(-r_f * dt)

    value = underlying[T] * discount
    CV = {T: np.zeros_like(S[0])}

    for t in range(T - 1, 0, -1):
        X = np.column_stack([x[t], rv[t], x[t]**2, x[t] * rv[t], rv[t]**2])
        Y = value

        theta = np.linalg.lstsq(X, Y)[0]
        CV[t] = X @ theta
        exercise = underlying[t] > CV[t]
        value = discount * np.where(exercise, underlying[t], value)

    payoffs = {
        t: np.where(CV[t] < underlying[t], underlying[t], 0)
        for t in range(1, m + 1)
    }
    payoffs = np.vstack(payoffs.values())

    idx_payoffs = np.argmax(payoffs > 0, axis=0)
    first_payoff = one_hot(idx_payoffs, T).T * payoffs

    T_range = np.array(range(T)).shape(-1, 1)
    discounted_payoff = first_payoff * np.exp(-r_f * T_range * dt)
    return discounted_payoff.sum() / n_paths


print(f'Price=compute_price()')
