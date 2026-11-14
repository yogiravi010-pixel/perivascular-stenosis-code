import numpy as np
from scipy.linalg import solve_banded

kappa = 0.65; L = 30.0; Ls = 10.0; zc = 15.0
Q0 = (1 - kappa**2) / 2

def Ri(z, s, kap=kappa, Lsv=Ls):
    z = np.atleast_1d(z)
    out = np.full_like(z, kap, dtype=float)
    m = np.abs(z - zc) <= Lsv / 2
    out[m] = kap + s / 2 * (1 - kap) * (1 + np.cos(2 * np.pi * (z[m] - zc) / Lsv))
    return out

def williamson(We, m):
    return lambda g: m + (1 - m) / (1 + We * g)

def carreau_yasuda(lam, n, a, m):
    # mu/mu0 = m + (1-m) [1+(lam g)^a]^((n-1)/a)
    return lambda g: m + (1 - m) * (1 + (lam * g) ** a) ** ((n - 1) / a)

def station(Rin, Q, muf, M=0.0, N=160, relax=0.7, tol=1e-10, maxit=200, ret=False):
    r = np.linspace(Rin, 1.0, N + 1); h = r[1] - r[0]
    rh = 0.5 * (r[:-1] + r[1:])
    mu = np.ones(N + 1)
    for it in range(maxit):
        muh = 0.5 * (mu[:-1] + mu[1:])
        n_int = N - 1
        ab = np.zeros((3, n_int))
        rr = r[1:-1]
        lo = rh[:-1] * muh[:-1] / h**2
        up = rh[1:] * muh[1:] / h**2
        diag = -(lo + up) - M**2 / rr
        ab[1] = -diag; ab[0, 1:] = -up[:-1]; ab[2, :-1] = -lo[1:]
        rhs = rr.copy()
        v = np.zeros(N + 1)
        v[1:-1] = solve_banded((1, 1), ab, rhs)
        G = Q / np.trapezoid(r * v, r)
        u = G * v
        du = np.zeros(N + 1)
        du[1:-1] = (u[2:] - u[:-2]) / (2 * h)
        du[0] = (-3 * u[0] + 4 * u[1] - u[2]) / (2 * h)
        du[-1] = (3 * u[-1] - 4 * u[-2] + u[-3]) / (2 * h)
        munew = muf(np.abs(du))
        err = np.max(np.abs(munew - mu))
        mu = mu + relax * (munew - mu)
        if err < tol: break
    if ret:
        return G, r, u, du, mu
    return G

def dp(s, Q, muf, M=0.0, N=160, nz=81, **kw):
    z = np.linspace(0, L, nz)
    Rz = Ri(z, s, **kw)
    # cache identical stations
    cache = {}
    Gs = []
    for R in Rz:
        key = round(R, 14)
        if key not in cache:
            cache[key] = station(R, Q, muf, M, N)
        Gs.append(cache[key])
    return np.trapezoid(Gs, z), np.array(Gs)

def Phi(s, muf, M=0.0, N=160, nz=81):
    target = dp(0.0, Q0, muf, M, N, nz)[0]
    lo, hi = 1e-4 * Q0, 4 * Q0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dp(s, mid, muf, M, N, nz)[0] > target: hi = mid
        else: lo = mid
        if (hi - lo) / mid < 1e-6: break
    return 0.5 * (lo + hi) / Q0

if __name__ == "__main__":
    for We in [0.0, 0.1]:
        f = williamson(We, 0.4)
        d0 = dp(0, Q0, f)[0]
        print(We, d0, [round(dp(s, Q0, f)[0] / d0, 3) for s in (0.5, 0.7)])
