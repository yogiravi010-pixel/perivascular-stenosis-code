import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from solver import *

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "lines.linewidth": 1.3, "figure.dpi": 150, "mathtext.fontset": "cm"})
os.makedirs("figs", exist_ok=True)
U0, Ro, mu0 = 20e-6, 30e-6, 2.5e-3
sc = U0 / Ro
NZ = 61
def cross(lam_hat, k, m): return lambda g: m + (1 - m) / (1 + (lam_hat * g) ** k)
LAW = {
    "N": williamson(0.0, 0.4),
    "W": williamson(0.1, 0.4),
    "CY": carreau_yasuda(0.3 * sc, 0.3, 2.0, 0.4),
    "X": cross((1 / 3.0) * sc, 2.0, 0.4),
}
D = {}
area = lambda s: 1 - (1 - (kappa + s * (1 - kappa)) ** 2) / (1 - kappa**2)

# ---- Table: Psi vs We, dp0
Wes = [0.0, 0.03, 0.1, 0.3, 1.0]; ss = [0.2, 0.4, 0.5, 0.6, 0.7]
T = {}
for We in Wes:
    f = williamson(We, 0.4)
    d0 = dp(0, Q0, f, nz=NZ)[0]
    T[We] = {"dp0": d0, **{s: dp(s, Q0, f, nz=NZ)[0] / d0 for s in ss}}
D["psi_table"] = {str(k): {str(a): b for a, b in v.items()} for k, v in T.items()}
print("psi table done")

# ---- Phi table
PH = {}
for We in [0.0, 0.1, 0.3, 1.0]:
    f = williamson(We, 0.4)
    PH[We] = {s: Phi(s, f, nz=NZ) for s in (0.4, 0.5, 0.6, 0.7)}
D["phi_table"] = {str(k): {str(a): b for a, b in v.items()} for k, v in PH.items()}
print("phi done")

# ---- Magnetic table
MT = {}
f = williamson(0.1, 0.4)
for M in (0.0, 0.5, 1.0, 2.0, 3.0):
    d0 = dp(0, Q0, f, M=M, nz=NZ)[0]; d5 = dp(0.5, Q0, f, M=M, nz=NZ)[0]; d7 = dp(0.7, Q0, f, M=M, nz=NZ)[0]
    MT[M] = {"d0": d0, "d5": d5, "psi5": d5 / d0, "psi7": d7 / d0, "phi5": Phi(0.5, f, M=M, nz=NZ)}
D["mag_table"] = {str(k): v for k, v in MT.items()}
print("mag done")

# ---- Law comparison
LC = {}
for name, f in LAW.items():
    d0 = dp(0, Q0, f, nz=NZ)[0]
    row = {"dp0": d0}
    for s in (0.5, 0.7):
        d = dp(s, Q0, f, nz=NZ)[0]
        row[f"dp{s}"] = d; row[f"psi{s}"] = d / d0; row[f"phi{s}"] = Phi(s, f, nz=NZ)
    for s in (0.0, 0.5, 0.7):
        G, r, u, du, mu = station(kappa + s * (1 - kappa), Q0, f, ret=True)
        row[f"gw{s}"] = abs(du[0]) * sc; row[f"muw{s}"] = mu[0]
        row[f"mumax{s}"] = float(mu.max()); row[f"mumid{s}"] = float(np.interp(0.5 * (r[0] + 1), r, mu))
    LC[name] = row
D["law"] = LC
print("law done")

# ---- Pulsatile via interpolation of dp(Q)
def dpQ_interp(s, f):
    qs = Q0 * np.logspace(-3.5, 1.1, 70)
    vals = np.array([dp(s, q, f, nz=NZ)[0] for q in qs])
    return PchipInterpolator(np.log(qs), np.log(vals))
def psiA(s, f, A, interp_s, interp_0, nth=8192):
    th = 2 * np.pi * (np.arange(nth) + 0.5) / nth
    Q = Q0 * (1 + A * np.sin(th))
    def sdp(interp):
        return np.sign(Q) * np.exp(interp(np.log(np.abs(Q))))
    return np.mean(sdp(interp_s)) / np.mean(sdp(interp_0))
PL = {}
for name in ("W", "CY", "X"):
    f = LAW[name]; i0 = dpQ_interp(0.0, f)
    PL[name] = {}
    for s in (0.5, 0.7):
        i_s = dpQ_interp(s, f)
        PL[name][str(s)] = {str(A): psiA(s, f, A, i_s, i0) for A in (0, 1, 3, 10)}
D["pulse"] = PL
print("pulse done", PL)

# ---- Figure data
# Fig 1: geometry + wall shear rate (Williamson)
z = np.linspace(0, L, 301)
fW = LAW["W"]
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
for s, ls in ((0, "-"), (0.5, "--"), (0.7, ":")):
    ax[0].plot(z, Ri(z, s), ls, label=f"$s={s}$")
ax[0].axhline(1.0, color="k", lw=1.6)
ax[0].set_xlabel("axial position $z$"); ax[0].set_ylabel("radius $r$"); ax[0].legend(frameon=False, loc="center left", fontsize=7)
ax[0].set_ylim(0.58, 1.04); ax[0].set_title("(a) gap geometry (outer wall $r=1$)", fontsize=8)
zs = np.linspace(0, L, 121); cache = {}
for s, ls in ((0, "-"), (0.5, "--"), (0.7, ":")):
    gw = []
    for R in Ri(zs, s):
        k = round(float(R), 12)
        if k not in cache:
            G, r, u, du, mu = station(R, Q0, fW, ret=True); cache[k] = abs(du[0]) * sc
        gw.append(cache[k])
    ax[1].plot(zs, gw, ls, label=f"$s={s}$")
ax[1].axhline(10, color="gray", lw=0.7, ls="-."); ax[1].text(0.5, 12, "10 s$^{-1}$", fontsize=7, color="gray")
ax[1].set_xlabel("axial position $z$"); ax[1].set_ylabel("wall shear rate (s$^{-1}$)"); ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("(b) arterial-wall shear rate, $We=0.1$", fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig1.pdf"); plt.close()

# Fig 2: pressure gradient (s=0.5) and Psi vs area loss
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
for We, ls in ((0.0, "-"), (0.1, "--"), (1.0, ":")):
    f = williamson(We, 0.4); Gs = dp(0.5, Q0, f, nz=NZ)[1]; zz = np.linspace(0, L, NZ)
    zf = np.linspace(0, L, 301)
    Gf = [station(R, Q0, f) for R in Ri(zf, 0.5)] if False else None
    ax[0].plot(zz, Gs, ls, label=f"$We={We:g}$")
ax[0].set_xlabel("axial position $z$"); ax[0].set_ylabel("$G=-dp/dz$"); ax[0].legend(frameon=False, fontsize=7)
ax[0].set_title("(a) pressure gradient, $s=0.5$, fixed flux", fontsize=8)
sg = np.linspace(0, 0.7, 15)
for We, ls in ((0.0, "-"), (0.1, "--"), (0.3, "-."), (1.0, ":")):
    f = williamson(We, 0.4); d0 = dp(0, Q0, f, nz=NZ)[0]
    ax[1].plot([100 * area(s) for s in sg], [dp(s, Q0, f, nz=NZ)[0] / d0 for s in sg], ls, label=f"$We={We:g}$")
ax[1].set_xlabel("loss of PVS area (%)"); ax[1].set_ylabel("resistance index $\\Psi$"); ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("(b) resistance index vs. area loss", fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig2.pdf"); plt.close()

# Fig 3: Psi vs We and viscosity laws
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
Wsweep = np.logspace(-2, 2, 21)
psi_n = {0.5: 1.7080, 0.7: 3.5803}
for s, c in ((0.5, "C0"), (0.7, "C3")):
    ys = []
    for We in Wsweep:
        f = williamson(We, 0.4); ys.append(dp(s, Q0, f, nz=NZ)[0] / dp(0, Q0, f, nz=NZ)[0])
    ax[0].semilogx(Wsweep, ys, c, label=f"$s={s}$")
    ax[0].axhline(psi_n[s], color=c, ls="--", lw=0.7)
ax[0].set_xlabel("Williamson number $We$"); ax[0].set_ylabel("resistance index $\\Psi$"); ax[0].legend(frameon=False, fontsize=7, loc="center right")
ax[0].set_title("(a) non-monotonic shielding (dashed: Newtonian)", fontsize=8)
g = np.logspace(-1, 2.5, 200)
for name, lab, ls in (("W", "Williamson", "-"), ("CY", "Carreau-Yasuda", "--"), ("X", "Cross, $k=2$", "-.")):
    ax[1].semilogx(g, 2.5 * LAW[name](g / sc), ls, label=lab)
ax[1].axvline(10, color="gray", lw=0.7, ls=":"); ax[1].axvline(13.8, color="k", lw=0.5, ls=":"); ax[1].axvline(45.9, color="k", lw=0.5, ls=":"); ax[1].axvline(116.8, color="k", lw=0.5, ls=":")
ax[1].axhline(1.0, color="gray", lw=0.6)
ax[1].set_xlabel("shear rate (s$^{-1}$)"); ax[1].set_ylabel("viscosity (mPa s)"); ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("(b) viscosity laws; dotted: 10, 14, 46, 117 s$^{-1}$", fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig3.pdf"); plt.close()

# Fig 4: pulsatile
fig, ax = plt.subplots(figsize=(3.4, 2.4))
Aq = [0, 1, 3, 10]
for name, lab, mk in (("W", "Williamson", "o-"), ("CY", "Carreau-Yasuda", "s--"), ("X", "Cross, $k=2$", "^-.")):
    ax.plot(Aq, [PL[name]["0.7"][str(A)] for A in Aq], mk, ms=3, label=lab)
ax.axhline(3.5803, color="k", ls=":", lw=0.8); ax.text(0.2, 3.60, "Newtonian", fontsize=7)
ax.set_xlabel("oscillation ratio $A$"); ax.set_ylabel("cycle-averaged $\\Psi$ ($s=0.7$)"); ax.legend(frameon=False, fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig("figs/fig4.pdf"); plt.close()

# ---- verification numbers
def exact_poiseuille_G():
    k = kappa
    return 16 * Q0 / ((1 - k**4) - (1 - k**2) ** 2 / np.log(1 / k))
Gh, r, u, du, mu = station(kappa, Q0, LAW["N"], N=400, ret=True)
vv = {"G_num": Gh, "G_ex": exact_poiseuille_G()}
# exact profile
k = kappa; Gex = exact_poiseuille_G()
A_ = Gex * (1 - k**2) / (4 * np.log(1 / k)); B_ = -Gex / 4
uex = -Gex * r**2 / 4 + A_ * np.log(r) * 0  # placeholder replaced below
# u = -G r^2/4 + C1 ln r + C2 with u(k)=u(1)=0
C1 = Gex * (1 - k**2) / (4 * np.log(1 / k)) * (-1) * (-1)
C1 = (Gex / 4) * (1 - k**2) / np.log(1 / k) * (-1) * (-1)
# solve directly
M_ = np.array([[np.log(k), 1.0], [0.0, 1.0]]); rhs_ = np.array([Gex * k**2 / 4, Gex / 4])
C1, C2 = np.linalg.solve(M_, rhs_)
uex = -Gex * r**2 / 4 + C1 * np.log(r) + C2
vv["max_err_u"] = float(np.max(np.abs(uex - u)))
# MHD M=1: u = -G r^2/3 + A r + B/r
Gm = 1.0
Mx = np.array([[k, 1 / k], [1.0, 1.0]]); rx = np.array([Gm * k**2 / 3, Gm / 3]); Aa, Bb = np.linalg.solve(Mx, rx)
rr = np.linspace(k, 1, 20001); uexm = -Gm * rr**2 / 3 + Aa * rr + Bb / rr
Qexm = np.trapezoid(rr * uexm, rr)
G1, r1, u1, du1, mu1 = station(kappa, Qexm, LAW["N"], M=1.0, N=400, ret=True)
vv["mhd_G_over_exact"] = G1
vv["mhd_max_err_u"] = float(np.max(np.abs(u1 - (-Gm * r1**2 / 3 + Aa * r1 + Bb / r1))))
vv["mhd_flux_exact"] = Qexm
Gl = station(kappa, Q0, williamson(1e6, 0.4), N=400); G0_ = station(kappa, Q0, williamson(0, 0.4), N=400)
vv["ratio_We1e6"] = Gl / G0_
# grid refinement s=0.5 We=0.3 M=1 61 stations
fW3 = williamson(0.3, 0.4); gr = {}
for N in (50, 100, 200, 400, 800):
    gr[N] = dp(0.5, Q0, fW3, M=1.0, N=N, nz=61)[0]
vv["grid"] = {str(k_): v for k_, v in gr.items()}
D["verif"] = vv
print(vv)
json.dump(D, open("data.json", "w"), indent=1, default=float)
