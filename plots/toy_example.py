import numpy as np
import matplotlib.pyplot as plt
from maxsmooth.DCF import smooth
from maxsmooth.best_basis import basis_test
from maxsmooth.chidist_plotter import chi_plotter

##---------------------------------------------------------------------------##

# This script produces slightly different results for each run since maxsmooth uses random initialization

f = np.linspace(10, 50)
fn = f / 30
A = 1e5
idx1 = 2.54
idx2 = 2.56

P1 = A * (1 / fn ** idx1)
P2 = A * (1 / fn ** idx2)
P = P1 + P2

# basis_test(f, P, base_dir="./maxsmooth/")


Ppoly = smooth(f, P, 15, model_type="difference_polynomial", base_dir='./maxsmooth/').y_fit
Plogpoly = smooth(f, P, 14, model_type="log_polynomial", base_dir='./maxsmooth/').y_fit
Ploglogpoly = smooth(f, P, 15, model_type="loglog_polynomial", base_dir='./maxsmooth/').y_fit

resPpoly = P - Ppoly
resPlogpoly = P - Plogpoly
resPloglogpoly = P - Ploglogpoly

deltapoly = np.max(np.abs(resPpoly))
deltalogpoly = np.max(np.abs(resPlogpoly))
deltaloglogpoly = np.max(np.abs(resPloglogpoly))

orderpoly = int(np.floor(np.log10(deltapoly)))
orderlogpoly = int(np.floor(np.log10(deltalogpoly)))
orderloglogpoly = int(np.floor(np.log10(deltaloglogpoly)))

##---------------------------------------------------------------------------##

fig, ax = plt.subplots(
    nrows=2,
    sharex=True,
    constrained_layout=True,
    gridspec_kw=dict(height_ratios=[2,1]),
)
cfg = "gray"
cplaw = "C2"
cres = "C2"
ax[0].plot(f, P1, ls="--", lw=1.5, label=f"Power Law 1, index = {idx1}")
ax[0].plot(f, P2, ls="--", lw=1.5, label=f"Power Law 2, index = {idx2}")
ax[0].plot(f, P, c=cfg, lw=2.5, markersize=2 ** 2, label="Foreground: P1 + P2")
ax[0].plot(f, Ppoly, ls="--", lw=1.5, label="Polynomial Fit",c="C3")
ax[0].plot(f, Plogpoly, ls="--", lw=1.5, label="Log Polynomial Fit", c="C4")
ax[0].plot(f, Ploglogpoly, ls="--", lw=1.5, label="Log-Log Fit", c="C2")


ax[1].plot(f, resPpoly/10**orderpoly, lw=1.5, label=rf"Polynomial Residual/$10^{orderpoly}$", c="C3")
ax[1].plot(f, resPlogpoly/10**orderlogpoly, lw=1.5, label=rf"Log Polynomial Residual/$10^{orderlogpoly}$", c="C4")
ax[1].plot(f, resPloglogpoly/10**orderloglogpoly, lw=1.5, label=rf"Log-Log Residual/$10^{orderloglogpoly}$", c="C2")

ax[0].set_title("Maximally Smooth Function Fitting to Sum of Power Laws")
ax[0].set_yscale("log")
ax[0].set_ylabel(r"$T$ [K]")
ax[0].legend(loc="lower left",fontsize="small")

ax[1].set_ylabel(r"$\Delta T$ [K]")
ax[1].legend(loc="upper right",fontsize="small")
ax[1].set_xlabel("Frequency [MHz]")

# ax[2].set_ylabel(r"$\Delta T$ [K]")
# ax[2].legend(loc="upper right",fontsize="small")

print("saving..")
plt.savefig("toy_example.pdf")
plt.show()
