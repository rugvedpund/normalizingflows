import numpy as np
import NormalizingFlow as nf
import matplotlib.pyplot as plt

##---------------------------------------------------------------------------##

npoints = [70, 71, 69]  # THIS IS WEIRD BUT CORRECT
truths = [40, 14, 16.4]
mins = [0.8 * 40, 13.0, 14.0]
maxs = [3.0 * 40, 17.0, 18.0]
limits = dict()
limits["amin"], limits["wmin"], limits["nmin"] = mins
limits["amax"], limits["wmax"], limits["nmax"] = maxs
limits["a"] = np.linspace(mins[0], maxs[0], npoints[1])
limits["w"] = np.linspace(mins[1], maxs[1], npoints[0])
limits["n"] = np.linspace(mins[2], maxs[2], npoints[2])
limits["da"] = (limits["a"][1] - limits["a"][0]) / 2
limits["dw"] = (limits["w"][1] - limits["w"][0]) / 2
limits["dn"] = (limits["n"][1] - limits["n"][0]) / 2
extent = dict()
extent["WvA"] = [
    limits["amin"] - limits["da"],
    limits["amax"] + limits["da"],
    limits["wmin"] - limits["dw"],
    limits["wmax"] + limits["dw"],
]
extent["NvA"] = [
    limits["amin"] - limits["da"],
    limits["amax"] + limits["da"],
    limits["nmin"] - limits["dn"],
    limits["nmax"] + limits["dn"],
]
extent["NvW"] = [
    limits["wmin"] - limits["dw"],
    limits["wmax"] + limits["dw"],
    limits["nmin"] - limits["dn"],
    limits["nmax"] + limits["dn"],
]
csColors = {"": "gray", "4": "C2", "4 6": "C3"}
labels = {
    "": "$2^\circ$",
    "4": "$2^\circ+4^\circ$",
    "4 6": "$2^\circ+4^\circ+6^\circ$",
}
axlabels = {
    "a": "" r"$A$",
    "w": r"$\nu_{\rm rms}$",
    "n": r"$\nu_{\rm min}$",
}
units = {
    "a": "[mK]",
    "w": "[MHz]",
    "n": "[MHz]",
}
##---------------------------------------------------------------------------##
# helper functions


def norm(x):
    return x / np.max(x)


def quantile(x, likelihoods):
    q16, q50, q84 = corner.core.quantile(x, [0.16, 0.5, 0.84], weights=likelihoods)
    qm, qp = q50 - q16, q84 - q50
    return rf"${q50:.2f}^{{+{qp:.2f}}}_{{-{qm:.2f}}}$"


##---------------------------------------------------------------------------##

args, s, ll, ls = dict(), dict(), dict(), dict()

for cs in ["", "4", "4 6"]:
    args[cs] = nf.Args(SNRpp=1e24, appendLik="_smallcube", combineSigma=cs)
    s[cs], ll[cs] = nf.get_samplesAndLikelihood(args[cs], plot="all")
    s[cs][:, 0] *= 40
    ls[cs] = nf.exp(ll[cs] - max(ll[cs])).reshape(npoints)

fig, ax = plt.subplots(3, 3, figsize=(8, 8), sharex="col")
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.05)

for cs in ["", "4", "4 6"]:
    ax[0, 0].step(limits["a"], norm(ls[cs].sum(axis=(0, 2))), c=csColors[cs])
    ax[1, 1].step(limits["w"], norm(ls[cs].sum(axis=(1, 2))), c=csColors[cs])
    ax[2, 2].step(limits["n"], norm(ls[cs].sum(axis=(0, 1))), c=csColors[cs])

ax[1, 0].imshow(
    norm(np.dstack([ls[cs].sum(axis=2) for cs in ["", "4", "4 6"]])),
    extent=extent["WvA"],
    origin="lower",
    aspect="auto",
)
ax[2, 0].imshow(
    norm(np.dstack([ls[cs].sum(axis=0).T for cs in ["", "4", "4 6"]])),
    extent=extent["NvA"],
    origin="lower",
    aspect="auto",
)
im = ax[2, 1].imshow(
    norm(np.dstack([ls[cs].sum(axis=1).T for cs in ["", "4", "4 6"]])),
    extent=extent["NvW"],
    origin="lower",
    aspect="auto",
)

# ax[0, 0].step(limits["a"], likelihoods.sum(axis=(0, 2)), c="gray")
# ax[1, 1].step(limits["w"], likelihoods.sum(axis=(1, 2)), c="gray")
# ax[2, 2].step(limits["n"], likelihoods.sum(axis=(0, 1)), c="gray")

# # THESE TRANSPOSES ARE WEIRD BUT CORRECT
# ax[1, 0].imshow(
#     norm(likelihoods.sum(axis=2)),
#     extent=extent["WvA"],
#     origin="lower",
#     aspect="auto",
#     cmap="Greys",
# )
# ax[2, 0].imshow(
#     norm(likelihoods.sum(axis=0)).T,
#     extent=extent["NvA"],
#     origin="lower",
#     aspect="auto",
#     cmap="Greys",
# )
# im = ax[2, 1].imshow(
#     norm(likelihoods.sum(axis=1)).T,
#     extent=extent["NvW"],
#     origin="lower",
#     aspect="auto",
#     cmap="Greys",
# )

plt.colorbar(im, ax=ax[0, 2], label="Normalized Likelihood", orientation="vertical")
ax[0, 2].errorbar([], [], fmt="ro", xerr=[], yerr=[], label="Truth")
ax[0, 2].legend(loc="center")

ax[2, 0].set_xlabel(axlabels["a"] + " " + units["a"])
ax[2, 1].set_xlabel(axlabels["w"] + " " + units["w"])
ax[2, 2].set_xlabel(axlabels["n"] + " " + units["n"])
ax[1, 0].set_ylabel(axlabels["w"] + " " + units["w"])
ax[2, 0].set_ylabel(axlabels["n"] + " " + units["n"])

ax[0, 0].set_yticks([])
ax[1, 1].set_yticks([])
ax[2, 2].set_yticks([])

ax[2, 1].set_yticklabels([])

[l.set_rotation(45) for l in ax[2, 0].get_xticklabels()]
[l.set_rotation(45) for l in ax[2, 1].get_xticklabels()]
[l.set_rotation(45) for l in ax[2, 2].get_xticklabels()]
[l.set_rotation(45) for l in ax[1, 0].get_yticklabels()]
[l.set_rotation(45) for l in ax[2, 0].get_yticklabels()]

ax[1, 0].axhline(truths[1], c="r", alpha=0.5, lw=0.4)
ax[2, 0].axhline(truths[2], c="r", alpha=0.5, lw=0.4)
ax[2, 1].axhline(truths[2], c="r", alpha=0.5, lw=0.4)

ax[0, 0].axvline(truths[0], c="r", alpha=0.5, lw=0.4)
ax[1, 0].axvline(truths[0], c="r", alpha=0.5, lw=0.4)
ax[1, 1].axvline(truths[1], c="r", alpha=0.5, lw=0.4)
ax[2, 0].axvline(truths[0], c="r", alpha=0.5, lw=0.4)
ax[2, 1].axvline(truths[1], c="r", alpha=0.5, lw=0.4)
ax[2, 2].axvline(truths[2], c="r", alpha=0.5, lw=0.4)

ax[1, 0].plot(truths[0], truths[1], "ro")
ax[2, 0].plot(truths[0], truths[2], "ro")
ax[2, 1].plot(truths[1], truths[2], "ro")

ax[0, 1].set_visible(False)
ax[0, 2].axis("off")
ax[1, 2].set_visible(False)

# plt.figtext(
#     0.8,
#     0.7,
#     parser.argstostring(args),
#     bbox=dict(facecolor="gray", alpha=0.5),
#     fontsize="xx-small",
#     ma="center",
# )

plt.show()
