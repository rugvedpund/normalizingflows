# create the Dark Ages combineSigmas 3D plot

import NormalizingFlow as nf
import numpy as np
import matplotlib.pyplot as plt
import corner

args = nf.Args()
args.fgFITS, args.freqs, args.chromatic = "ulsa.fits", "1 51", False
args.SNRpp = 1e12
args.appendLik = "_walkers"

cosmicdawn = True if args.fgFITS == "gsm16.fits" else False
colors = {"": "gray", "4": "C2", "4 6": "C3"}
truths = [130, 20, 67.5] if cosmicdawn else [40, 14, 16.4]
labels = {
    "": r"$2^\circ$",
    "4": r"$2^\circ+4^\circ$",
    "4 6": r"$2^\circ+4^\circ+6^\circ$",
}
ranges = (
    [(110, 150), (18, 22), (65, 70)] if cosmicdawn else [(30, 60), (13.5, 16), (15, 18)]
)
fg = "Cosmic Dawn" if cosmicdawn else "Dark Ages"
cornerkwargs = {
    "show_titles": True,
    "levels": [1 - np.exp(-0.5), 1 - np.exp(-2)],
    "bins": 50,
    "range": ranges,
    "labels": [r"A", r"$\nu_{\rm rms}$", r"$\nu_{\rm min}$"],
    "plot_datapoints": True,
    # "hist_kwargs":{"density": True},
}
# do i need hist_kwargs={"density": True}

args.combineSigma = ""
s, ll = nf.get_samplesAndLikelihood(args, plot="all")
s[:, 0] *= truths[0]
fig = corner.corner(
    s,
    weights=nf.exp(ll),
    color=colors[args.combineSigma],
    **cornerkwargs,
)

args.combineSigma = "4"
s, ll = nf.get_samplesAndLikelihood(args, plot="all")
s[:, 0] *= truths[0]
corner.corner(
    s,
    weights=nf.exp(ll),
    color=colors[args.combineSigma],
    fig=fig,
    **cornerkwargs,
)
args.appendLik = "_game"
args.combineSigma = "4 6"
s, ll = nf.get_samplesAndLikelihood(args, plot="all")
s[:, 0] *= truths[0]
corner.corner(
    s,
    weights=nf.exp(ll),
    color=colors[args.combineSigma],
    fig=fig,
    **cornerkwargs,
)

corner.overplot_lines(fig, truths, color="k", ls="--", lw=1)
plt.plot([], [], c="k", ls="--", label="Truth")
for cs in ["", "4", "4 6"]:
    plt.plot([], [], c=colors[cs], label=labels[cs], lw=5)
plt.legend(bbox_to_anchor=(0, 2.5), loc="center left", borderaxespad=0)

plt.suptitle(f"{fg} Signal Constraints for SNR={args.SNRpp:.0e}")
# plt.savefig(f'plots/{fg}_combineSigmas.pdf',dpi=300,bbox_inches='tight')
plt.savefig(f"./test.pdf", dpi=300, bbox_inches="tight")
