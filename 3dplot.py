import numpy as np
import NormalizingFlow as nf
import corner
import matplotlib.pyplot as plt
import parser

##---------------------------------------------------------------------------##
# argparser

argparser = parser.create_parser()
args = argparser.parse_args()

# must set this
args.noisyT21 = True
args.diffCombineSigma = True

parser.prettyprint(args)

##---------------------------------------------------------------------------##
# setup corner plot params

cosmicdawn = True if args.fgFITS == "gsm16.fits" else False
truths = [130, 20, 67.5] if cosmicdawn else [40, 14, 16.4]
ranges = (
    [(125, 135), (19, 21), (66, 69)] if cosmicdawn else [(30, 80), (13.5, 16), (13, 18)]
)
fg = "CD" if cosmicdawn else "DA"
labels = {
    "": "$2^\circ$",
    "4": "$2^\circ+4^\circ$",
    "4 6": "$2^\circ+4^\circ+6^\circ$",
}
cornerkwargs = {
    "show_titles": True,
    "levels": [1 - np.exp(-0.5), 1 - np.exp(-2)],
    "bins": 40,
    "range": ranges,
    "labels": [r"A", r"$\nu_{\rm rms}$", r"$\nu_{\rm min}$"],
    "plot_datapoints": True,
    # "hist_kwargs":{"density": True},
}
suptitle=rf"{fg} for {labels[args.combineSigma]},"
if args.SNRpp is not None: suptitle+=f' SNR={args.SNRpp:.0e}'
else: suptitle+=f' Noise={args.noise:.0e} K'
suptitle+='\n'

##---------------------------------------------------------------------------##
# plot samples and likelihoods

s, ll = nf.get_samplesAndLikelihood(args, plot="all")
s[:, 0] *= truths[0]
fig = corner.corner(s, weights=nf.exp(ll), **cornerkwargs)
corner.overplot_lines(fig, truths, color="k", ls="--", lw=1)
plt.suptitle(suptitle)
plt.show()

# savefig should be available via the matplotlib gui
