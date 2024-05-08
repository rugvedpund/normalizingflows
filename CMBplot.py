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
# setup plot params

cosmicdawn = True if args.fgFITS == "gsm16.fits" else False
truth = 2.725
fg = "CD" if cosmicdawn else "DA"
labels = {
    "": "$2^\circ$",
    "4": "$2^\circ+4^\circ$",
    "4 6": "$2^\circ+4^\circ+6^\circ$",
}
suptitle=rf"CMB likelihood \\ {fg} for {labels[args.combineSigma]},"
if args.SNRpp is not None: suptitle+=f' SNR={args.SNRpp:.0e}'
else: suptitle+=f' Noise={args.noise:.0e} K'

##---------------------------------------------------------------------------##
# plot

s, ll = nf.get_samplesAndLikelihood(args, plot="A")
quantiles = corner.core.quantile(s[:, 0], [0.32, 0.5, 0.68, 0.95], weights=nf.exp(ll))
dq = quantiles[2] - quantiles[0]
maxll = s[np.argmax(ll), 0]
for x in quantiles:
    plt.axvline(2.718 * x, c="k", alpha=0.5, lw=0.5)
plt.axvline(2.718,color='k',label='Truth')
plt.plot(2.718 * s, nf.exp(ll))
plt.xscale("log")
plt.text(10, 0.9, rf'Amplitude {2.718*quantiles[1]:.3f} +/-{2.718*dq:.3f} K')
plt.text(10,0.8, rf'$\leq$ {2.718*quantiles[3]:.3f} K 95\% CL')
plt.xlim(0.3,300)
plt.title(
        suptitle
)
plt.show()
