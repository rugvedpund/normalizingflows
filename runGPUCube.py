# python runCubeGame.py --noise 0.0 --combineSigma '4 6' --freqs '1 51' --fgFITS 'ulsa.fits'

import cubesampler
import numpy as np
import torch
import FlowLikelihood
import parser
import corner
import matplotlib.pyplot as plt

##---------------------------------------------------------------------------##
# argparser block

argparser = parser.create_parser()
args = argparser.parse_args()

# must have --noisyT21 and --diffCombineSigma!!!
args.noisyT21 = True
args.diffCombineSigma = True

if args.appendLik == "":
    args.appendLik = "_gpucube"

parser.prettyprint(args)

##---------------------------------------------------------------------------##

flow = FlowLikelihood.FlowLikelihood(args)

##---------------------------------------------------------------------------##
# main sampler block

cube = cubesampler.Cube(
    flow.likelihood, [(0.8, 1.2), (13, 15), (16, 17)]
)
cube.run()

##---------------------------------------------------------------------------##
# save block

samples = cube.samples
loglikelihood = cube.cubelikes

cornerkwargs = {
    "show_titles": True,
    "levels": [1 - np.exp(-0.5), 1 - np.exp(-2)],
    "bins": 50,
    "range": [(0.8, 1.2), (18, 22), (65, 70)] if cosmicdawn
    # else [(0.8, 1.2), (12, 16), (15.6, 17)],
    else [(0.1, 10), (1, 25), (1, 40)],
    "labels": [r"A", r"$\nu_{\rm rms}$", r"$\nu_{\rm min}$"],
    "truths": [1, 20, 67.5] if cosmicdawn else [1.0, 14.0, 16.4],
    "plot_datapoints": False,
}

fig = corner.corner(samples, weights=nf.exp(loglikelihood), **cornerkwargs)
plt.suptitle('Cube Samples')
cname = nf.get_lname(args, plot="corner")
cname += ".pdf"
print(f"saving corner plot pdf to {cname}")
plt.savefig(cname, dpi=300)

lname = nf.get_lname(args, plot="all")
print(f"saving corner likelihood results to {lname}")
np.savetxt(
    lname,
    np.column_stack([samples, loglikelihood]),
    header="amp,width,numin,loglikelihood",
)
