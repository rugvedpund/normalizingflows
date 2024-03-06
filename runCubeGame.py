# Example:
# python runCubeGame.py --noise 0.0 --combineSigma '4 6' --freqs '1 51' --fgFITS 'ulsa.fits'

import cubegamesampler
import numpy as np
import torch
import fitsio
import NormalizingFlow as nf
import lusee
import parser
import os
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
    args.appendLik = "_cubegame"

parser.prettyprint(args)

##---------------------------------------------------------------------------##

# set seed
print(f"setting noise seed {args.noiseSeed} and torch seed {args.torchSeed}")
np.random.seed(args.noiseSeed)
torch.manual_seed(args.torchSeed)
torch.cuda.manual_seed_all(args.torchSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")
torch.set_default_tensor_type("torch.cuda.FloatTensor")


fname = nf.get_fname(args)

print(f"loading flow from {fname}")
flow = nf.FlowAnalyzerV2(nocuda=False, loadPath=fname)
flow.set_fg(args)

if args.fgFITS == "ulsa.fits":
    print("using DA model")
    t21 = lusee.MonoSkyModels.T_DarkAges_Scaled(
        flow.freqs, nu_rms=14, nu_min=16.4, A=0.04
    )
    cosmicdawn = False
elif args.fgFITS == "gsm16.fits":
    print("using CD model")
    t21 = lusee.MonoSkyModels.T_CosmicDawn_Scaled(
        flow.freqs, nu_rms=20, nu_min=67.5, A=0.130
    )
    cosmicdawn = True
flow.set_t21(t21)
if args.retrain:
    flow.train(
        flow.train_data, flow.validate_data, nocuda=False, savePath=fname, retrain=True
    )

##---------------------------------------------------------------------------##
# main sampler block

cg = cubegamesampler.CubeGame(
    flow.get_likelihoodFromSamplesGAME, [(0.01, 10), (1, 30), (1, 30)]
)
cg.run()

##---------------------------------------------------------------------------##
# save block

samples = np.vstack([[sa.pars for sa in ga.sample_list] for ga in cg.Games])
ww = np.hstack([[sa.we for sa in ga.sample_list] for ga in cg.Games])
wsumsa = ww / ww.sum()


cornerkwargs = {
    "show_titles": True,
    "levels": [1 - np.exp(-0.5), 1 - np.exp(-2)],
    "bins": 50,
    "range": [(0.8, 1.2), (18, 22), (65, 70)] if cosmicdawn
    # else [(0.8, 1.2), (12, 16), (15.6, 17)],
    else [(0.1, 10), (1, 25), (1, 40)],
    "labels": [r"A", r"$\nu_{\rm rms}$", r"$\nu_{\rm min}$"],
    "truths": [1, 20, 67.5] if cosmicdawn else [1.0, 14.0, 16.4],
    "plot_datapoints": True,
}


def plotel(G):
    global fig
    cov = G.cov
    print(G.cov)
    val, vec = np.linalg.eig(cov)
    vec = vec.T

    vec[0] *= np.sqrt(np.real(val[0]))
    vec[1] *= np.sqrt(np.real(val[1]))
    print(vec[0], "A")
    print(vec[1], "B")
    corner.overplot_points(fig, G.mean[None], c="b", marker="o", ms=1.0)
    corner.overplot_points(
        fig, [G.mean - vec[0], G.mean + vec[0]], c="r", marker="", linestyle="-", lw=0.5
    )
    corner.overplot_points(
        fig, [G.mean - vec[1], G.mean + vec[1]], c="r", marker="", linestyle="-", lw=0.5
    )
    corner.overplot_points(
        fig, [G.mean - vec[2], G.mean + vec[2]], c="r", marker="", linestyle="-", lw=0.5
    )

fig = corner.corner(samples, weights=wsumsa, **cornerkwargs)
for ga in cg.Games:
    for G in ga.Gausses:
        plotel(G)
plt.suptitle("Cube Game Samples")
cname = nf.get_lname(args, plot="corner")
cname += ".pdf"
plt.savefig(cname, dpi=300)

lname = nf.get_lname(args, plot="all")
print(f"saving corner likelihood results to {lname}")
np.savetxt(
    lname,
    np.column_stack([samples, np.log(ww)]),
    header="amp,width,numin,loglikelihood",
)
