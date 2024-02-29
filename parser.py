from argparse import ArgumentParser


def create_parser():

    parser = ArgumentParser(description="Normalizing Flow for ULSA/GSM maps")

    parser.add_argument("--fgFITS", type=str, required=False, default="ulsa.fits")
    parser.add_argument("--sigma", type=float, default=2.0, required=False)
    parser.add_argument("--subsample_factor", type=int, default=None, required=False)
    parser.add_argument("--chromatic", action="store_true")
    parser.add_argument("--galcut", type=float, default=20.0, required=False)
    parser.add_argument("--noPCA", action="store_true")
    parser.add_argument(
        "--freqs", type=str, default="1 51", required=False
    )  # e.g. '1 51' separated by space

    parser.add_argument(
        "--combineSigma", type=str, required=False, default=""
    )  # e.g. '4 6' separated by space

    parser.add_argument("--SNRpp", type=float, default=None, required=False)
    parser.add_argument("--noise", type=float, default=0.0, required=False)
    parser.add_argument("--noiseSeed", type=int, default=0, required=False)
    parser.add_argument("--torchSeed", type=int, default=0, required=False)
    parser.add_argument("--subsampleSigma", type=float, default=2.0, required=False)

    parser.add_argument("--noisyT21", action="store_true")
    parser.add_argument(
        "--gainFluctuationLevel", type=float, default=0.0, required=False
    )
    parser.add_argument("--gFdebug", type=int, default=0, required=False)
    parser.add_argument("--append", type=str, default="_SVD")

    parser.add_argument("--DA_factor", type=float, required=False, default=1.0)
    parser.add_argument(
        "--plot", type=str, default="all", required=False
    )  # 1dAmplitude 1dWidth 1dNuMin WvA NvA WvN
    parser.add_argument(
        "--freqFluctuationLevel", type=float, required=False, default=0.0
    )
    parser.add_argument(
        "--nPCA", type=str, default=""
    )  # e.g 'nmin nmax' separated by space

    parser.add_argument(
        "--diffCombineSigma", action="store_true"
    )  # bool, whether to do fg_cS[sigma]-fg_cS[sigma-1]
    parser.add_argument(
        "--avgAdjacentFreqBins", action="store_true"
    )  # bool, whether to average adjacent freq bins

    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--appendLik", type=str, default="", required=False)

    return parser
