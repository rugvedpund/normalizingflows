# normalizingflows

This package implements a normalizing flow algorithm for Dark Ages signal separation. All results are published at https://arxiv.org/pdf/2310.06134.pdf

## Usage

Simply ensure the repository is in your `PYTHONPATH`. 
Prerequisites: `SINF`,`luseepy` can be installed from https://github.com/rugvedpund/SINF and https://github.com/lusee-night/luseepy respectively.

By default output is stored to the bash path variable `$NF_WORKDIR` which can be set with:
```bash
export NF_WORKDIR=/home/rugved/path/to/working/directory
``` 

To run the code, you can use the two scripts `runDA.py` and `runCMB.py` for Dark Ages and CMB signal separation respectively as follows:

```bash
python runDA.py --noisyT21 --sigma 2.0 --noise 0.0 --retrain
```
which should train and generate the corner plot and the amplitude plot for the simple case for a 2 deg noiseless map. Further arguments can be passed to the script, explained inside it.
