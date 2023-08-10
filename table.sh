#!/bin/bash
shopt -s expand_aliases
alias optimize="python optimizeTriangle.py --noisyT21 --diffCombineSigma"

noises='0.00001 0.001 0.1'
sigmas='2.0 4.0 6.0'

for s in $sigmas
do
optimize --noise 0.0 --sigma $s
done

for n in $noises
do
optimize --noise $n --sigma 2.0
done

