#!/bin/bash


python runAll.py 2.0 --noise 0.0 --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' --chromatic &
python runAll.py 4.0 --noise 0.0 --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' --chromatic &
python runAll.py 6.0 --noise 0.0 --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' --chromatic 

wait 

python runAll.py 2.0 --noise 0.0 --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' --combineSigma '4' --diffCombineSigma --chromatic &
python runAll.py 2.0 --noise 0.0 --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' --combineSigma '4 6' --diffCombineSigma --chromatic 

wait



noise='0.00001 0.001 0.1'

for n in $noise
do

python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' --combineSigma '4' --diffCombineSigma &
python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' --combineSigma '4 6' --diffCombineSigma 

wait

python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' --combineSigma '4' --diffCombineSigma --chromatic &
python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' --combineSigma '4 6' --diffCombineSigma --chromatic 

wait

done


python runAll.py 2.0 --noise 0.0 --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' &
python runAll.py 4.0 --noise 0.0 --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' &
python runAll.py 6.0 --noise 0.0 --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' 

wait 

python runAll.py 2.0 --noise 0.0 --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' --combineSigma '4' --diffCombineSigma &
python runAll.py 2.0 --noise 0.0 --plot 'all' --noisyT21 --noiseSeed 0 --append '_SVD' --combineSigma '4 6' --diffCombineSigma 

wait

echo 