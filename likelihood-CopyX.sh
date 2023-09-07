#!/bin/bash

noiseSeed='0 1 2'
appendId='0 1 2 3 4 5 6 7 8 9'

for ns in $noiseSeed
do

for app in $appendId
do

python runAll.py 2.0 --noise 0.0 --plot 'all' --noisyT21 --noiseSeed $ns --append "_SVD_${app}" --chromatic --retrain
# python runAll.py 2.0 --noise 0.1 --plot 'all' --noisyT21 --noiseSeed $ns --append "_SVD_${app}" --retrain

wait

done
done

echo 


# appendId='0 1 2 3 4'

# for a in $appendId
# do

# python runAll.py 2.0 --noise 0.1 --plot 'all' --noisyT21 --noiseSeed 0 --append "_${a}" --retrain &
# python runAll.py 2.0 --noise 0.1 --plot 'all' --noisyT21 --noiseSeed 1 --append "_${a}" --retrain &
# python runAll.py 2.0 --noise 0.1 --plot 'all' --noisyT21 --noiseSeed 2 --append "_${a}" --retrain 
# wait

# done

# echo 





# noise='0.0 0.00001 0.001 0.1'

# # for n in $noise
# # do

# # python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 0 --combineSigma '4 6' --diffCombineSigma --append '_diffCombineSigmav2True' &
# # python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 1 --combineSigma '4 6' --diffCombineSigma --append '_diffCombineSigmav2True'

# # wait

# python runAll.py 2.0 --noise 0.001 --plot 'all' --noisyT21 --noiseSeed 2 --combineSigma '4 6' --diffCombineSigma --append '_diffCombineSigmav2True' --retrain
# wait
# python runAll.py 2.0 --noise 0.001 --plot 'all' --noisyT21 --noiseSeed 3 --combineSigma '4 6' --diffCombineSigma --append '_diffCombineSigmav2True' --retrain

# # wait

# # python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 0 --combineSigma '4' --diffCombineSigma --append '_diffCombineSigmav2True' &
# # python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 1 --combineSigma '4' --diffCombineSigma --append '_diffCombineSigmav2True'

# # wait

# # python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 2 --combineSigma '4' --diffCombineSigma --append '_diffCombineSigmav2True' &
# # python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 3 --combineSigma '4' --diffCombineSigma --append '_diffCombineSigmav2True'

# # wait

# # done

# echo 
