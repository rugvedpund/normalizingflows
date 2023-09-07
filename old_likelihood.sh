#!/bin/bash

python run2DLikelihood.py 2.0 --galcut 20 --noise 0.0  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
python run2DLikelihood.py 2.0 --galcut 20 --noise 0.0  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &
wait


python run2DLikelihood.py 2.0 --galcut 20 --noise 0.0  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
python run2DLikelihood.py 2.0 --galcut 20 --noise 0.0  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &
wait


python run2DLikelihood.py 2.0 --galcut 20 --noise 0.0001  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
python run2DLikelihood.py 2.0 --galcut 20 --noise 0.0001  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &


python run2DLikelihood.py 2.0 --galcut 20 --noise 0.0001  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
python run2DLikelihood.py 2.0 --galcut 20 --noise 0.0001  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &
wait


python run2DLikelihood.py 2.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
python run2DLikelihood.py 2.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &


python run2DLikelihood.py 2.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
python run2DLikelihood.py 2.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &
wait


python run2DLikelihood.py 2.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
python run2DLikelihood.py 2.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &
wait


python run2DLikelihood.py 2.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
python run2DLikelihood.py 2.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &
wait







# #noise 0.0
# python runLikelihood.py 1.0 --galcut 20 --noise 0.0 --noiseSeed 0 --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 1.0 --galcut 20 --noise 0.0  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 1.0 --galcut 20 --noise 0.0  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &
# wait

# python runLikelihood.py 1.0 --galcut 20 --noise 0.0 --noiseSeed 0 --chromatic --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 1.0 --galcut 20 --noise 0.0  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 1.0 --galcut 20 --noise 0.0  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &
# wait

# wait
# python runLikelihood.py 8.0 --galcut 20 --noise 0.0 --noiseSeed 0 --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 8.0 --galcut 20 --noise 0.0  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 8.0 --galcut 20 --noise 0.0  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 8.0 --galcut 20 --noise 0.0 --noiseSeed 0 --chromatic --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 8.0 --galcut 20 --noise 0.0  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 8.0 --galcut 20 --noise 0.0  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 4.0 --galcut 20 --noise 0.0 --noiseSeed 0 --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 4.0 --galcut 20 --noise 0.0  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 4.0 --galcut 20 --noise 0.0  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 4.0 --galcut 20 --noise 0.0 --noiseSeed 0 --chromatic --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 --&
# python run2DLikelihood.py 4.0 --galcut 20 --noise 0.0  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 --&
# python run2DLikelihood.py 4.0 --galcut 20 --noise 0.0  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 --&

# #noise 0.00001
# python runLikelihood.py 1.0 --galcut 20 --noise 0.00001 --noiseSeed 0 --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 1.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 1.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 1.0 --galcut 20 --noise 0.00001 --noiseSeed 0 --chromatic --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 1.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 1.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 8.0 --galcut 20 --noise 0.00001 --noiseSeed 0 --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 8.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 8.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 8.0 --galcut 20 --noise 0.00001 --noiseSeed 0 --chromatic --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 8.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 8.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 4.0 --galcut 20 --noise 0.00001 --noiseSeed 0 --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 4.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 4.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 4.0 --galcut 20 --noise 0.00001 --noiseSeed 0 --chromatic --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 4.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 4.0 --galcut 20 --noise 0.00001  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &


# #noise 0.000001
# wait
# python runLikelihood.py 1.0 --galcut 20 --noise 0.000001 --noiseSeed 0 --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 1.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 1.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 1.0 --galcut 20 --noise 0.000001 --noiseSeed 0 --chromatic --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 1.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 1.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 8.0 --galcut 20 --noise 0.000001 --noiseSeed 0 --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 8.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 8.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 8.0 --galcut 20 --noise 0.000001 --noiseSeed 0 --chromatic --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 8.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 8.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 4.0 --galcut 20 --noise 0.000001 --noiseSeed 0 --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 4.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 4.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

# wait
# python runLikelihood.py 4.0 --galcut 20 --noise 0.000001 --noiseSeed 0 --chromatic --noisyT21 --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 4.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --chromatic --noisyT21 --vs "Width" --DA_factor 1.0 --subsampleSigma 2.0 &
# python run2DLikelihood.py 4.0 --galcut 20 --noise 0.000001  --noiseSeed 0 --chromatic --noisyT21 --vs "NuMin" --DA_factor 1.0 --subsampleSigma 2.0 &

