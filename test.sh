#!/bin/bash

python flowtest.py 1;
python liketest.py 1&
python flowtest.py 2
python liketest.py 2&
python flowtest.py 3
python liketest.py 3;

wait
