#!/bin/bash

declare -a params=(
    "-e 55 -lr 0.005 -opt adamw -sch step -sg 0.95 -ss 5 -bs 128 -wd 1e-4 --aug True"
    )
for param in "${params[@]}"
do
    python train_cnn.py $param
done
