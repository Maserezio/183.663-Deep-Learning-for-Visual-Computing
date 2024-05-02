#!/bin/bash

declare -a params=(
    "-e 30 -lr 0.005 -opt adamw -sch exp -sg 0.8 -ss 5 -bs 64 -wd 1e-5 --aug False"
    "-e 35 -lr 0.05 -opt sgd -sch exp -sg 0.9 -ss 2 -bs 256 -wd 5e-4 --aug True"
    "-e 25 -lr 0.001 -opt adam -sch exp -sg 0.7 -ss 4 -bs 128 -wd 1e-4 --aug True"
    "-e 40 -lr 0.01 -opt adamw -sch exp -sg 0.85 -ss 3 -bs 64 -wd 1e-5 --aug True"
    "-e 30 -lr 0.01 -opt sgd -sch exp -sg 0.75 -ss 6 -bs 256 -wd 5e-4 --aug True"
    "-e 35 -lr 0.005 -opt adamw -sch step -sg 0.95 -ss 5 -bs 128 -wd 1e-4 --aug True")
for param in "${params[@]}"
do
    python train_resnet18.py $param
done
