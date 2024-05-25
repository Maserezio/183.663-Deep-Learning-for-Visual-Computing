#!/bin/bash

declare -a params=(
    "-e 55 -lr 0.001 -opt adamw -sch step -sg 0.95 -ss 5 -bs 128 -wd 1e-4 --aug True"
    "-e 30 -lr 0.0005 -opt adam -sch exp -sg 0.9 -ss 1 -bs 64 -wd 1e-3 --aug False"
    "-e 40 -lr 0.01 -opt adam -sch exp -sg 0.85 -ss 3 -bs 256 -wd 1e-5 --aug True"
    "-e 20 -lr 0.005 -opt sgd -sch step -sg 0.95 -ss 2 -bs 128 -wd 1e-4 --aug False"
    "-e 35 -lr 0.002 -opt adamw -sch exp -sg 0.9 -ss 5 -bs 64 -wd 1e-4 --aug True"
    "-e 45 -lr 0.0008 -opt adamw -sch step -sg 0.85 -ss 3 -bs 128 -wd 1e-3 --aug False"
    "-e 50 -lr 0.0055 -opt sgd -sch exp -sg 0.95 -ss 1 -bs 256 -wd 1e-5 --aug True"
    "-e 25 -lr 0.003 -opt sgd -sch step -sg 0.85 -ss 4 -bs 64 -wd 1e-4 --aug False"
    "-e 30 -lr 0.0025 -opt adam -sch step -sg 0.9 -ss 2 -bs 128 -wd 1e-3 --aug True"
    "-e 55 -lr 0.006 -opt adam -sch exp -sg 0.95 -ss 3 -bs 256 -wd 1e-4 --aug False"
    "-e 40 -lr 0.0035 -opt sgd -sch step -sg 0.85 -ss 5 -bs 64 -wd 1e-5 --aug True"
    "-e 20 -lr 0.007 -opt adamw -sch exp -sg 0.9 -ss 1 -bs 128 -wd 1e-4 --aug False"
    "-e 45 -lr 0.0013 -opt adam -sch step -sg 0.95 -ss 4 -bs 256 -wd 1e-3 --aug True"
    "-e 50 -lr 0.006 -opt adam -sch exp -sg 0.85 -ss 2 -bs 128 -wd 1e-5 --aug False"
    "-e 25 -lr 0.0032 -opt sgd -sch step -sg 0.9 -ss 3 -bs 64 -wd 1e-4 --aug True"
    "-e 30 -lr 0.0065 -opt sgd -sch exp -sg 0.95 -ss 1 -bs 256 -wd 1e-3 --aug False"
    "-e 55 -lr 0.005 -opt adamw -sch step -sg 0.9 -ss 5 -bs 128 -wd 1e-5 --aug True"
    "-e 40 -lr 0.0038 -opt adam -sch exp -sg 0.85 -ss 3 -bs 64 -wd 1e-4 --aug False"
    "-e 20 -lr 0.008 -opt adam -sch step -sg 0.95 -ss 2 -bs 256 -wd 1e-5 --aug True"
    "-e 45 -lr 0.0014 -opt sgd -sch exp -sg 0.9 -ss 4 -bs 128 -wd 1e-3 --aug False"
    "-e 50 -lr 0.0058 -opt sgd -sch step -sg 0.85 -ss 1 -bs 64 -wd 1e-4 --aug True"
    "-e 25 -lr 0.0035 -opt adamw -sch exp -sg 0.95 -ss 3 -bs 256 -wd 1e-5 --aug False"
    "-e 30 -lr 0.0068 -opt adamw -sch step -sg 0.9 -ss 5 -bs 128 -wd 1e-4 --aug True"
    "-e 55 -lr 0.0052 -opt sgd -sch exp -sg 0.85 -ss 2 -bs 64 -wd 1e-3 --aug False"
    "-e 40 -lr 0.0042 -opt sgd -sch step -sg 0.95 -ss 3 -bs 256 -wd 1e-5 --aug True"
    "-e 20 -lr 0.009 -opt adamw -sch exp -sg 0.9 -ss 1 -bs 128 -wd 1e-4 --aug False"
    "-e 45 -lr 0.0016 -opt adam -sch step -sg 0.95 -ss 4 -bs 64 -wd 1e-3 --aug True"
    "-e 50 -lr 0.0055 -opt adam -sch exp -sg 0.85 -ss 2 -bs 256 -wd 1e-5 --aug False"
    "-e 25 -lr 0.004 -opt sgd -sch step -sg 0.9 -ss 3 -bs 128 -wd 1e-4 --aug True"
)

for param in "${params[@]}"
do
    python train_cnn.py $param
done