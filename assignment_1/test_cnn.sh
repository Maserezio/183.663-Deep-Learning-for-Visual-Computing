#!/bin/bash
declare -a models=(
    "CNN__l_rate_0.00012157665459056936_optim_Adam_scheduler_ExponentialLR_num_epochs_20_batch_size_128.pth"
    "CNN__l_rate_0.0001647129056757193_optim_Adam_scheduler_ExponentialLR_num_epochs_20_batch_size_64.pth"
    "CNN__l_rate_0.00027738957312183364_optim_AdamW_scheduler_ExponentialLR_num_epochs_25_batch_size_256.pth"
    "CNN__l_rate_0.00033286748774620056_optim_AdamW_scheduler_StepLR_num_epochs_25_batch_size_256.pth"
    "CNN__l_rate_0.0005987369392383785_optim_AdamW_scheduler_StepLR_num_epochs_55_batch_size_128.pth"
    "CNN__l_rate_0.0007394401199593976_optim_Adam_scheduler_StepLR_num_epochs_45_batch_size_256.pth"
    "CNN__l_rate_0.0011220198089843746_optim_SGD_scheduler_StepLR_num_epochs_40_batch_size_64.pth"
    "CNN__l_rate_0.001131448546875_optim_SGD_scheduler_StepLR_num_epochs_25_batch_size_64.pth"
    "CNN__l_rate_0.0012515777524966226_optim_SGD_scheduler_ExponentialLR_num_epochs_35_batch_size_256.pth"
    "CNN__l_rate_0.001_optim_Adam_scheduler_ExponentialLR_num_epochs_5_batch_size_128.pth"
    "CNN__l_rate_0.007350918906249998_optim_SGD_scheduler_StepLR_num_epochs_25_batch_size_256.pth"
)

for model in "${models[@]}"
do
    python test.py -m cnn -p "saved_models/best_models/$model"
done