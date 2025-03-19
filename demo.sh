#!/usr/bin/env 

############### Host ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"alpha")
    PYTHON="/mnt/data/anaconda3/envs/pytorch_2_0/bin/python" # python environment path
    TENSORBOARD='/home/elliot/anaconda3/envs/pytorch041/bin/tensorboard' # tensorboard environment path
    data_path='/mnt/data/zms/ICCV25/Train_models/data'
    ;;
esac

DATE=`date +%Y-%m-%d`


############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=cgc_resnet56
dataset=cifar10
epochs=500
train_batch_size=128
test_batch_size=128
optimizer=SGD

label_info=demo

save_path=./save/${dataset}_${model}_${epochs}_${optimizer}_${label_info}
tb_path=${save_path}/tb_log  #tensorboard log path

PYTHON="/mnt/data/anaconda3/envs/pytorch_2_0/bin/python"
data_path='/mnt/data/zms/ICCV25/Train_models/data'

echo $PYTHON

############### Neural network ############################
{
$PYTHON cg_train.py --dataset ${dataset} --data_path ${data_path}   \
    --arch ${model} --save_path ${save_path} \
    --epochs ${epochs} --learning_rate 0.01 \
    --optimizer ${optimizer} \
	--schedule 80 120 140 --gammas 0.1 0.1 0.2 \
    --train_batch_size ${train_batch_size} \
    --test_batch_size ${test_batch_size} \
    --workers 4 --ngpu 1 --gpu_id 5 \
    --print_freq 100 --decay 0.0005 --momentum 0.9 \
    --cgc 
}&
wait