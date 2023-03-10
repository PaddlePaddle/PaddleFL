export CUDA_VISIBLE_DEVICES=2
dir_name="hrank"
dir_suffix=""
models=("cnn" "vgg" "resnet" "lenet")
datasets=("cifar10" "mnist" "fashionmnist" "cifar100")
dataset=${datasets[0]}
echo ${dataset}
central_train=0
decay=0.99

share_l=5

share_percents=(1 5 10)


prune_interval=30
prune_rate=0.6
auto_rate=1

epochs=500
local_ep=5
local_bs=10
unequal=0

server_mu=0.0
client_mu=0.0
auto_mu=0


dir_name="${dir_name}${local_bs}"
if [ ${unequal} -eq "0" ];then
    dir_name="${dir_name}_equal"
else
    dir_name="${dir_name}_unequal"
fi
dir_name="${dir_name}_l${share_l}"
if [ ${central_train} -eq "0" ];then
    dir_name="${dir_name}_nocenter"
else
    dir_name="${dir_name}_d${decay}"
fi
if [ ${auto_rate} -eq "1" ];then
  dir_name="${dir_name}_auto"
else
  dir_name="${dir_name}_${prune_rate}"
fi
dir_name="${dir_name}${dir_suffix}"
echo ${dir_name}

if [ ! -d "./log/${dataset}" ];then
  mkdir -p ./log/${dataset}
else
  echo "文件夹已经存在"
fi

for model in ${models[*]}; do
echo ${model}
  for share_percent in ${share_percents[*]}; do
    if [[ ${share_percent} == 10 || ${share_percent} == 0 ]]; then
      nohup python ../hrankmain.py --iid 0 --share_percent ${share_percent} --unequal ${unequal} --dataset ${dataset} \
      --result_dir ${dir_name} --epochs ${epochs} --local_bs ${local_bs} --local_ep ${local_ep} --decay ${decay}\
      --prune_interval ${prune_interval} --prune_rate ${prune_rate} --auto_rate ${auto_rate} \
      --server_mu ${server_mu} --client_mu ${client_mu} --auto_mu ${auto_mu} --model ${model} --share_l ${share_l} \
      --central_train ${central_train} > ./log/${dataset}/"${model}_${dir_name}" 2>&1 &
    else
      nohup python ../hrankmain.py --iid 0 --share_percent ${share_percent} --unequal ${unequal} --dataset ${dataset} \
      --result_dir ${dir_name} --epochs ${epochs} --local_bs ${local_bs} --local_ep ${local_ep} --decay ${decay}\
      --prune_interval ${prune_interval} --prune_rate ${prune_rate} --auto_rate ${auto_rate} \
      --server_mu ${server_mu} --client_mu ${client_mu} --auto_mu ${auto_mu} --model ${model} --share_l ${share_l} \
      --central_train ${central_train}  &
    fi
  done
done

#nohup python ../hrankmain.py --iid 1 --share_percent 1 --unequal ${unequal} --result_dir ${dir_name} --epochs ${epochs} \
# --local_bs ${local_bs} --local_ep ${local_ep} --prune_interval ${prune_interval} --prune_rate ${prune_rate} \
# --auto_rate ${auto_rate} --server_mu ${server_mu} --client_mu ${client_mu} --auto_mu ${auto_mu} --model ${model} &