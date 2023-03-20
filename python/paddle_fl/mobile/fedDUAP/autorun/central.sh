export CUDA_VISIBLE_DEVICES=0
dir_name="central"
dir_suffix="_decay999"
models=("cnn" "vgg" "lenet")
datasets=("cifar10" "mnist" "fashionmnist" "cifar100")
dataset=${datasets[0]}
echo ${dataset}
decay=0.999

share_l=5

share_percents=(10)

prune_interval=-1
prune_rate=0.6
auto_rate=0

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
dir_name="${dir_name}_d${decay}"
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
      nohup python ../centralmain.py --iid 0 --share_percent ${share_percent} --unequal ${unequal} --dataset ${dataset} \
      --result_dir ${dir_name} --epochs ${epochs} --local_bs ${local_bs} --local_ep ${local_ep} --decay ${decay}\
      --prune_interval ${prune_interval} --prune_rate ${prune_rate} --auto_rate ${auto_rate} \
      --server_mu ${server_mu} --client_mu ${client_mu} --auto_mu ${auto_mu} --model ${model} --share_l ${share_l} \
        > ./log/${dataset}/"${model}_${dir_name}" 2>&1 &
    else
      nohup python ../centralmain.py --iid 0 --share_percent ${share_percent} --unequal ${unequal} --dataset ${dataset} \
       --result_dir ${dir_name} --epochs ${epochs} --local_bs ${local_bs} --local_ep ${local_ep} --decay ${decay}\
       --prune_interval ${prune_interval} --prune_rate ${prune_rate} --auto_rate ${auto_rate} \
       --server_mu ${server_mu} --client_mu ${client_mu} --auto_mu ${auto_mu} --model ${model} --share_l ${share_l} &
    fi
  done
done

#nohup python ../centralmain.py --iid 1 --share_percent 1 --unequal ${unequal} --result_dir ${dir_name} --epochs ${epochs} \
# --local_bs ${local_bs} --local_ep ${local_ep} --prune_interval ${prune_interval} --prune_rate ${prune_rate} \
# --auto_rate ${auto_rate} --server_mu ${server_mu} --client_mu ${client_mu} --auto_mu ${auto_mu} --model ${model} &