export CUDA_VISIBLE_DEVICES=2
dir_name="datashare"
dir_suffix="_origin"
models=("cnn" "vgg" "resnet" "lenet")
datasets=("cifar10" "mnist" "fashionmnist" "cifar100")
dataset=${datasets[0]}
echo ${dataset}

share_l=5

share_percents=(1 5 10)

epochs=500
local_ep=5
local_bs=10
unequal=0


dir_name="${dir_name}${local_bs}"
if [ ${unequal} -eq "0" ];then
    dir_name="${dir_name}_equal"
else
    dir_name="${dir_name}_unequal"
fi
dir_name="${dir_name}_l${share_l}"
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
      nohup python ../datasharemain.py --iid 0 --share_percent ${share_percent}  --unequal ${unequal} --dataset ${dataset} \
       --result_dir ${dir_name} --epochs ${epochs} --local_bs ${local_bs} --local_ep ${local_ep} \
       --model ${model} --share_l ${share_l} \
      > ./log/${dataset}/"${model}_${dir_name}" 2>&1 &
    else
      nohup python ../datasharemain.py --iid 0 --share_percent ${share_percent} --unequal ${unequal} --dataset ${dataset} \
      --result_dir ${dir_name} --epochs ${epochs} --local_bs ${local_bs} --local_ep ${local_ep} \
      --model ${model} --share_l ${share_l}&
    fi
  done
done


#nohup python ../datasharemain.py --iid 1 --share_percent 1 --unequal ${unequal} --result_dir ${dir_name} --epochs ${epochs} \
# --local_bs ${local_bs} --local_ep ${local_ep} --model ${model} &