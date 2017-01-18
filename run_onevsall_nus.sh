#!/bin/bash

num_classes=81
num_threads=20
numTrain=6000
batch_size=1000
max_rho=100
mult_rho=1.1

num_per_thread=$((num_classes / num_threads))
num_per_thread_extra=$((num_classes-$((num_threads*num_per_thread))))
num_reg_threads=$((num_classes-$((num_per_thread_extra*$((num_per_thread+1))))))

# train one-vs-all categories in parallel
for ci in $(seq 1 $num_per_thread $num_reg_threads);do
echo $ci
ci_last=$((ci+num_per_thread-1))
srun --mem=12000  /usr/local/bin/matlab-R2012a -nodesktop -nosplash -r  "try; deep_onevsall_nus_data(0,10,100,0,$numTrain,$num_classes,20,'nus-lite','val','SADMM',$batch_size,$max_rho,$mult_rho,1,0,$ci,$ci_last,0,1,5);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
done
for ci in $(seq $((num_reg_threads+1)) $((num_per_thread+1)) $num_classes);do
echo $ci
ci_last=$((ci+num_per_thread_extra-1))
srun --mem=12000 /usr/local/bin/matlab-R2012a -nodesktop -nosplash -r  "try; deep_onevsall_nus_data(0,10,100,0,$numTrain,$num_classes,20,'nus-lite','val','SADMM',$batch_size,$max_rho,$mult_rho,1,0,$ci,$ci_last,0,1,5);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
done

sleep 3600
#Evaluate the trained model
srun --mem=12000 /usr/local/bin/matlab-R2012a -nodesktop -nosplash -r  "try; deep_onevsall_nus_data(0,10,100,0,6000,81,20,'nus-lite','val','SADMM',1000,100,1.1,1,0,1,81,1,1,5);catch; disp(lasterr); matlabpool close;  exit; end; exit" &

