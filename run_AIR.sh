%Parameters used in the paper to run the multiclass classification solver with AIR


%CIFAR10, 1000 samples, 50% noise 
deep_reg_multiclass(0,10,10,0,1000,10,'cifar10',1,0.5,'test','SADMM',50,3000,100,1.1,0.2,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
%CIFAR10, 1000 samples, 40% noise 
deep_reg_multiclass(0,10,10,0,1000,10,'cifar10',1,0.4,'test','SADMM',60,3000,100,1.1,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
%CIFAR10, 1000 samples, 30% noise 
deep_reg_multiclass(0,10,10,0,1000,10,'cifar10',1,0.3,'test','SADMM',80,3000,100,1.1,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &


%CIFAR10, 2000 samples, 50% noise 
deep_reg_multiclass(0,10,10,0,2000,10,'cifar10',1,0.5,'test','SADMM',50,4000,50,1.3,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
%CIFAR10, 2000 samples, 40% noise 
deep_reg_multiclass(0,10,10,0,2000,10,'cifar10',1,0.4,'test','SADMM',110,8000,100,1.1,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
%CIFAR10, 2000 samples, 30% noise 
deep_reg_multiclass(0,10,10,0,2000,10,'cifar10',1,0.3,'test','SADMM',130,10000,100,1.1,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &


%CIFAR10, 3000 samples, 50% noise 
deep_reg_multiclass(0,10,10,0,3000,10,'cifar10',1,0.5,'test','SADMM',60,10000,100,1.5,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
%CIFAR10, 3000 samples, 40% noise 
deep_reg_multiclass(0,10,10,0,3000,10,'cifar10',1,0.4,'test','SADMM',150,14000,100,1.3,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
%CIFAR10, 3000 samples, 30% noise 
deep_reg_multiclass(0,10,10,0,3000,10,'cifar10',1,0.3,'test','SADMM',130,15000,100,1.3,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &

%CIFAR10, 4000 samples, 50% noise 
deep_reg_multiclass(0,10,10,0,4000,10,'cifar10',1,0.5,'test','SADMM',60,15000,100,1.1,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
%CIFAR10, 4000 samples, 40% noise 
deep_reg_multiclass(0,10,10,0,4000,10,'cifar10',1,0.4,'test','SADMM',200,18000,100,1.1,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
%CIFAR10, 4000 samples, 30% noise 
deep_reg_multiclass(0,10,10,0,4000,10,'cifar10',1,0.3,'test','SADMM',200,20000,100,1.1,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &

%CIFAR10, 5000 samples, 50% noise 
deep_reg_multiclass(0,10,10,0,5000,10,'cifar10',1,0.5,'test','SADMM',60,18000,100,1.1,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
%CIFAR10, 5000 samples, 40% noise 
deep_reg_multiclass(0,10,10,0,5000,10,'cifar10',1,0.4,'test','SADMM',200,23000,100,1.1,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
%CIFAR10, 5000 samples, 30% noise 
deep_reg_multiclass(0,10,10,0,5000,10,'cifar10',1,0.3,'test','SADMM',200,26000,100,1.1,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &

%MNIST, 6000 samples, 50% noise
deep_reg_multiclass(0,10,10,0,6000,10,'mnist',1,0.5,'test','SADMM',150,18000,100,1.3,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &
 

%Imagenet7k, 100 samples, 50% noise
deep_reg_multiclass(0,10,10,0,100,50,'imagenet',1,0.5,'test','SADMM',50,500,1000,1.7,1,0);catch; disp(lasterr); matlabpool close;  exit; end; exit" &




