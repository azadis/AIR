
Auxiliary Image Regularization for Deep CNNs with Noisy Labels
---------------------------------------------------------------------

Copyright (C) {2016} {Samaneh Azadi, Jiashi Feng, Stefanie Jegelka, Trevor Darrell}

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
 
The research is to be cited in the bibliography as:

S. Azadi, J. Feng, S. Jegelka, T. Darrell, “Auxiliary Image Regularization for Deep CNNs with Noisy Labels”, International Conference on Learning Representations (ICLR), 2016.

---------------------------------------------------------------------

 The code is to run Auxiliary Image Regularizer (AIR) for classification on different datasets
 Paper at: http://arxiv.org/pdf/1511.07069v2.pdf
 Code by: Samaneh Azadi

---------------------------------------------------------------------
												Help:
---------------------------------------------------------------------

 OBJECTIVE = lambda_sen * normalized \sum ||.||_2 + lambda_las * ||.||_1 +...
					 (1/num_samples) * Loss + (mu/2*num_classes) * ||w||_2^2 

rho : the factor used in the Augmented Lagrangain term if use 'SADMM' as the solver and regularization parameter if use 'svm-sgd'

train : number of training samples per class

num_classes : number of categories

dataType : 'imagenet' / 'cifar10' /  'mnist' 

noisy_label : 0/1 want to add noise to the labels?

noise_ratio : with what ratio? e.g. if you want "50% = 1/2" noise, set noise_ratio to "2"

state : 'test'/'val'

solver : 'SADMM'/'svm-sgd'

percent_features : What percent of the feature values to be kept? want to make the feature vector more sparse for the sake of memory/computation cost 

num_parts : divide the training and testing data into 'num_parts' batches to save them in a matrix at the first time of reading the features 

id : 0-9 which random setting to use?

---------------------------------------------------------------------
 						Continue Help : Only Valid for 'SADMM' solver
---------------------------------------------------------------------
epochs : number of epochs to do training

batch_size : number of samples to use per iteration

max_rho: the max value for adaptive rho used in the Augmented Lagrangian term

mult_rho : {1.1, 1.3, 1.5, 1.7} multiplier factor for rho in each step

ci_first, ci_last, par_finished : can run the one-vs-all classification for different categories in parallel to make it finish faster ===>> 

ci_first : the first category in the current parallel run of algorithm 

ci_last : the last category in the current parallel run of algorithm 

par_finished : 0/1 : Is the parallel training of all categories done and want to start testing?
				(If parallel training not finished yet, it continues training on the categories 
				not trained  yet)

(While using "deep_onevsall_nus_data.m", you should first train the one-vs-others classifiers
in parallel cores with "par_finished = 0" and then set "par_finished = 1" to do evaluation.
If want to train only one classifier at each time, you can set "par_finished = 1").


---------------------------------------------------------------------
										Instruction to run the code
---------------------------------------------------------------------
1. Fill all lines with the correct directory path in "SADMM_normalized.m" as well as
"deep_reg_multiclass.m" or "deep_classification_onevsall_nus.m".

The extracted deep feature vectors should be saved as '.h5' files under their own category directory
as 'calDir/train/specific_category/'. We extracted FC_7 features from AlexNet for all images.

calDir : includes extracted features (.h5) in subdirectories of "calDir/train/class_name_i" and "calDir/test/class_name_i";Read both train and validation data from this directory
For CIFAR10 features should be saved in subdirs  "calDir/data_batch_i/class_name_i" and
"test/class_name_i"

caffeDir : includes the original png images; only the path name is needed to produce
a txt file including images as a list

dataDir: the path to save the model weights 

outDir: the path to save the final result

paramDir: the path to save the parameters

For NUS_WIDE_LITE dataset, the images are read based on the txt file in the NUS_LITE_data directory.

2. Start MATLAB and run the VLFeat setup command:
vlfeat-0.9.19/toolbox/vl_setup

3. Run deep_reg_multiclass.m or dep_onevsall_nus_data.m
(the final hyper-parameters found for the results in the paper were run similar to run_AIR.sh file. For NUS_LITE dataset, one can use the parameters used in run_onevsall_nus.sh to get the same results as in the paper.)
	 
	 





    


 






