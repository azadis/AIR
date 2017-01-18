	%---------------------------------------------------------------------
function deep_onevsall_nus_data(lambda_las,lambda_sen,rho,mu,train,num_classes,epochs,dataType,state,solver,batch_size,max_rho,mult_rho,percent_features,id,ci_first,ci_last,par_finished,num_parts,top_best)
%---------------------------------------------------------------------

% Run Auxiliary Image Regularizer for one vs. all classification on NUS dataset
% Paper at: http://arxiv.org/pdf/1511.07069v2.pdf
% Code by: Samaneh Azadi

%---------------------------------------------------------------------
%																Help:
%---------------------------------------------------------------------

% OBJECTIVE = lambda_sen * normalized \sum ||.||_2 + lambda_las * ||.||_1 +...
%					(1/num_samples) * Loss + (mu/2*num_classes) * ||w||_2^2 

%rho : the factor used in the Augmented Lagrangain term if use 'SADMM' as the solver and
%		regularization parameter if use 'svm-sgd'
%train : number of training samples per class
%num_classes : number of categories
%dataType : 'nus-lite' 
%state : 'test'/'val'
%solver : 'SADMM'/'svm-sgd'
%percent_features : What percent of the feature values to be kept? 
%		want to make the feature vector more sparse for the sake of memory/computation cost
%id : 0-9 which random setting to use?
%num_parts : divide the training and testing data into 'num_parts' batches to save them in 
%				a matrix at the first time of reading the features 
%top_best : integer number X e.g. 5 : top 5 predictions to be considered while computing 
%			precision and recall  

%---------------------------------------------------------------------
% 						Continue Help : Only Valid for 'SADMM' solver
%---------------------------------------------------------------------
%epochs : number of epochs to do training
%batch_size : number of samples to use per iteration
%max_rho: the max value for adaptive rho used in the Augmented Lagrangian term
%mult_rho : {1.1, 1.3, 1.5, 1.7} multiplier factor for rho in each step
%ci_first, ci_last, par_finished : can run the one-vs-all classification for different
%		 categories in parallel to make it finish faster ===>> 
%ci_first : the first category in the current parallel run of algorithm 
%ci_last : the last category in the current parallel run of algorithm 
%par_finished : 0/1 : Is the parallel training of all categories done and want to start testing?
%				(If parallel training not finished yet, it continues training on the categories 
%				not trained  yet)

if nargin == 12
% when svm is called as a solver
	epochs = 10
	batch_size = 100
	max_rho = 10
	mult_rho = 1.1
	ci_first = 1
	ci_last = num_classes
	par_finished = 1
end

%mainDir : save read data as a matrix in this directory
%caffeDir : includes the original png images; only the path name is needed to produce
% 				a txt file including images as a list
%dataDir: the path to save the model weights 
%outDir: the path to save the final result
%paramDir: save the parameters as a matrix in this directory



% --------------------------------------------------------------------
%                                                           Setup data
% --------------------------------------------------------------------


conf.dataset = dataType;
conf.mainDir = '/n/whiskey/xy/vis/samaneh/NUS-WIDE-LITE/';
conf.dataDir  =  sprintf('nus_lite_data') ;
conf.outDir = 'nus_lite_data/outputHardMulti';
conf.caffeDir = '/y/sazadi/NUS-WIDE/images/NUS_Wide_Lite_deep_fc7';
conf.paramDir  =  '/n/whiskey/xy/vis/samaneh/NUS-WIDE-LITE/data/parameters/try';
conf.prefix = 'baseline' ;
conf.M=fullfile('/u/vis/x1/samaneh/NUS-WIDE-LITE/data/', [conf.prefix '-M-test.mat']) ;
conf.numTest  =  3000;
conf.numTest_val = 3000;
conf.numClasses = num_classes;
conf.numTrain = train ;		
conf.numTrain_val = train;	
conf.state = state; %% state: 'val'/'test' (Tuning hyper parameters or Testing)

conf.svm.C = rho ;
conf.svm.biasMultiplier = 1 ;
conf.randSeed = 1 ;

conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']) ;
conf.grpnorm.solver=solver;%'svm-sgd/SADMM'; 
conf.svm.solver='sgd';
conf.loss='log';%'log/hinge';
noise_ratio=0;
model.w = [] ;
model.b = [] ;


if ~exist(conf.dataDir,'dir')
	mkdir(conf.dataDir)
end

if ~exist(conf.outDir,'dir')
	mkdir(conf.outDir)
end

classes=cell(1,num_classes);
class_step=floor(81/num_classes);
class_vec=[0:class_step:80];
for i=1:num_classes
	classes{i}=num2str(class_vec(i));
end

conf.numClasses=length(classes);
	

images_train = {} ;
imageClass_train = {} ;
images_test={};
imageClass_test={};
if strcmp(conf.state,'test')
	numTrain = conf.numTrain * conf.numClasses;
	numTest = conf.numTest * conf.numClasses;
else
	numTrain = conf.numTrain_val * conf.numClasses;
	numTest = conf.numTest_val * conf.numClasses;
end

im_numm_test = 0;
picked_name_test = '';
im_numm_train = 0;
picked_name_train = '';
lbs = [];

if ~exist(strcat(conf.mainDir,'rng_setting'),'file')
	for id_ = 0:9
		rng('shuffle','v4');
		rng_setting(id_+1) = rng;
	end
	save(strcat(conf.mainDir,'rng_setting'),'rng_setting');
else
	load(strcat(conf.mainDir,'rng_setting'));
end


if strcmp(conf.state,'test')
	file_train  =  '/n/whiskey/xy/vis/samaneh/NUS-WIDE-LITE/NUS-Wide-Lite-test-train-full-tages.txt';
	file_test = '/n/whiskey/xy/vis/samaneh/NUS-WIDE-LITE/NUS-Wide-Lite-test-gtruth.txt';
else
	file_train = '/n/whiskey/xy/vis/samaneh/NUS-WIDE-LITE/NUS-Wide-Lite-train-train-samedist.txt';
	file_test = '/n/whiskey/xy/vis/samaneh/NUS-WIDE-LITE/NUS-Wide-Lite-train-val-gtruth-samedist.txt';
end


fileID = fopen(file_train);
C = textscan(fileID,'%s %s\n');
fclose(fileID);
images_train = C{1};
imageClass_train = C{2};

fileID = fopen(file_test);
C = textscan(fileID,'%s %s\n');
fclose(fileID);
images_test = C{1};
imageClass_test = C{2};

images_train = {images_train{1:min(numTrain,length(images_train))}}';
numTrain = length(images_train);
imageClass_train = {imageClass_train{1:length(images_train)}}';
images_test = {images_test{1:min(numTest,length(images_test))}}';
imageClass_test = {imageClass_test{1:length(images_test)}}';
numTest = length(images_test);

selTrain  =  [1:length(images_train)];
selTest = [1:length(images_test)] ;
numTest = length(images_test);
if ~exist(sprintf('%s/saved_train_%g_%g_%g_%g.mat',conf.mainDir,conf.numClasses,numTrain,...
num_parts,num_parts))

	cols_im = [];
	rows_im = [];
	vals_im = [];
	for i = 1:length(imageClass_train)
		cols_im = [cols_im,str2num(imageClass_train{i})+1];
		rows_im = [rows_im,i*ones(1,length(str2num(imageClass_train{i})))];
		vals_im = [vals_im,ones(1,length(str2num(imageClass_train{i})))];
		imageClass_train{i} = strsplit(imageClass_train{i},',');
	end

	imageClass_train_mat = sparse(rows_im,cols_im,double(vals_im),length(imageClass_train),conf.numClasses);

	feat = {};
	for ii  =  1:length(selTrain)
		fprintf('Processing %s (%.2f %%)\n', images_train{ii}, 100 * ii / length(images_train)) ;
		feat{ii} = h5read(fullfile(images_train{selTrain(ii)}),'/feature') ;
		feat{ii} = reshape(feat{ii},1,length(feat{ii}));  
	end

	images_train = feat;
	images_train = cat(1,images_train{:})';

	%%--------------------------------------
	%%  		Normalize train features
	%%--------------------------------------

	for i = 1:size(images_train,2)
		l2_norm = norm(images_train(:,i));
		images_train(:,i) = images_train(:,i)./l2_norm;
	
	end
	%--------------------------------------
	len_part = floor(numTrain/num_parts);
	for btch = 1:num_parts
		images_train_ = images_train(:,((btch-1)*len_part+1):len_part*btch);
		imageClass_train_ = imageClass_train(((btch-1)*len_part+1):len_part*btch);
		save(sprintf('%s/saved_train_%g_%g_%g_%g.mat',conf.mainDir,conf.numClasses,numTrain,...
		btch,num_parts),'images_train_','imageClass_train_');
	end
else
	len_part = floor(numTrain/num_parts);
	images_train = zeros(4096,numTrain);
	imageClass_train = cell(1,numTrain);
	for btch = 1:num_parts
		load(sprintf('%s/saved_train_%g_%g_%g_%g.mat',conf.mainDir,conf.numClasses,numTrain,btch,num_parts));
		images_train(:,((btch-1)*len_part+1):len_part*btch) = images_train_;
		imageClass_train(((btch-1)*len_part+1):len_part*btch) = imageClass_train_;
	end
end


if ~exist(sprintf('%s/saved_test_%g_%g_%g_%g.mat',conf.mainDir,conf.numClasses,numTest,...
	max(1,num_parts/2),max(1,num_parts/2)))
	rows_im = [];
	cols_im = [];
	vals_im = [];
	for i = 1:length(imageClass_test)
		cols_im = [cols_im,str2num(imageClass_test{i})+1];
		rows_im = [rows_im,i*ones(1,length(str2num(imageClass_test{i})))];
		vals_im = [vals_im,ones(1,length(str2num(imageClass_test{i})))];
		imageClass_test{i} = strsplit(imageClass_test{i},',');
	end
	imageClass_test_mat = sparse(rows_im,cols_im,double(vals_im),length(imageClass_test),conf.numClasses);

	%%read the test images:

	feat = {};
	for ii = 1:length(selTest)
		fprintf('Processing %s (%.2f %%)\n', images_test{ii}, 100 * ii / length(images_test)) ;
		feat{ii} = h5read(fullfile(images_test{selTest(ii)}),'/feature') ;
		feat{ii} = reshape(feat{ii},1,length(feat{ii}));  
	end
	images_test = feat;
	images_test = cat(1,images_test{:})';


	%%--------------------------------------
	%%  		Normalize test features
	%%--------------------------------------


	for i = 1:size(images_test,2)
		l2_norm = norm(images_test(:,i));
		images_test(:,i) = images_test(:,i)/l2_norm;
	end

	len_part = floor(numTest/max(1,(num_parts/2)));
	for btch = 1:(max(1,num_parts/2))
		images_test_ = images_test(:,((btch-1)*len_part+1):(len_part*btch));
		imageClass_test_mat_ = imageClass_test_mat(((btch-1)*len_part+1):len_part*btch,:);

	save(sprintf('%s/saved_test_%g_%g_%g_%g.mat',conf.mainDir,conf.numClasses,numTest,btch,...
	max(1,num_parts/2)),'images_test_','imageClass_test_mat_');

	end
else
	len_part = floor(numTest/(max(1,num_parts/2)));
	images_test = zeros(4096,numTest);
	imageClass_test_mat = zeros(numTest,conf.numClasses);
	for btch = 1:max(1,num_parts/2)
		load(sprintf('%s/saved_test_%g_%g_%g_%g.mat',conf.mainDir,conf.numClasses,numTest,...
		btch,max(1,num_parts/2)));
		images_test(:,((btch-1)*len_part+1):len_part*btch) = images_test_;
		imageClass_test_mat(((btch-1)*len_part+1):len_part*btch,:) = imageClass_test_mat_;
	end

end

% --------------------------------------------------------------------
%                                        Start Training the classifier
% --------------------------------------------------------------------
        
switch conf.grpnorm.solver

	case {'svm-sgd', 'svm-sdca'}
        lambda = 1 / (conf.svm.C *  length(selTrain)) 
        w = [] ;
        
        for ci = 1:length(classes)
            perm = randperm(length(selTrain)) ;
            fprintf('Training model for class %s\n', classes{ci}) ;
            y = zeros(length(selTrain),1)';
            for ind = 1:length(selTrain)
            	im_ind = selTrain(ind);
            	y(im_ind) = 2*ismember(classes{ci},imageClass_train{im_ind})-1 ;
            end
            num_ci = sum(y==1);
            sprintf('number of images with label %s is %g',classes{ci},num_ci)
				lambda
                [w(:,ci) b(ci) svminfo] = vl_svmtrain(images_train(:, selTrain(perm)), y(perm), lambda, ...
                'Solver', conf.svm.solver, ...
                'MaxNumIterations', 500/lambda, ...
                'BiasMultiplier', conf.svm.biasMultiplier, ...
                'Epsilon', 1e-3);
		        svminfo
		        model.svminfo = svminfo;
          
        end
        model.b = conf.svm.biasMultiplier * b ;
        b_ = model.b;
        model.w = w ;
        save(conf.modelPath, 'model') ;

        if strcmp(conf.state,'test')
        	save(sprintf('%s/output_param_svm_%g_%g_%g_%g_%s_%g.mat',conf.paramDir,rho,numTrain,...
        	noise_ratio,conf.numClasses,conf.loss,id),'w','b_');
        end

        
    case {'SADMM'}
		% --------------------------------------------------------------------
		%                                           		Compute Matrix M
		% --------------------------------------------------------------------
		nonzeros_per_sent_per_img = cell(length(selTrain),1);
		starting_im_per_total = zeros(length(selTrain),1);
		row_num = 0;
		cols_M = [];rows_M = [];vals_M = [];
		ind_per_im = 0;
		ind_prev = 0;
		
		sorted_M = sort(abs(images_train(:)),'descend');
		
		thr = sorted_M(floor(length(selTrain)*(percent_features*size(images_train,1))))
		conf.thr = thr;
		images_train_ = images_train.*(abs(images_train)>thr);
		
		nz_per_img = sum(images_train_>0);
		
		
		for ii  =  1:length(selTrain)
			ind_prev = ind_prev+ind_per_im;
			ind_per_im = 0;
			row_start = row_num+1;
			starting_im_per_total(ii) = row_start;
			ind_per_im = ind_per_im+1;
			
			non_zeros = find(images_train_(:,ii)~=0)';
			cols_M = [cols_M,non_zeros];
			rows_M = [rows_M,row_num+[1:nnz(images_train_(:,ii))]];
			vals_M = [vals_M;images_train_(non_zeros,ii)]; %%the last zero is used for bias term
			row_num = row_num+nnz(images_train_(:,ii));
			
			nonzeros_per_sent_per_img{ii}=nnz(images_train_(:,ii));
		end
		
		M = sparse(rows_M',cols_M,double(vals_M),length(rows_M),size(images_train_,1));
		clear cols_M;
		clear rows_M;
		clear vals_M;
		removing_ind = find(sum(M,1)==0);
		M(:,removing_ind) = [];
		
		images_train(removing_ind,:) = [];
		images_train_(removing_ind,:) = [];
		
		%%removing the unneeded data from the test images
		
		images_test(removing_ind,:) = [];
		
		clear feat
	    save(conf.M,'M','nonzeros_per_sent_per_img','starting_im_per_total','images_test','images_train');
			   
	
		w = [] ;
		rp = [1:length(selTrain)];
		Idx = [];
		for d = rp(1:end)
			Idx = [Idx,starting_im_per_total(d):(nonzeros_per_sent_per_img{d}+starting_im_per_total(d)-1)];
		end
		M_before_rows = size(M,1);
		M = M(Idx,:);
		M(size(M,1):M_before_rows,:) = 0;


	W_iterations_ = [];
	OBJECTIVE = cell(length(classes),1);
	REGULARIZATION = cell(length(classes),1);
	training_time = 0;

	class_tot = length(classes);
	for ci = ci_first:ci_last
		y = zeros(1,length(selTrain));
		for ind = 1:length(selTrain)
			im_ind = selTrain(ind);
			y(im_ind) = 2 * ismember(classes{ci},imageClass_train{im_ind}) - 1 ;
		end

		num_ci = sum(y==1);


		if (~ exist(sprintf('%s/output_par_optADMM_%g_%g_%g_%g_%g_%g_%g_%g_%s_%g_%g_%g_%g_%g.mat',...
		conf.paramDir,lambda_las,lambda_sen,rho,mu,numTrain,noise_ratio,ci,class_tot,conf.loss,...
		batch_size,max_rho,mult_rho,1,id),'file')) 
			[w,info] = opt_SADMM(conf.dataset,images_train,images_train_, ...
			y,M,nz_per_img,lambda_las,lambda_sen,rho,mu,rp,ci,class_tot,'hard',conf.loss,numTrain,...
			epochs,batch_size,max_rho,mult_rho,noise_ratio,conf.state,1,id);

		else
			load(sprintf('%s/output_par_optADMM_%g_%g_%g_%g_%g_%g_%g_%g_%s_%g_%g_%g_%g_%g.mat',...
			conf.paramDir,lambda_las,lambda_sen,rho,mu,numTrain,noise_ratio,ci,class_tot,...
			conf.loss,batch_size,max_rho,mult_rho,1,id));
			w = W_iterations;
		end
	
	

		if ci>ci_first
			if size(w,2)<size(W_iterations_,2)
				w1 = zeros(size(W_iterations_,1),size(W_iterations_,2));
				diff_size = size(W_iterations_,2)-size(w,2);
				w1(:,1:size(w,2)) = w;
				w1(:,size(w,2):size(W_iterations_,2)) = repmat(w(:,end),1,diff_size+1);
				w = w1;
				info.elapsedTime = [info.elapsedTime;(info.elapsedTime(end)+10)*ones(diff_size,1)];
			
			
			elseif size(w,2) > size(W_iterations_,2)
				W1_iterations = zeros(size(w,1),size(w,2),ci);
				for cc = 1:(ci-1)
					diff_size = size(w,2)-size(W_iterations_,2);
					W1_iterations(:,1:size(W_iterations_,2),cc) = W_iterations_(:,1:size(W_iterations_,2),cc);
					W1_iterations(:,size(W_iterations_,2):size(w,2),cc) = repmat(W_iterations_(:,end,cc),1,diff_size+1);
				
				end
				W_iterations_ = W1_iterations;
				training_time = [training_time;(info.elapsedTime(end)+10)*ones(diff_size,1)];
			end
		end
		W_iterations_(:,:,ci) = w;
		OBJECTIVE{ci} = info.objective;
		training_time = training_time+info.elapsedTime;
	
	end
end

% --------------------------------------------------------------------
%                                                Test and evaluate
% --------------------------------------------------------------------

switch conf.grpnorm.solver
	case{'SADMM'}
	if (par_finished==1)
		w = W_iterations_;
		save(sprintf('%s/weight-%s-%s-%s-%g-%g-%g-%g-%g-%g-%s-%d-%g.mat',conf.outDir,...
		conf.dataset,conf.grpnorm.solver,conf.loss,lambda_las,lambda_sen,rho,mu,conf.numClasses,...
		numTrain,conf.state,noise_ratio,id),'w')
		info.elapsedTime = training_time;
		iterations = [0:10,11:5:1000,1001:50:100000];

		info.elapsedIter = iterations(1:size(W_iterations_,2));

		model.b = zeros(1,length(classes));
		model.w = w ;



		% Estimate the class of the test images
		disp 'Start to test'

		acc = [];

		prob = zeros(conf.numClasses,length(selTest),size(model.w,2));
		prediction = zeros(length(selTest),size(model.w,2));
		num_testing = sum(sum(imageClass_test_mat(:,str2num(char(classes))+1),2)>0);
		precision_ = zeros(1,size(model.w,2));
		recall_ = zeros(1,size(model.w,2));
		precision_avg  =  zeros(1,size(model.w,2));
		recall_avg = zeros(1,size(model.w,2));

		for j = 1:size(model.w,2)
			w1 = zeros(size(model.w,1),size(model.w,3));
			w1(:,:) = model.w(:,j,:);
			prob(:,:,j) = computeProb(w1,images_test,selTest);
		
			%	-------------------------------------------------
			%	precision-recall
			%	-------------------------------------------------
			pr_rc = prec_recall(imageClass_test_mat,prob(:,:,j),length(selTest),top_best,conf.numClasses);
			
			precision_(j)  =  pr_rc{1};
			recall_(j) = pr_rc{2};
			precision_avg(j) = pr_rc{3};
			recall_avg(j) = pr_rc{4};
		
		
			%  -------------------------------------------------
			%	Accuracy
			%  --------------------------------------------------
			[pm,prediction(:,j)] = max(prob(:,:,j));
			rows_prediction = [1:size(prediction,1)];
			vals_prediction = ones(1,size(prediction,1));
		
			prediction_mat = sparse(rows_prediction,prediction(:,j)',vals_prediction,size(prediction,1),conf.numClasses);
			acc(j) = sum(sum(prediction_mat.*imageClass_test_mat,2))/num_testing;	
		end
		prob_last = prob(:,:,end);
		acc
		elapsedIter = info.elapsedIter;


		save(sprintf('%s/output-%s-%s-%g-%g-%g-%g-%g-%g-%s-%d-%s-%g-%g-%g-%g-%g-%g.mat',...
		conf.outDir,conf.dataset,conf.grpnorm.solver,lambda_las,lambda_sen,rho,mu,conf.numClasses,...
		numTrain,conf.state,noise_ratio,conf.loss,batch_size,max_rho,mult_rho,...
		percent_features,top_best,id),'elapsedIter','OBJECTIVE','acc','training_time','thr',...
		'prob_last','imageClass_test_mat','precision_','recall_','precision_avg','recall_avg')

end
	
case{'svm-sgd', 'svm-sdca','svm-liblinear'}
	
	scores = model.w' * images_test + model.b' * ones(1,size(images_test,2)) ;
	%	-------------------------------------------------
	%	precision-recall
	%	-------------------------------------------------
	pr_rc = prec_recall(imageClass_test_mat,scores,length(selTest),top_best,conf.numClasses);	
	precision_  =  pr_rc{1};
	recall_ = pr_rc{2};
	precision_avg = pr_rc{3};
	recall_avg = pr_rc{4};
			
	%  -------------------------------------------------
	%	Accuracy
	%  --------------------------------------------------
		
	[drop, imageEstClass] = max(scores, [], 1) ;
	acc_im = [];
	num_testing = 0;	
	for im = 1:length(selTest)
		if (sum(ismember(imageClass_test{im},classes))>0)
			num_testing = num_testing+1;
			acc_im = [acc_im,ismember(classes{imageEstClass(im)},imageClass_test{im})];
		end
	end
	acc = sum(acc_im)/num_testing
	svminfo = model.svminfo;

	save(sprintf('%s/output-%s-%s-%g-%g-%g-%s-%d-%s-%g-%g.mat',conf.outDir,conf.dataset,...
	conf.grpnorm.solver,conf.svm.C,conf.numClasses,numTrain,conf.state,...
	noise_ratio,conf.loss,top_best,id),'acc','svminfo','scores','precision_','recall_',...
	'precision_avg','recall_avg')

end


% ---------------------------------------------------------------------
function y = prec_recall(imageClass_test_mat,scores,num_test,top_best,numClasses)
% ---------------------------------------------------------------------
rows_best = [];
cols_best = [];
vals_best = ones(num_test*top_best,1);


for im = 1:num_test
	[sorted,ind_sorted] = sort(scores(:,im),'descend');
	cols_best = [cols_best,ind_sorted(1:top_best)];
	rows_best = [rows_best,im*ones(top_best,1)];
end

best_predictions = sparse(rows_best,cols_best,vals_best,num_test,numClasses);

precision  =  sum(sum(best_predictions.*imageClass_test_mat,2))/nnz(best_predictions);
recall = sum(sum(best_predictions.*imageClass_test_mat,2))/nnz(imageClass_test_mat);
precision_per_im = 0;
recall_per_im = 0;
for im = 1:num_test
	precision_per_im = precision_per_im+sum(best_predictions(im,:).*imageClass_test_mat(im,:))/nnz(best_predictions(im,:));
	recall_per_im = recall_per_im+sum(best_predictions(im,:).*imageClass_test_mat(im,:))/nnz(imageClass_test_mat(im,:));
end	
precision_avg = precision_per_im/num_test;
recall_avg = recall_per_im/num_test;
y=cell(1,4)
y{1} = precision;
y{2} = recall;
y{3} = precision_avg;
y{4} = recall_avg;
		


% ---------------------------------------------------------------------
function y = computeProb(w,samples,samples_part)
% ---------------------------------------------------------------------
wTs = w'*samples(:,samples_part);
wTs  =  bsxfun(@minus, wTs, max(wTs, [], 1)); %Prevent overflow: subtract some large constant from each wTs before the exponential.
prob = bsxfun(@rdivide, exp(wTs), sum(exp(wTs)));
y = prob;
