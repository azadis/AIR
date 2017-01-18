%---------------------------------------------------------------------
function deep_reg_multiclass(lambda_las,lambda_sen,rho,mu,train,num_classes,dataType,noisy_label,noise_ratio,state,solver,epochs,batch_size,max_rho,mult_rho,percent_features,id)
%---------------------------------------------------------------------

% Run Auxiliary Image Regularizer for classification on different datasets
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
%dataType : 'imagenet' / 'cifar10' /  'mnist' 
%noisy_label : 0/1 want to add noise to the labels?
%noise_ratio : with what ratio? e.g. if you want "50%" noise, set noise_ratio to "0.5"
%state : 'test'/'val'
%solver : 'SADMM'/'svm-sgd'
%percent_features : What percent of the feature values to be kept? 
%		want to make the feature vector more sparse for the sake of memory/computation cost 
%id : 0-9 which random setting to use?

%---------------------------------------------------------------------
% 						Continue Help : Only Valid for 'SADMM' solver
%---------------------------------------------------------------------
%epochs : number of epochs to do training
%batch_size : number of samples to use per iteration
%max_rho: the max value for adaptive rho used in the Augmented Lagrangian term
%mult_rho : {1.1, 1.3, 1.5, 1.7} multiplier factor for rho in each step

if nargin == 14
% when svm is called as a solver
	epochs = 10
	batch_size = 100
	max_rho = 10
	mult_rho = 1.1
end

%calDir : includes extracted features (.h5) in subdirectories of "calDir/train/class_name_i"
%		  and "calDir/test/class_name_i";
%         Read both train and validation data from this directory
%		  For CIFAR10 features are saved in subdirs  "calDir/data_batch_i/class_name_i" and
%		 "test/class_name_i"
%		  (Each feature in its corresponding category folder)
%caffeDir : includes the original png images; only the path name is needed to produce
% 				a txt file including images as a list
%dataDir: the path to save the model weights 
%outDir: the path to save the final result



% --------------------------------------------------------------------
%                                                           Setup data
% --------------------------------------------------------------------

conf.dataset=dataType;



switch conf.dataset
	case 'imagenet'
		conf.calDir='/n/whiskey/xy/vis/samaneh/imagenet_100/';
		conf.caffeDir='/x/data/imagenet7k';
		conf.dataDir = sprintf('imagenetData_100/') ;
		conf.outDir='imagenetData_100/outputHardMulti';
		conf.label_flipping = 'permutation'; % 'confusion'/'permutation'
		conf.numTest = 50;
		conf.numTest_val=20;
		conf.numClasses = num_classes;
		conf.loss='softmax';
	case 'cifar10'
		conf.calDir='/n/whiskey/xy/vis/samaneh/cifar-10/deep_fc7_features/';
		conf.dataDir = sprintf('cifar10_data/') ;
		conf.outDir='cifar10_data/outputHardMulti_from_grpnorm/sain_confusion';
		%conf.outDir='cifar10_data/outputHardMulti_from_l2norm/sain_confusion';
		conf.caffeDir='/y/sazadi/cifar-10/images';
		conf.label_flipping = 'sain_confusion'; % 'confusion'/'permutation'
		conf.numTest = 1000;
		conf.numTest_val=1000;
		conf.numClasses = num_classes;
		conf.loss='softmax';
	case 'mnist'
		conf.calDir='/n/whiskey/xy/vis/samaneh/mnist/deep_fc7_features/';
		conf.dataDir = sprintf('mnist_data/') ;
		conf.outDir='mnist_data/outputHardMulti';
		conf.caffeDir='/y/sazadi/mnist/images';
		conf.label_flipping = 'confusion'; % 'confusion'/'permutation'
		conf.numTest = 1000;
		conf.numTest_val=1000;
		conf.numClasses = num_classes;
		conf.loss='softmax';
end

conf.featureChar='_fc7';
conf.numTrain = train ;		
conf.numTrain_val=train;	
conf.state=state; %% state: 'val'/'test' (Tuning hyper parameters or Testing)

conf.svm.C = rho;
conf.svm.biasMultiplier = 1; 
conf.prefix = 'baseline' ;
conf.randSeed = 1 ;

conf.M=fullfile(conf.dataDir, [conf.prefix '-M.mat']) ;
conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']) ;
conf.grpnorm.solver=solver;%'svm-sgd/SADMM'; 
conf.svm.solver='sgd';

if ~exist(conf.dataDir,'dir')
	mkdir(conf.dataDir)
end

if ~exist(conf.outDir,'dir')
	mkdir(conf.outDir)
end

%rand('state',conf.randSeed) ;
%vl_twister('state',conf.randSeed) ;

if strcmp(conf.dataset,'cifar10')
	classes = dir(fullfile(conf.calDir,'data_batch_5')) ;
else
	
	classes = dir(fullfile(conf.calDir,'train')) ;
end
classes = classes([classes.isdir]) ;
classes = {classes(3:conf.numClasses+2).name} 
conf.numClasses=length(classes);

images_train = {} ;
imageClass_train = {} ;
images_test={};
imageClass_test={};
if strcmp(conf.state,'test')
    numTrain = conf.numTrain;
    numTest = conf.numTest;
else
    numTrain = conf.numTrain_val;
    numTest = conf.numTest_val;
end

im_numm_test=0;
picked_name_test='';
im_numm_train=0;
picked_name_train='';
picked_feat_train='';

lbs=[];


%-------------------------------------------------------------------------
%														Fixed randomness
%-------------------------------------------------------------------------
if ~exist(strcat(conf.calDir,'rng_setting'),'file')
	for id_=0:9
		rng('shuffle','v4');
		rng_setting(id_+1)=rng;
	end
	save(strcat(conf.calDir,'rng_setting'),'rng_setting');
else
	load(strcat(conf.calDir,'rng_setting'));
end


%------------------------------------------------------------------------
%													Read Training Data
%------------------------------------------------------------------------

selTest_each_class=[];

switch conf.dataset
case 'cifar10'

	file = sprintf('/n/whiskey/xy/vis/samaneh/cifar-10/cifar10-train-data-id%g.txt',id);
	fileID=fopen(file);
	C=textscan(fileID,'%s %s\n');
	fclose(fileID);
	images_train=C{1};
	imageClass_train=C{2};

	images_train={images_train{1:min(numTrain*conf.numClasses,length(images_train))}}';
	imageClass_train={imageClass_train{1:length(images_train)}}';
	for ii=1:length(imageClass_train)
		imageClass_train{ii}=str2num(imageClass_train{ii});
	end;
 otherwise

val_subset_tr=cell(length(classes),1);
for ci = 1:length(classes)
        ims = dir(fullfile(conf.calDir, 'train',classes{ci}, '*.h5'))' ;
        s=rng_setting(id+1);
        rng(s);
    if strcmpi(conf.state,'test')	
        ims = vl_colsubset(ims, min(numTrain,length(ims)),'Random');
        ims.name;
    elseif strcmpi(conf.state,'val')
    	val_subset_tr{ci} = randperm(length(ims));
    	ims = vl_colsubset(ims(val_subset_tr{ci}), min(numTrain,length(ims)),'Beginning') ;
    end
    ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
    lbs = [lbs,repmat(ci-1,1,min(numTrain,length(ims)))];
    images_train = {images_train{:}, ims{:}} ;
    imageClass_train{end+1} = ci * ones(1,length(ims)) ;
end
end

selTrain = [1:length(images_train)];
imageClass_train = cat(2, imageClass_train{:}) ;


%------------------------------------------------------------------------
%													Making Noisy Labels
%------------------------------------------------------------------------
feat={};

if noisy_label==1

	if strcmp(conf.label_flipping,'permutation') %premute the labels randomly
		if ~exist(sprintf('%s/randomization_labeling_c%g_s%g_%g.mat',conf.calDir,conf.numClasses,...
			numTrain,noise_ratio),'file')
			num_noisy_per_class=floor(noise_ratio*numTrain);
			noisy_label_ind=[];
			for ii=1:conf.numClasses
				permuted_samples_per_class = randi(numTrain,1,num_noisy_per_class);
				permuted_samples_per_class = (ii-1)*numTrain+permuted_samples_per_class;
				noisy_label_ind = [noisy_label_ind,permuted_samples_per_class];
			end
		
			num_noisy_labels = num_noisy_per_class*conf.numClasses;
			randomized_labels = randi(conf.numClasses,1,num_noisy_labels);
				save(sprintf('%s/randomization_labeling_c%g_s%g_%g.mat',conf.calDir,conf.numClasses,...
				numTrain,noise_ratio),'randomized_labels','num_noisy_labels','noisy_label_ind')
		else
			load(sprintf('%s/randomization_labeling_c%g_s%g_%g.mat',conf.calDir,conf.numClasses,...
			numTrain,noise_ratio))	
		end
		imageClass_train(noisy_label_ind) = randomized_labels.*(randomized_labels~= ...
		imageClass_train(noisy_label_ind))+(min(randomized_labels+1,conf.numClasses)).*( ...
		randomized_labels==imageClass_train(noisy_label_ind));

	else
		num_nz_Q_mat = 0.8*(conf.numClasses^2);
		if ~exist(sprintf('%s/randomization_labeling_%s_c%g_s%g_%g.mat',conf.calDir,...
		conf.label_flipping,conf.numClasses,numTrain,noise_ratio),'file')
			if strcmp(conf.label_flipping,'confusion') %use confusion matrix
				Q=generate_confusion_mat(conf.numClasses,noise_ratio,num_nz_Q_mat);
			else
				load(sprintf('/n/banquet/df/samaneh/Projects/deepLearning/sift-desciptor/confusion_mat_Q_%g_sainbayar.mat',...
				noise_ratio));
			end
			randomized_labels = random_labeling(Q,imageClass_train);
			save(sprintf('%s/randomization_labeling_%s_c%g_s%g_%g.mat',conf.calDir,...
			conf.label_flipping,conf.numClasses,numTrain,noise_ratio),'randomized_labels','Q')
		else
			load(sprintf('%s/randomization_labeling_%s_c%g_s%g_%g.mat',conf.calDir,...
			conf.label_flipping,conf.numClasses,numTrain,noise_ratio))
		end	
		
		imageClass_train=randomized_labels';

	end
end

%------------------------------------------------------------------------
%											Read Training Feature Values
%------------------------------------------------------------------------


for ii = 1:length(selTrain)
    fprintf('Processing %s (%.2f %%)\n', images_train{ii}, 100 * ii / length(images_train)) ;
    %--------------------------------------------------------------------
    %save the training images in a txt file with their corresponding labels;
    %--------------------------------------------------------------------
	splitted=strsplit(images_train{selTrain(ii)},'_fc7_features.h5');
	class_num_ = ceil(ii/numTrain);
    if ~strcmp(conf.dataset,'cifar10')
    	%splitted=strsplit(images_train{selTrain(ii)},'/');
		%splitted= strsplit(splitted{2},conf.featureChar);
		%splitted= strsplit(splitted{1},conf.featureChar);
		%splitted=strsplit(images_train{selTrain(ii)},'?');
		cur_name=strcat(splitted{1},sprintf('.jpg %d',imageClass_train(ii)-1));
	else
		%splitted= strsplit(splitted{7:end},conf.featureChar)
		cur_name=strcat(splitted{1},sprintf('.png %d',imageClass_train(ii)-1));
		
	end
	if strcmp(conf.dataset,'cifar10')
		picked_name_train = sprintf('%s%s/train/%s\n',picked_name_train,conf.caffeDir,cur_name);
		feat{ii} = h5read(fullfile(images_train{selTrain(ii)}),'/feature') ;
	else
	feat_name=images_train{selTrain(ii)};
	feat_lbl=imageClass_train(ii)-1;
	picked_feat_train=sprintf('%s%s/train/%s %d\n',picked_feat_train,conf.caffeDir,feat_name,feat_lbl);
	picked_name_train = sprintf('%s%s/train/%s\n',picked_name_train,conf.caffeDir,cur_name); 
	feat{ii} = h5read(fullfile(conf.calDir,'train', images_train{selTrain(ii)}),'/feature') ;
	end
	feat{ii}=reshape(feat{ii},1,length(feat{ii}));
  
end

images_train=feat;
images_train=cat(1,images_train{:})';



%------------------------------------------------------------------------
%											Read Testing Feature Values
%------------------------------------------------------------------------

for ci = 1:length(classes)    
    %%setting up the test data
    ims={};    				
    if strcmpi(conf.state,'test')
        ims = dir(fullfile(conf.calDir, 'test',classes{ci}, '*.h5'))' ;
        ims = vl_colsubset(ims, min(numTest,length(ims)),'Beginning') ;
        selTest_each_class=[selTest_each_class,length(ims)];
  
    elseif strcmpi(conf.state,'val')
    	switch conf.dataset
			case 'cifar10'
				ims = dir(fullfile(conf.calDir, 'data_batch_5',classes{ci}, '*.h5'))' ;
				ims = vl_colsubset(ims, min(numTest,length(ims)),'Beginning') ; 
				
			otherwise
				ims = dir(fullfile(conf.calDir, 'train',classes{ci}, '*.h5'))' ;
				ims = vl_colsubset(ims(val_subset_tr{ci}), min(numTest,length(ims)),'Ending') ;
		end
		selTest_each_class=[selTest_each_class,length(ims)];
    end
    

    ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;

    images_test = {images_test{:}, ims{:}} ;
    imageClass_test{end+1} = ci * ones(1,length(ims)) ;
    for picked_im=1:min(numTest,length(ims))
        im_numm_test = im_numm_test+1;
		splitted= strsplit(ims{picked_im},conf.featureChar);
		cur_name=strcat(splitted{1},sprintf('.jpg %d',ci-1));
		if strcmpi(conf.state,'val')
			picked_name_test = sprintf('%s%s/train/%s\n',picked_name_test,conf.caffeDir,cur_name);
		else	
        	picked_name_test = sprintf('%s%s/test/%s\n',picked_name_test,conf.caffeDir,cur_name);
        end
    end
end



selTest = [1:length(images_test)] ;
imageClass_test = cat(2, imageClass_test{:}) ;

model.classes = classes ;
model.vocab = [] ;
model.w = [] ;
model.b = [] ;
model.svminfo = [];

%%read test images:

feat={};
for ii = 1:length(selTest)
    fprintf('Processing %s (%.2f %%)\n', images_test{ii}, 100 * ii / length(images_test)) ;
    if strcmpi(conf.state,'test')
        feat{ii} = h5read(fullfile(conf.calDir,'test', images_test{selTest(ii)}),'/feature') ;
        feat{ii}=reshape(feat{ii},1,length(feat{ii}));

 elseif strcmpi(conf.state,'val')
 		switch conf.dataset
			case 'cifar10'
				feat{ii} = h5read(fullfile(conf.calDir,'data_batch_5', ...
				images_test{selTest(ii)}),'/feature') ;
				feat{ii}=reshape(feat{ii},1,length(feat{ii}));
				
			otherwise 
				feat{ii} = h5read(fullfile(conf.calDir,'train', images_test{selTest(ii)}),'/feature') ;
				feat{ii}=reshape(feat{ii},1,length(feat{ii}));
		end
    end
end
images_test=feat;
images_test=cat(1,images_test{:})';



if noisy_label==1
	nsy = 'noisy_label';
else
	nsy = 'no_noise';
	noise_ratio=0;
end

fileID=fopen(sprintf('%s-%d-feat-train-%d-with_labels_%s_%g_mod_%s_id_%g.txt',conf.dataset,...
length(classes),numTrain,nsy,noise_ratio,conf.state,id),'w');
fprintf(fileID,picked_feat_train);
fclose(fileID);
fileID=fopen(sprintf('%s-%d-train-%d-with_labels_%s_%g_mod_%s_id_%g.txt',conf.dataset,length(classes),...
numTrain,nsy,noise_ratio,conf.state,id),'w');
fprintf(fileID,picked_name_train);
fclose(fileID);
fileID=fopen(sprintf('%s-%d-test-%d-with_labels_mod_%s.txt',conf.dataset,length(classes),...
numTest,conf.state),'w');
fprintf(fileID,picked_name_test);
fclose(fileID);
	

%%--------------------------------------------------------------------
%%  												Normalize features
%%--------------------------------------------------------------------

for i=1:size(images_train,2)
	l2_norm=norm(images_train(:,i));
	images_train(:,i)=images_train(:,i)./l2_norm;
	
end


for i=1:size(images_test,2)
	l2_norm=norm(images_test(:,i));
	images_test(:,i)=images_test(:,i)/l2_norm;
end
if strcmp(conf.state,'test')
	save(sprintf('%s/%s_full_training_%g_%g_%g.mat',conf.calDir,conf.dataset,conf.numClasses,...
	numTrain,noise_ratio),'images_train','imageClass_train')
	save(sprintf('%s/%s_full_testing_%g.mat',conf.calDir,conf.dataset,conf.numClasses),...
	'images_test','imageClass_test')
end

%--------------------------------------------------------------------
%													Train the model 
%--------------------------------------------------------------------


switch conf.grpnorm.solver
    case {'svm-sgd', 'svm-sdca'}
        lambda = 1 / (conf.svm.C *  length(selTrain)) 
        w = [] ;
        
        for ci = 1:length(classes)
            perm = randperm(length(selTrain)) ;
            fprintf('Training model for class %s\n', classes{ci}) ;
            y = 2 * (imageClass_train(selTrain) == ci) - 1 ;
				lambda
                [w(:,ci) b(ci) svminfo] = vl_svmtrain(images_train(:, selTrain(perm)), y(perm), lambda, ...
                'Solver', conf.svm.solver, ...
                'MaxNumIterations', 500/lambda, ...
                'BiasMultiplier', conf.svm.biasMultiplier, ...
                'Epsilon', 1e-3);
		        svminfo
		        model.svminfo=svminfo;
          
        end
        model.b = conf.svm.biasMultiplier * b ;
        b_=model.b
        model.w = w ;
        save(conf.modelPath, 'model') ;
        
        
    case {'SADMM'}
    	t_start=tic;
    	
    	%--------------------------------------------------
    	%									Create Matrix M
    	%--------------------------------------------------
		nonzeros_per_sent_per_img=cell(length(selTrain),1);
		starting_im_per_total=zeros(length(selTrain),1);
		row_num=0;
		cols_M=[];rows_M=[];vals_M=[];
		ind_per_im=0;
		ind_prev=0;
		
		sorted_M=sort(abs(images_train(:)),'descend');
		thr=sorted_M(floor(length(selTrain)*(percent_features*size(images_train,1))))
		conf.thr=thr;
		images_train_=images_train.*(abs(images_train)>thr);
		nz_per_img = sum(images_train_>0);
		for ii = 1:length(selTrain)
			ind_prev=ind_prev+ind_per_im;
			ind_per_im=0;
			row_start=row_num+1;
			starting_im_per_total(ii)=row_start;
			ind_per_im=ind_per_im+1;
			
			non_zeros=find(images_train_(:,ii)~=0)';
			cols_M=[cols_M,non_zeros];
			rows_M=[rows_M,row_num+[1:nnz(images_train_(:,ii))]];
			vals_M=[vals_M;images_train_(non_zeros,ii)]; %%the last zero is used for bias term
			row_num=row_num+nnz(images_train_(:,ii));
			
			nonzeros_per_sent_per_img{ii}=nnz(images_train_(:,ii));
						
		end

		M=sparse(rows_M',cols_M,double(vals_M),length(rows_M),size(images_train_,1));
		clear cols_M;
		clear rows_M;
		clear vals_M;

		save(conf.M,'M','nonzeros_per_sent_per_img','starting_im_per_total','images_test','images_train');
		%%removing the unneeded data from the train images

		removing_ind=find(sum(M,1)==0);
		M(:,removing_ind)=[];
		
		images_train(removing_ind,:)=[];
		images_train_(removing_ind,:) = [];
		
		%%removing the unneeded data from the test images
		
		images_test(removing_ind,:)=[];
		
		clear feat
            
        w = [] ;
        rp=randperm(length(selTrain));
        r_=1;
        rp_picked=randperm(floor(length(selTrain)*r_));
        rp=rp(rp_picked);
        Idx=[];
        for d=rp(1:end)
            Idx=[Idx,starting_im_per_total(d):(nonzeros_per_sent_per_img{d}+starting_im_per_total(d)-1)];
        end
        M_before_rows=size(M,1);
        M=M(Idx,:);
        M(size(M,1):M_before_rows,:)=0;
        y= (imageClass_train(selTrain));
        t_pre=toc(t_start)


		%--------------------------------------------------------
		%								Run Optimal Stochastic ADMM
		%--------------------------------------------------------
    	[w,info] = opt_SADMM(conf.dataset,images_train, images_train_,y,M,nz_per_img,...
    	lambda_las,lambda_sen,rho,mu,rp,conf.numClasses,conf.numClasses,'hard',...
    	conf.loss,numTrain,epochs,batch_size,max_rho,mult_rho,noise_ratio,conf.state,1,id);

        OBJECTIVE=info.objective;
        training_time=info.elapsedTime;
    
        info.elapsedTime=training_time;
        iterations=[0:10,11:5:1000,1001:50:100000]; 
        info.elapsedIter=iterations(1:size(w,3));
        
        model.b = zeros(1,length(classes));
        model.w = w ;
end

% --------------------------------------------------------------------------
%                                                Test the Model and Evaluate
% --------------------------------------------------------------------------

% Estimate the class of the test images
acc=[];

switch conf.grpnorm.solver
	
	case{'SADMM'}
		size(model.w)
		prob=zeros(conf.numClasses,length(selTest),size(model.w,3));
		prediction=zeros(length(selTest),size(model.w,3));
		for j=1:size(model.w,3)
			w1=zeros(size(model.w,1),size(model.w,2));
			w1(:,:)=model.w(:,:,j);
			prob(:,:,j)=computeProb(w1,images_test,selTest);
			[pm,prediction(:,j)]=max(prob(:,:,j));
			acc(j)=sum(prediction(:,j)'==imageClass_test(selTest))/length(selTest);

		end
		elapsedIter=0;

		thr=conf.thr;
		save(sprintf('%s/output-%s-%s-%g-%g-%g-%g-%g-%g-%s-%g-%s-%g-%g-%g-%g-%g-%g.mat',...
		conf.outDir,conf.dataset,conf.grpnorm.solver,lambda_las,lambda_sen,rho,mu,conf.numClasses,...
		numTrain,conf.state,noise_ratio,conf.loss,batch_size,max_rho,...
		mult_rho,percent_features,r_,id),'elapsedIter','OBJECTIVE','acc','training_time','thr')
	
	case{'svm-sgd', 'svm-sdca','svm-liblinear'}
	
	
	 scores = model.w' * images_test + model.b' * ones(1,size(images_test,2)) ;



	[drop, imageEstClass] = max(scores, [], 1) ;
	acc=sum(imageEstClass(selTest)==imageClass_test(selTest))/length(selTest)
	svminfo=model.svminfo;

	save(sprintf('%s/output-%s-%s-%g-%g-%g-%s-%g-%s-%g.mat',conf.outDir,conf.dataset,...
	conf.grpnorm.solver,conf.svm.C,conf.numClasses,numTrain,conf.state,...
	noise_ratio,conf.loss,id),'acc','svminfo')
end




% ---------------------------------------------------------------------
function y = computeProb(w,samples,samples_part)
% ---------------------------------------------------------------------
wTs=w'*samples(:,samples_part);
wTs = bsxfun(@minus, wTs, max(wTs, [], 1)); 
%Prevent overflow: subtract some large constant from each wTs before the exponential.
prob = bsxfun(@rdivide, exp(wTs), sum(exp(wTs)));
y=prob;

% ---------------------------------------------------------------------
function randomized_labels = random_labeling(Q,labels)
% ---------------------------------------------------------------------

numClasses=size(Q,1);
randomized_labels=zeros(length(labels),1);
for i = 1:numClasses
    ind = (find(labels==i));
    ind =ind(randperm(length(ind)));
    classes=find(Q(i,:));
    cnt=1;
    for j=1:length(classes)
        start_i=cnt;
        if j==length(classes)
            end_i=length(ind);
        else
            end_i=min(cnt+(round(Q(i,classes(j))*length(ind))-1),length(ind));
        end
        
        randomized_labels(ind(start_i:end_i))=classes(j);
        cnt=end_i+1;
    end
end

% ---------------------------------------------------------------------    
function Q = generate_confusion_mat(num_class,noise_ratio,num_nz)
% ---------------------------------------------------------------------

Q=zeros(num_class,num_class);
density=num_nz/(num_class^2-num_class);
Q=sprand(num_class,num_class,density);
Q=Q-diag(diag(Q));
Q=spdiags((noise_ratio)./(sum(Q,2)+0.00001),0,sparse(num_class,num_class))*Q;
Q=Q+(1-noise_ratio)*speye(num_class);
Q=spdiags(1./(sum(Q,2)+0.00001),0,sparse(num_class,num_class))*Q;




