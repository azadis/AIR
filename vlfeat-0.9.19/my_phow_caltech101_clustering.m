function my_phow_caltech101_clustering(num_blocks)
% PHOW_CALTECH101 Image classification in the Caltech-101 dataset
%   This program demonstrates how to use VLFeat to construct an image
%   classifier on the Caltech-101 data. The classifier uses PHOW
%   features (dense SIFT), spatial histograms of visual words, and a
%   Chi2 SVM. To speedup computation it uses VLFeat fast dense SIFT,
%   kd-trees, and homogeneous kernel map. The program also
%   demonstrates VLFeat PEGASOS SVM solver, although for this small
%   dataset other solvers such as LIBLINEAR can be more efficient.
%
%   By default 15 training images are used, which should result in
%   about 64% performance (a good performance considering that only a
%   single feature type is being used).
%
%   Call PHOW_CALTECH101 to train and test a classifier on a small
%   subset of the Caltech-101 data. Note that the program
%   automatically downloads a copy of the Caltech-101 data from the
%   Internet if it cannot find a local copy.
%
%   Edit the PHOW_CALTECH101 file to change the program configuration.
%
%   To run on the entire dataset change CONF.TINYPROBLEM to FALSE.
%
%   The Caltech-101 data is saved into CONF.CALDIR, which defaults to
%   'data/caltech-101'. Change this path to the desired location, for
%   instance to point to an existing copy of the Caltech-101 data.
%
%   The program can also be used to train a model on custom data by
%   pointing CONF.CALDIR to it. Just create a subdirectory for each
%   class and put the training images there. Make sure to adjust
%   CONF.NUMTRAIN accordingly.
%
%   Intermediate files are stored in the directory CONF.DATADIR. All
%   such files begin with the prefix CONF.PREFIX, which can be changed
%   to test different parameter settings without overriding previous
%   results.
%
%   The program saves the trained model in
%   <CONF.DATADIR>/<CONF.PREFIX>-model.mat. This model can be used to
%   test novel images independently of the Caltech data.
%
%     load('data/baseline-model.mat') ; # change to the model path
%     label = model.classify(model, im) ;
%

% Author: Andrea Vedaldi

% Copyright (C) 2011-2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

conf.calDir = 'data/caltech-101' ;
%conf.calDir='101_ObjectCategories';
conf.dataDir = sprintf('data/num_blocks%g',num_blocks) ;
conf.autoDownloadData = true ;
conf.numTrain = 15 ;
conf.numTest = 15 ;
conf.numClasses = 102 ;
conf.numWords = 600 ;
conf.numSpatialX = [2 4] ;
conf.numSpatialY = [2 4] ;
conf.quantizer = 'kdtree' ;
conf.svm.C = 10 ;

conf.svm.solver = 'sdca' ;
%conf.svm.solver = 'sgd' ;
%conf.svm.solver = 'liblinear' ;

conf.svm.biasMultiplier = 1 ;
conf.phowOpts = {'Step', 3} ;
conf.clobber = false ;
conf.tinyProblem = true;
conf.prefix = 'baseline' ;
conf.randSeed = 1 ;

if conf.tinyProblem
    conf.prefix = 'tiny' ;
    conf.numClasses = 5 ;
    conf.numSpatialX = 2 ;
    conf.numSpatialY = 2 ;
    conf.numWords = 300 ;
    conf.phowOpts = {'Verbose', 2, 'Sizes', 7, 'Step', 5} ;
end

conf.vocabPath = fullfile(conf.dataDir, [conf.prefix '-vocab.mat']) ;
conf.histPath = fullfile(conf.dataDir, [conf.prefix '-hists.mat']) ;
conf.histBlockPath = fullfile(conf.dataDir, [conf.prefix '-histsBlock.mat']) ;
conf.M=fullfile(conf.dataDir, [conf.prefix '-M.mat']) ;
conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']) ;
conf.resultPath = fullfile(conf.dataDir, [conf.prefix '-result']) ;
conf.grpnorm.solver='admm';

randn('state',conf.randSeed) ;
rand('state',conf.randSeed) ;
vl_twister('state',conf.randSeed) ;

% --------------------------------------------------------------------
%                                            Download Caltech-101 data
% --------------------------------------------------------------------

if ~exist(conf.calDir, 'dir') || ...
        (~exist(fullfile(conf.calDir, 'airplanes'),'dir') && ...
        ~exist(fullfile(conf.calDir, '101_ObjectCategories', 'airplanes')))
    if ~conf.autoDownloadData
        error(...
            ['Caltech-101 data not found. ' ...
            'Set conf.autoDownloadData=true to download the required data.']) ;
    end
    vl_xmkdir(conf.calDir) ;
    calUrl = ['http://www.vision.caltech.edu/Image_Datasets/' ...
        'Caltech101/101_ObjectCategories.tar.gz'] ;
    fprintf('Downloading Caltech-101 data to ''%s''. This will take a while.', conf.calDir) ;
    untar(calUrl, conf.calDir) ;
end

if ~exist(fullfile(conf.calDir, 'airplanes'),'dir')
    conf.calDir = fullfile(conf.calDir, '101_ObjectCategories') ;
end

% --------------------------------------------------------------------
%                                                           Setup data
% --------------------------------------------------------------------
classes = dir(conf.calDir) ;
classes = classes([classes.isdir]) ;
classes = {classes(3:conf.numClasses+2).name} ;

images = {} ;
imageClass = {} ;
for ci = 1:length(classes)
    ims = dir(fullfile(conf.calDir, classes{ci}, '*.jpg'))' ;
    ims = vl_colsubset(ims, conf.numTrain + conf.numTest) ;
    ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
    images = {images{:}, ims{:}} ;
    imageClass{end+1} = ci * ones(1,length(ims)) ;
end
selTrain = find(mod(0:length(images)-1, conf.numTrain+conf.numTest) < conf.numTrain) ;
selTest = setdiff(1:length(images), selTrain) ;
imageClass = cat(2, imageClass{:}) ;

model.classes = classes ;
model.phowOpts = conf.phowOpts ;
model.numSpatialX = conf.numSpatialX ;
model.numSpatialY = conf.numSpatialY ;
model.quantizer = conf.quantizer ;
model.vocab = [] ;
model.w = [] ;
model.b = [] ;
model.classify = @classify ;

% --------------------------------------------------------------------
%                                                     Train vocabulary
% --------------------------------------------------------------------

if ~exist(conf.vocabPath) || conf.clobber
    
    % Get some PHOW descriptors to train the dictionary
    selTrainFeats = vl_colsubset(selTrain, 30) ;
    descrs = {} ;
    %for ii = 1:length(selTrainFeats)
    parfor ii = 1:length(selTrainFeats)
        im = imread(fullfile(conf.calDir, images{selTrainFeats(ii)})) ;
        im = standarizeImage(im) ;
        [drop, descrs{ii}] = vl_phow(im, model.phowOpts{:}) ;
    end
    
    descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
    descrs = single(descrs) ;
    
    % Quantize the descriptors to get the visual words
    vocab = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
    save(conf.vocabPath, 'vocab') ;
else
    load(conf.vocabPath) ;
end

model.vocab = vocab ;

if strcmp(model.quantizer, 'kdtree')
    model.kdtree = vl_kdtreebuild(vocab) ;
end

% --------------------------------------------------------------------
%                                           Compute spatial histograms
% --------------------------------------------------------------------

if ~exist(conf.histPath) || ~exist(conf.histBlockPath) || conf.clobber
    hists = {} ;
    parfor ii = 1:length(images)
        % for ii = 1:length(images)
        fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images)) ;
        im = imread(fullfile(conf.calDir, images{ii})) ;
        [hists_blocks{ii},sum_hist] = getImageDescriptor(model, im);
    end
    
    hists = cat(2, hists{:}) ;
    save(conf.histPath, 'hists') ;
    
    %%%my added lines to compute the histograms for blocks of each image
    num_blocks=num_blocks^2;
    psix_param=1;
    hists_blocks={};
    hists_blocks_unnorm={};
    psix_blocks={};
    hist_im=cell(length(selTrain),1);
    sentence_per_image=cell(length(selTrain),1);
    nonzeros_per_sent_per_img=cell(length(selTrain),1);
    starting_im_per_total=zeros(length(selTrain),1);
    row_num=0;
    cols_M=[];rows_M=[];vals_M=[];
    ind_per_im=0;
    ind_prev=0;
    for ii = 1:length(selTrain)
        % for ii = 1:length(images)
        fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images)) ;
        im = imread(fullfile(conf.calDir, images{selTrain(ii)})) ;
        [rows,cols,rgb]=size(im);
        ind_prev=ind_prev+ind_per_im;
        ind_per_im=0;
        row_start=row_num+1;
        starting_im_per_total(ii)=row_start;
        [hists_blocks_one,sum_hists_blocks] = getImageDescriptor(model, im);
        for jj=1:num_blocks
                ind_per_im=ind_per_im+1;
                psix_blocks{jj} = vl_homkermap(hists_blocks_one{jj}, psix_param, 'kchi2', 'gamma', .5) ;
                hists_blocks{jj}=double(hists_blocks_one{jj});
                hists_blocks_unnorm{jj}=round(hists_blocks{jj}*sum_hists_blocks{jj});
                %                 non_zeros=find(hists_blocks_unnorm{ind}~=0);
                %                 cols_M=[cols_M;non_zeros];
                %                 rows_M=[rows_M,row_num+[1:nnz(hists_blocks_unnorm{ind})]];
                %                 vals_M=[vals_M;hists_blocks_unnorm{ind}(non_zeros)];
                %                 row_num=row_num+nnz(hists_blocks_unnorm{ind});
                non_zeros=find(psix_blocks{jj}~=0);
                non_zeros_hist=find(hists_blocks_unnorm{jj}~=0);
                cols_M=[cols_M;non_zeros];
                rows_M=[rows_M,row_num+[1:nnz(psix_blocks{jj})]];
                %vals_M=[vals_M;reshape(repmat(hists_blocks_unnorm{jj}(non_zeros_hist),2*psix_param+1,1),length(hists_blocks_unnorm{jj}(non_zeros_hist))*(2*psix_param+1),1)];
                vals_M=[vals_M;psix_blocks{jj}(non_zeros)];
                row_num=row_num+nnz(psix_blocks{jj});
            
            sentence_per_image{ii}(jj)=sum_hists_blocks;
            %             nonzeros_per_sent_per_img{ii}(jj)=nnz(hists_blocks_unnorm{ind});
            nonzeros_per_sent_per_img{ii}(jj)=nnz(psix_blocks{jj});
            
            
        end
        
        
        hist_im{ii}=sum(cat(2,hists_blocks_unnorm{(ind_prev+1):(ind_prev+ind_per_im)}),2);
        psix_im_train{ii}=sum(cat(2,psix_blocks{(ind_prev+1):(ind_prev+ind_per_im)}),2);
        
    end
    %     M=sparse(rows_M',cols_M,double(vals_M),length(rows_M),size(hists_blocks{1},1));
    M=sparse(rows_M',cols_M,double(vals_M),length(rows_M),size(hists_blocks{1},1)*(2*psix_param+1));
    
    hists_blocks = cat(2, hists_blocks{:}) ;
    hists_blocks_unnorm = cat(2, hists_blocks_unnorm{:}) ;
    hist_im=cat(2,hist_im{:});
    psix_im_train=double(cat(2,psix_im_train{:}));
    save(conf.M,'M','sentence_per_image','nonzeros_per_sent_per_img','starting_im_per_total');
    
    %%%The test image
    
    hists_blocks={};
    hists_blocks_unnorm={};
    hist_im_test=cell(length(selTest),1);
    psix_blocks={};
    row_num=0;
    ind_per_im=0;
    ind_prev=0;
    for ii = 1:length(selTest)
        % for ii = 1:length(images)
        fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images)) ;
        im = imread(fullfile(conf.calDir, images{selTest(ii)})) ;
        [rows,cols,rgb]=size(im);
        ind_prev=ind_prev+ind_per_im;
        ind_per_im=0;
        for jj=1:(num_blocks)
            
            [hists_blocks_one,sum_hists_blocks] = getImageDescriptor(model, im);
                ind_per_im=ind_per_im+1;
                psix_blocks{jj}=vl_homkermap(hists_blocks_one{jj}, psix_param, 'kchi2', 'gamma', .5) ;
                hists_blocks{jj}=double(hists_blocks_one{jj});
                hists_blocks_unnorm{jj}=round(hists_blocks{jj}*sum_hists_blocks{jj});
        end
        
        
        hist_im_test{ii}=sum(cat(2,hists_blocks_unnorm{(ind_prev+1):(ind_prev+ind_per_im)}),2);
        psix_im_test{ii}=sum(cat(2,psix_blocks{(ind_prev+1):(ind_prev+ind_per_im)}),2);
        
        
    end
    hist_im_test=cat(2,hist_im_test{:});
    psix_im_test=double(cat(2,psix_im_test{:}));
    save(conf.histBlockPath, 'hists','hists_blocks','hists_blocks_unnorm','hist_im','hist_im_test','psix_im_train','psix_im_test') ;
    
else
    load(conf.histPath) ;
    load(conf.histBlockPath);
    load(conf.M);
end

% --------------------------------------------------------------------
%                                                  Compute feature map
% --------------------------------------------------------------------
%psix_blocks = vl_homkermap(hists_blocks, 1, 'kchi2', 'gamma', .5) ;
%psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;

% --------------------------------------------------------------------
%                                                            Train SVM
% --------------------------------------------------------------------

% if exist(conf.modelPath) & ~conf.clobber
%         load(conf.modelPath) ;
% else
% switch conf.grpnorm.solver
% case 'sadmm'
%                             svm = train(imageClass(selTrain)', ...
%                                 sparse(double(psix(:,selTrain))),  ...
%                                 sprintf(' -s 3 -B %f -c %f', ...
%                                 conf.svm.biasMultiplier, conf.svm.C), ...
%                                 'col') ;
%                             w = svm.w(:,1:end-1)' ;
%                             b =  svm.w(:,end)' ;
%                             case 'svm'
%                                 load my_rand_data.mat
%                                 lambda = 1 / (conf.svm.C *  length(selTrain)) ;
%                                 w = [] ;
%                                 parfor ci = 1:length(classes)
%                                     perm = randperm(length(selTrain)) ;
%                                     fprintf('Training model for class %s\n', classes{ci}) ;
%                                     y = 2 * (imageClass(selTrain) == ci) - 1 ;
%                                     [w(:,ci) b(ci) info] = vl_svmtrain(hist_im(:, rp), y(rp), lambda, ...
%                                         'Solver', conf.svm.solver, ...
%                                         'MaxNumIterations', 50/lambda, ...
%                                         'BiasMultiplier', conf.svm.biasMultiplier, ...
%                                         'Epsilon', 1e-3);
%                                 end
%         case 'admm'

w = [] ;
rp=[1:length(selTrain)];
% rp=randperm(length(selTrain));
% save('my_rand_data_caltech.mat','rp')
if conf.tinyProblem
    load my_rand_data_caltech_tiny.mat
else
load my_rand_data_caltech.mat
end

Idx=[];
for d=rp(1:length(selTrain))
    cnt=0;
    for sd=1:length(nonzeros_per_sent_per_img{d})
        cnt=cnt+nonzeros_per_sent_per_img{d}(sd);
    end
    Idx=[Idx,starting_im_per_total(d):(cnt+starting_im_per_total(d)-1)];
end
M=M(Idx,:);
training_time=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%loop over the parameters to find  the best ones
ll_las=[0.001,0.1,10,1000];
ll_sen=[0.001,0.1,10,1000];
rho_vec=[0.01,0.1,10,1000];
ll_las=0.001;
ll_sen=0.001;
rho_vec=0.01;
ind_ll=0;
ind_ll_str=[];
for ll=1:length(ll_las)
    for ls=1:length(ll_sen)
        for rh=1:length(rho_vec)
            ind_ll=ind_ll+1;
            ind_ll_str=[ind_ll_str,strcat('a',num2str(ind_ll)),strcat('b',num2str(ind_ll)),strcat('c',num2str(ind_ll)),strcat('d',num2str(ind_ll))];
            lambda_las=ll_las(ll);
            lambda_sen=ll_sen(ls);
            rho=rho_vec(rh);
            W_iterations=[];
            OBJECTIVE=cell(length(classes),1);
            training_time=0;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for ci = 1:length(classes)
                fprintf('Training model for class %s\n', classes{ci}) ;
                y = 2 * (imageClass(selTrain) == ci) - 1 ;
                %                 lambda_las=0.01;rho=1;
                %                 [w(:,ci),info] = ADMM(hists_blocks(:, selTrain(perm)), y(perm),eye(size(M,1),size(M,2)),lambda_las,1,rho,'simple');
                %lambda_las=0.001;rho=10;lambda_sen=0.01;
                
                %                 lambda_las=0.1,rho=10;lambda_sen=0.01;
                
                [w,info] = ADMM(psix_im_train(:, (rp)), y(rp),M,nonzeros_per_sent_per_img, lambda_las,lambda_sen,rho,rp,'hard');
                %                 [w,info] = ADMM(hist_im(:, (rp)), y(rp),M,nonzeros_per_sent_per_img, lambda_las,lambda_sen,rho,rp,'hard');
                
                if ci>1
                    if size(w,2)<size(W_iterations,2)
                        w1=zeros(size(W_iterations,1),size(W_iterations,2));
                        diff_size=size(W_iterations,2)-size(w,2);
                        w1(:,1:size(w,2))=w;
                        w1(:,size(w,2):size(W_iterations,2))=repmat(w(:,end),1,diff_size+1);
                        w=w1;
                        info.elapsedTime=[info.elapsedTime;(info.elapsedTime(end)+10)*ones(diff_size,1)];
                        
                        
                    elseif size(w,2) > size(W_iterations,2)
                        W1_iterations=zeros(size(w,1),size(w,2),ci);
                        for cc=1:(ci-1)
                            diff_size=size(w,2)-size(W_iterations,2);
                            W1_iterations(:,1:size(W_iterations,2),cc)=W_iterations(:,1:size(W_iterations,2),cc);
                            W1_iterations(:,size(W_iterations,2):size(w,2),cc)=repmat(W_iterations(:,end,cc),1,diff_size+1);
                            
                        end
                        W_iterations=W1_iterations;
                        training_time=[training_time;(info.elapsedTime(end)+10)*ones(diff_size,1)];
                    end
                end
                W_iterations(:,:,ci)=w;
                OBJECTIVE{ci}=info.objective;
                training_time=training_time+info.elapsedTime;
                
                
            end
            w=W_iterations;
            info.elapsedTime=training_time;
            iterations=[0,1:5:100,101:20:3000,3001:100:4000,4001:200:5000,6000];
%             if size(W_iterations,2)<length(iterations)
                info.elapsedIter=iterations(1:size(W_iterations,2));
%             else
%                 info.elapsedIter=iterations;
%                 
%             end
            
            %                     end
            
            model.b = zeros(1,length(classes));
            model.w = w ;
            
            save(conf.modelPath, 'model','info') ;
            
            
            
            % --------------------------------------------------------------------
            %                                                Test SVM and evaluate
            % --------------------------------------------------------------------
            
            % Estimate the class of the test images
            %  scores = model.w' * psix + model.b' * ones(1,size(psix,2)) ;
            % model.w=rand(size(model.w,1),size(model.w,2));
            acc=[];
            for j=1:size(model.w,2)
                w1=zeros(size(model.w,1),size(model.w,3));
                w1(:,:)=model.w(:,j,:);
                %     scores = w1' * [hist_im,hist_im_test] + model.b' * ones(1,size([hist_im,hist_im_test],2)) ;
                scores = w1' * [psix_im_train,psix_im_test] + model.b' * ones(1,size([psix_im_train,psix_im_test],2)) ;
                
                [drop, imageEstClass] = max(scores, [], 1) ;
                
                % Compute the confusion matrix
                idx = sub2ind([length(classes), length(classes)], ...
                    imageClass(selTest), imageEstClass(76:end)) ;
                confus = zeros(length(classes)) ;
                confus = vl_binsum(confus, ones(size(idx)), idx) ;
                acc=[acc,100 * mean(diag(confus)/conf.numTest)];
            end
            % Plots
            figure(7) ; clf;
            subplot(1,2,1) ;
            imagesc(scores(:,[selTrain selTest])) ; title('Scores') ;
            set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
            subplot(1,2,2) ;
            imagesc(confus) ;
            title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
                100 * mean(diag(confus)/conf.numTest) )) ;
            print('-depsc2', [conf.resultPath '.ps']) ;
            save([conf.resultPath '.mat'], 'confus', 'conf') ;
            figure(12);hold on;
            color=rand(3,1);
            plot(info.elapsedIter,acc,'color',color)
        end
    end
end
figure(13);
legend(ind_ll_str);
figure(12);
legend(ind_ll_str);
elapsedIter=info.elapsedIter;
save(sprintf('data/output/output-calnet-%g-%g-%g-%g.mat',lambda_las,lambda_sen,rho,num_blocks),'elaspedIter','OBJECTIVE','acc')




% -------------------------------------------------------------------------
function im = standarizeImage(im)
% -------------------------------------------------------------------------

im = im2single(im) ;
if size(im,1) > 480, im = imresize(im, [480 NaN]) ; end
end
% -------------------------------------------------------------------------
function [hist,sum_hist] = getImageDescriptor(model, im)
% -------------------------------------------------------------------------

im = standarizeImage(im) ;
width = size(im,2) ;
height = size(im,1) ;
numWords = size(model.vocab, 2) ;

% get PHOW features
[frames, descrs] = vl_phow(im, model.phowOpts{:}) ;

%MY ADDED LINES%cluster the frames based on their location
[frameCent,frameNum]=vl_kmeans(frames(1:2,:),num_blocks,'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50);
hist=cell(num_blocks,1);
sum_hist=cell(num_blocks,1);
for clusterNum=1:num_blocks
    descrs_each=descrs(find(frameNum==clusterNum));
    frames_each=frames(:,find(frameNum==clusterNum));
% quantize local descriptors into visual words
switch model.quantizer
    case 'vq'
        [drop, binsa] = min(vl_alldist(model.vocab, single(descrs_each)), [], 1) ;
    case 'kdtree'
        binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
            single(descrs_each), ...
            'MaxComparisons', 50)) ;
end

for i = 1:length(model.numSpatialX)
    binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames_each(1,:)) ;
    binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames_each(2,:)) ;
    
    % combined quantization
    bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
        binsy,binsx,binsa) ;
    hist{clusterNum} = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
    hist{clusterNum} = vl_binsum(hist, ones(size(bins)), bins) ;
    hists{i} = single(hist{clusterNum} / sum(hist{clusterNum})) ;
end
sum_hist{clusterNum}=sum(hist{clusterNum});
if isnan(sum_hist{clusterNum})
    sum_hist{clusterNum}
end
hist{clusterNum} = cat(1,hists{:}) ;

hist{clusterNum} = hist{clusterNum} / sum(hist{clusterNum}) ;
end
end

% -------------------------------------------------------------------------
function [className, score] = classify(model, im)
% -------------------------------------------------------------------------

hist = getImageDescriptor(model, im) ;
psix = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5) ;
scores = model.w' * psix + model.b' ;
[score, best] = max(scores) ;
className = model.classes{best} ;
end