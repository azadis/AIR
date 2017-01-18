%--------------------------------------------------------------------------
function [W_iterations,info] = opt_SADMM(dataset,samples, samples_, labels,M,nz_per_img,lambda_las,lambda_sen,rho,mu0,rp,class_number,class_tot,simplicity,loss,numTrain,epochs,batch_size,max_rho,mult_rho,noise_ratio,state,norm_per_class,id)
%--------------------------------------------------------------------------

%Doing optimization by use of Optimal Stochastic ADMM method


%W_iterations : optimal vector w found after specific iterations
%OBJval       : objective function value as a function of X_iterations


%-------------------------------------------------------------------------
%										    Data Setup and Initializations
%-------------------------------------------------------------------------
switch dataset
	case 'imagenet'
		paramDir = '/n/whiskey/xy/vis/samaneh/imagenet_100/data/parameters/parallel';
	case 'cifar10'
		paramDir = '/n/whiskey/xy/vis/samaneh/cifar-10/data/parameters/';
	case 'mnist'
		paramDir = '/n/whiskey/xy/vis/samaneh/mnist/data/parameters/';
	case 'nus-lite'
		paramDir = '/n/whiskey/xy/vis/samaneh/NUS-WIDE-LITE/data/parameters/try';
end

rho0 = rho;
samples = double(samples);
[V,D]  =  size(samples);
[N,V] = size(M);
num_lb_orig = length(unique(labels));
num_lb = length(unique(labels));
num_lb  =  (num_lb == 2) + (num_lb ~= 2) * num_lb;

num_train_samples = D;%num of samples used for training part
num_reg_samples  =  length(rp); %num of samples used for regularization part
max_it = epochs * D;
removing_ratio  =  num_reg_samples/num_train_samples;


Lbl = [];
for i = 1:num_lb
    Lbl = [Lbl;sparse(ismember(labels,i))];
end
Lbl = Lbl';

MTM = diag(sum(M.^2)');
mu = mu0;
w = zeros(V,num_lb);
v = M * w;
u = zeros(N,num_lb);

w_bar = w;
v_bar = v;
u_bar = u;

omega_sen = 0;
word_num = 0;
sent_num = 0;

rp_size = length(rp);
if (~ exist(sprintf('%s/data_vectors_%g_%g.mat',paramDir,rp_size,numTrain)))
	sent_num=length(rp);
	word_num=sum(nz_per_img(rp));
	data_bs=ones(word_num,1);
	rows_bs=nonzeros(reshape((samples_(:,rp)>0)*spdiags((1:length(rp))',0,length(rp),length(rp)),length(rp)*size(samples_,1),1));
	cols_bs=1:word_num;
	block_sum=sparse(rows_bs,cols_bs,data_bs,sent_num,N);
	lambda_each_sent=lambda_sen./sqrt(nz_per_img(rp))';
	pos_labels=labels-min(labels)+1;
	img_per_class=zeros(max(pos_labels)-min(pos_labels)+1,1);
	for i=1:length(img_per_class)
		img_per_class(i)=sum(pos_labels==i);
	end;
	if strcmp(dataset,'nus-lite')
		lambda_each_sent=lambda_each_sent.*(D/num_lb_orig)./(img_per_class(pos_labels(rp)));
	end
	v_norm_factor=1./sqrt(nz_per_img(rp)');
		
	save(sprintf('%s/data_vectors_%g_%g.mat',paramDir,rp_size,numTrain),'block_sum','lambda_each_sent','v_norm_factor')
else
	load(sprintf('%s/data_vectors_%g_%g.mat',paramDir,rp_size,numTrain));
end

block_expand = block_sum';
lambda_each_sent=repmat(lambda_each_sent,1,num_lb);

%-------------------------------------------------------------------------------	
% OBJECTIVE = \lambda_sen * normalized \sum ||.||_2 + \lambda_las * ||.||_1 + Loss
%-------------------------------------------------------------------------------

omega_sen = sum(sum(bsxfun(@times,v_norm_factor,((block_sum * (v_bar.^2)).^(1/2)))));
L  =  computeCost(w_bar,samples,labels,mu,loss,num_lb);
obj = lambda_sen * (1/num_reg_samples) * (1/num_lb) * omega_sen + lambda_las * sum(sum(abs(v_bar))) + L

%-------------------------------------------------------------------------------

saved_ind=[0:10,11:5:1000,1001:50:100000];
if size(w,2) == 1
	W_iterations = zeros(size(w,1),min(max_it,length(saved_ind)));
else 
	W_iterations = zeros(size(w,1),size(w,2),min(max_it,length(saved_ind)));
end

OBJval = zeros(length(saved_ind),1);
REGval = zeros(length(saved_ind),1);
t_final_iter = zeros(length(saved_ind),1);
ind_saved = 1;
if size(w,2) == 1
	W_iterations(:,ind_saved) = w_bar;
else 
	W_iterations(:,:,ind_saved) = w_bar;
end

OBJval(ind_saved) = obj;
REGval(ind_saved) = lambda_sen * (1/num_reg_samples) * (1/num_lb) * omega_sen;
t_final_iter(ind_saved) = 0;
k = 0;
t_iter = tic;
permutation = randperm(D);
samples = samples(:,permutation);
labels = labels(permutation);

unseen_samples = randperm(D);

for s_id_all = 1:batch_size:(epochs * D);
	
	len_rem = length(unseen_samples);
	if len_rem<batch_size
		samples_ind = unseen_samples;
		unseen_samples = randperm(D);
		samples_ind = [samples_ind,unseen_samples(1:(batch_size-len_rem))];
		unseen_samples = unseen_samples((batch_size-len_rem+1):end);
	else
		samples_ind = unseen_samples(1:(batch_size));
		unseen_samples = unseen_samples((batch_size+1):end);
	end
	sample = samples(:,samples_ind);
	label = labels(samples_ind);
	
	k = k+1;
	%-------------------------------
	%		set Parameters
	%-------------------------------
	%%hinge loss : strongly convex function
	rho = min(rho * mult_rho,max_rho);
	mu_ = 1;
	eta  =   2/(mu_ * (k+2));
   % eta  =  1/(1+sqrt(2 * k));
	teta_k = 2/(k+2);
	
	%-------------------------------
	%       Update w
	%-------------------------------
	gk  =  computeGrad(w,sample,label,mu,loss,num_lb,num_train_samples,V);
	
	w = (rho * MTM/removing_ratio+(1/(eta)) * speye(V))\(-gk+rho * (1/removing_ratio) *...
	 M' * (v+u/(rho))+(1/(eta)) * w);
	w_bar = (1-teta_k) * w_bar+ teta_k * w;
	
	%-------------------------------
	%       Update v
	%-------------------------------
	v_in = wthresh(M * w-u/(rho),'s',lambda_las/rho);
	v_prim = (block_sum * (v_in.^2)).^(1/2);
	v = (block_expand * (mysoftthresh(v_prim,lambda_each_sent/(rho)))) .* v_in;
	v_bar = (k == 1) * ((2/3) * v)+(k ~= 1) * ((1-teta_k) * v_bar+(teta_k) * v);
	
	%-------------------------------
	%       Update u
	%-------------------------------
	u = u-rho * (M * w-v);
	u_bar = (k == 1) * ((2/3) * u)+(k~= 1) * ((1-teta_k) * u_bar+(teta_k) * u);
	

	%-------------------------------------------------------------------------------	
	% OBJECTIVE = \lambda_sen * normalized \sum ||.||_2 + \lambda_las * ||.||_1 + Loss
	%-------------------------------------------------------------------------------
	omega_sen = sum(sum(bsxfun(@times,v_norm_factor,((block_sum * (v_bar.^2)).^(1/2)))));

	obj_lossPart = computeCost(w_bar,samples,labels,mu,loss,num_lb);
	
	obj = lambda_sen * (1/num_reg_samples) * (1/num_lb) * omega_sen + lambda_las * sum(sum(abs(v_bar))) + obj_lossPart
	
	
	if (ismember(k,saved_ind))
		ind_saved = ind_saved+1;
		t_final_iter(ind_saved) = toc(t_iter);
		if size(w,2) == 1
			W_iterations(:,ind_saved) = w_bar;
		else 
			W_iterations(:,:,ind_saved) = w_bar;
		end
		%%--------------------take snapshot of the parameters
		%save(sprintf('/n/whiskey/xy/vis/samaneh/imagenet_100/data/parameters/parameters_%g_%g_%g_%g_%g_%g_%s_.mat',...
		%lambda_las,lambda_sen,rho,numTrain,class_number,class_tot,loss),'W_iterations','saved_ind');
		%save(sprintf('/n/whiskey/xy/vis/samaneh/CUB_200_2011/data/parameters/parameters_%g_%g_%g_%g_%g_%g_%s_.mat',...
		%lambda_las,lambda_sen,rho,numTrain,class_number,class_tot,loss),'W_iterations','saved_ind');
		OBJval(ind_saved) = obj;
		REGval(ind_saved) = lambda_sen * omega_sen/(num_lb * num_reg_samples);
					
	end    
end
ind_saved = ind_saved+1;
t_final_iter(ind_saved) = toc(t_iter);
if size(w,2) == 1
	W_iterations(:,ind_saved) = w_bar;
else 
	W_iterations(:,:,ind_saved) = w_bar;
end
OBJval(ind_saved) = obj;
REGval(ind_saved) = lambda_sen * omega_sen/(num_lb * num_reg_samples);
if size(w,2) == 1
	W_iterations = W_iterations(:,1:ind_saved);
else 
	W_iterations = W_iterations(:,:,1:ind_saved);
end

OBJval = OBJval(1:ind_saved);
REGval = REGval(1:ind_saved);
t_final_iter = t_final_iter(1:ind_saved); 

info.elapsedTime = t_final_iter;
info.objective = OBJval;
info.W_iterations = W_iterations;
info.regularization = REGval;

if size(w,2) == 1
	W_iterations_last = W_iterations(:,end);
else
	W_iterations_last = W_iterations(:,:,end);
end
	

if strcmp(state,'test') & (num_lb ~= 1)
		save(sprintf('%s/output_par_optADMM_%g_%g_%g_%g_%g_%g_%g_%g_%s_%g_%g.mat',paramDir,...
		lambda_las,lambda_sen,rho,mu0,numTrain,noise_ratio,class_number,class_tot,loss,norm_per_class,id),...
		'W_iterations_last','info','saved_ind');
	elseif num_lb == 1
		save(sprintf('%s/output_par_optADMM_%g_%g_%g_%g_%g_%g_%g_%g_%s_%g_%g_%g_%g_%g.mat',...
		paramDir,lambda_las,lambda_sen,rho0,mu0,numTrain,noise_ratio,class_number,class_tot,...
		loss,batch_size,max_rho,mult_rho,norm_per_class,id),'W_iterations','info','saved_ind');
	end
end


%------------------------------------------------------------------
function g = sub_grad_log(sample,label,w)
%------------------------------------------------------------------
% Gradient of logistic loss

[V,D] = size(sample);
exponent_part = exp(-label' .* (sample' * w))./(1+exp(-label' .* (sample' * w)));
exponent_part(isnan(exponent_part)) = 1;
g = -sum(repmat(label,V,1)' .* sample' .* repmat(exponent_part,1,V),1);
g = g';

end

%------------------------------------------------------------------
function g = sub_grad_hinge(sample,label,w)
%------------------------------------------------------------------
% Gradient of Hinge Loss

[V,D] = size(sample);
if D == 1
if  (label' .* (sample' * w)<1)
	g = -label * sample;
else
    g = 0;
end;

else 
g =  -sum(repmat(((label' .* (sample' * w))<1),1,V) .* (repmat(label,V,1)' .* sample'));
g = g';
end
end


%------------------------------------------------------------------
function y1 = mysoftthresh(y,nu)
%------------------------------------------------------------------
y1 = max(y-nu,0)./y;
y1(isnan(y1)) = 0;
end


%------------------------------------------------------------------
function y  =  computeCost(w,samples,labels,mu,loss,num_lb)
%------------------------------------------------------------------
D  =  size(samples,2);
switch loss
    case 'hinge'
   		 L_in = -labels' .* (samples' * w);
        L = max(0,1+L_in);
    case 'log'
    L_in = -labels' .* (samples' * w);
    L = log(1+exp(L_in));
    L(find(isinf(L) == 1)) = L_in(find(isinf(L) == 1));
    case 'softmax'
	Lbl = [];
	for i = 1:num_lb
		Lbl = [Lbl;sparse(ismember(labels,i))];
	end
	Lbl = Lbl';
    L = computeCost_softmax(w' * samples,Lbl,mu);
    
end
y  =  (1/D) * sum(L)+(mu/(2 * num_lb)) * (sum(sum(w.^2)));

end



%------------------------------------------------------------------
function y = computeGrad(w,samples,labels,mu,loss,num_lb,num_train_samples,V)
%------------------------------------------------------------------
loss
switch loss
    case 'hinge'
        g_in = sub_grad_hinge(samples,labels,w);
    case 'log'
        g_in = sub_grad_log(samples,labels,w);
    case 'softmax'
		Lbl = [];
		for i = 1:num_lb
			Lbl = [Lbl;sparse(ismember(labels,i))];
		end
		Lbl = Lbl';
    	g_in = computeGradSoftmax(w,samples,Lbl,num_lb,V);
end
g_in = g_in * num_lb;
y = g_in+(mu * num_train_samples) * w; 
end

%------------------------------------------------------------------
function y = computeCost_softmax(wTs,Lbl,mu)
%------------------------------------------------------------------

wTs =  bsxfun(@minus, wTs, max(wTs, [], 1)); 
%Prevent overflow: subtract some large constant from each wTs before the exponential.

y = -sum(sum(Lbl .* (wTs')))+sum(log(sum(exp(wTs))));

end


%------------------------------------------------------------------
function y = computeGradSoftmax(w,samples,Lbl,num_lb,V)
%------------------------------------------------------------------
w = reshape(w,V,num_lb);
wTs  =  w'  * samples;
wTs  =  bsxfun(@minus, wTs, max(wTs, [], 1)); 
%Prevent overflow: subtract some large constant from each wTs before the exponential.
prob  =  bsxfun(@rdivide, exp(wTs), sum(exp(wTs)));
Lbl = Lbl';
for lb = 1:num_lb
   g_in(:,lb) = -sum(bsxfun(@times,samples,Lbl(lb,:)-prob(lb,:)),2);
end
y = g_in;    
end




