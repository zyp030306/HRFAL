clear;
clc;
warning off;
addpath(genpath('datasets/'));
addpath(genpath('measure/'));
data_name = 'UCI';
fprintf('\ndata_name: %s\n', data_name);
load([ data_name, '.mat']); 
Y1=gt;
%%
 k = length(unique(Y1)); 
 V = length(X); 
for v = 1 : V   
     X{v} =X{v}';
end
for v = 1 : V   
    X{v} = mapstd(X{v},0,1)';   
end
parma1 = [2,3,4];
alpha = [10];
beta = [1000];
gamma = [0.1];
p = [0.9];
%%
for i=parma1
    a=i;
    fprintf('params:\t a=%f \t alpha=%f \t beta=%f gamma=%f p=%f \n',a,alpha,beta,gamma,p);
    tic;
    [obj,results_log,C] = HRFAL(X, Y1, k,alpha,beta,a,gamma,p);
end




