function [obj,results_log,C] = HRFAL(X, Y1, k,alpha,beta,a,gamma,p)
%% initialize
v = length(X);
n = size(Y1, 1);
for i = 1:v
   dd(i) = size(X{i}, 1);
end
mu = 1;
max_mu = 10^6;
rho = 1.5;
m=a*k;
for i=1:v
    A{i} = randn(dd(i),m);
end
Z=zeros(m,n);
Y= zeros(m,n);
L=zeros(m);
C = rands(m,n);
for i = 1:v
   D{i} = C;
end
flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
    %% optimize A
for i=1:v
    A{i}=(2*X{i}*(C+D{i})')/(2*(C+D{i})*(C+D{i})'+2*alpha*L + eps);
end  
    %% optimize D
    for iv=1:v
        D{iv}=(A{iv}'*A{iv}+beta*C*C')\(A{iv}'*X{iv}-A{iv}'*A{iv}*C);
    end
    for i=1:v
        M{i}=D{i};      
        M{i} = M{i}./(sum(M{i},2)*ones(1,n));
        M{i}(find(M{i}<0.00005))=0;   
        D{i}=M{i};
    end
    %% optimize C
    tempA=zeros(m,n);
    tempB=zeros(m,n);
    tempC=zeros(m);
    tempD=zeros(m);
    for iv=1:v
        tempA=tempA+A{iv}'*X{iv};
        tempB=tempB+A{iv}'*A{iv}*D{iv};
        tempC=tempC+A{iv}'*A{iv};
        tempD=tempD+D{iv}*D{iv}';
    end
    C=(2*tempC+mu*eye(m)+2*beta*tempD)\(2*tempA-2*tempB+mu*Z-Y);
            param.num_view =1; 
        HG = gsp_nn_hypergraph(C, param);
        L = HG.L;
    %% optimize Z
  tempZ = C + Y/mu;
  [UUU,sigma,VVV] = svd(tempZ,'econ');
    sigma = diag(sigma);
    xi = spw(sigma,gamma/mu,p);
    Z = UUU*diag(xi)*VVV';
    %% optimize Y
        Y = Y + mu*(C-Z);
    mu = min(max_mu,rho*mu);   
        err = C-Z;
    err2= max(abs(err(:)));
    obj(iter) = err2;
    if (err2 < 1e-4 && iter>15)||iter>50
         flag = 0;
    end
end
rng(4234,'twister') 
labels=litekmeans(C', k, 'MaxIter', 100,'Replicates',10);
t=toc;
results_log = Clustering8Measure(labels,Y1) ; % ACC nmi Purity Fscore Precision Recall AR Entropy
fprintf('ACC:%5.4f\tNMI:%5.4f\tPurity:%5.4f\tFscore:%5.4f\tPrecision:%5.4f\tRecall:%5.4f\tAR:%5.4f\ttimes:%f\n',[results_log(1) results_log(2) results_log(3) results_log(4) results_log(5) results_log(6) results_log(7)],t);
 

         
         
    
