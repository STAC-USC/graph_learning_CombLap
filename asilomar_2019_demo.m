%asilomar 2019 demo
%
% Authors - Eduardo Pavez <eduardo.pavez.carvelli@gmail.com, pavezcar@usc.edu>
% Copyright Eduardo Pavez,  University of Southern California
%Los Angeles, USA, 04/02/2020
%
%%

clear;
rng('default');
addpath(genpath('mex/'));
% prepare data
NA = 100;
NB = NA;
d = 10;

muA = 0.5*ones(d,1);
muB = -1*muA;

%generate N random data points
n = NA + NB;
dataA = mvnrnd(muA,eye(d,d),NA);
dataB = mvnrnd(muB,eye(d,d),NB);
data = [dataA; dataB];

%compute edge cost matrix, in this case we use squared distances between
%rows of data

G = data*data';
max_diag = max(diag(G));
G = G/max_diag; %keeps the scale of the graph weights not so small
distance_matrix = diag(G) * ones(1,n) + ones(n,1)*diag(G)' -2*G;

%% set parameters
alpha =0; %regularization
param.niter = 1000; % max number of epochs
param.tol = 1.e-6;
param.n =n; %number of nodes
param.rule = 0; %edge selection rule (cyclic rule)

%% learn a graph without specifying edge set, i.e. using a complete edge
%set as input
param.isFull = 1; %indicates input graph is the full graph
edge=[]; %not required so leave empty
tic;
[ W1, obj_function1 ] = LearnLaplacian_Asilomar( distance_matrix, alpha, edge, param );
toc;


%% learn a graph with an input edge set based on K nearest neighbors.
param.isFull = 0; %indicate the input edge set is not the complete graph
param.rule = 2; %use PGS rule

K=5;
knn_out = knnsearch(data,data,'K',K+1);

edge = [];
e1 = 1:1:n;
e1 = e1';
idx = knn_out(:,2:end);
for i=1:K
    edge = [edge ; e1 , idx(:,i)];
    
end
A = sparse(edge(:,1), edge(:,2), 1, n,n);
A = spones(A+A');
[I,J] = find(triu(A,1));
edge = [I,J];
tic;
[ W2, obj_function2 ] = LearnLaplacian_Asilomar( distance_matrix, alpha, edge, param );
toc;




