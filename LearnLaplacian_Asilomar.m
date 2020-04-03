%
% Authors - Eduardo Pavez <eduardo.pavez.carvelli@gmail.com, pavezcar@usc.edu>
% Copyright Eduardo Pavez,  University of Southern California
%Los Angeles, USA, 04/02/2020
%This function implements the algorithm to learn a combinatorial laplacian proposed in  
%Pavez, Eduardo, and Antonio Ortega. "An Efficient Algorithm for Graph Laplacian Optimization Based on Effective Resistances."
%2019 53rd Asilomar Conference on Signals, Systems, and Computers. IEEE, 2019.
function [ Wout, obj_function ] = LearnLaplacian_Asilomar( edm, alpha, edge, param )

%
%%INPUT:
%edm: euclidean distance matrix (matrix form)
%reg: regularization matrix (matrix form)
% edge: edge list m x 2 size
%param: structure with parameters

niter       = param.niter; %max number of epochs
tol         = param.tol; %convergence tolerance
n           = param.n; %number of nodes
isFull      = param.isFull; %indicate if input graph is full (complete graph) or not
rule        = param.rule; %edge selection rule
% rule == 0  cyclic
% rule == 1  random uniform 
% rule == 2  GS, largest generalized gradient (delta_w)
% rule == 3  GS, largest  decrease in obj function

if(rule>3)
warning('rule can only be 0, 1, 2, 3')    ;
end

%compute number of edges
if(isFull && ~isempty(edge))
   warning('when param.isFull==1, edge is ignored');
end

if(isFull)
    m = n*(n-1)/2;
    if(~isempty(edge) && size(edge,1)<m)
        warning('param.isFull==1, override non empty "edge", learning complete graph');
    end
else
    m = size(edge,1);
end
if( numel(alpha)==1)
    reg = (ones(n,n) - eye(n))*alpha(1,1);
end
%compute MWST with weights 1/(edm(i,j) + reg(i,j))
if(isFull)
    A = spones(-eye(n) + ones(n));
else
    A = sparse(edge(:,1), edge(:,2), ones(m,1),n,n);
    A = spones(A+A');
end
%find tree for initializaiton of algo
edm_reg = edm+reg;
[T, ~] = graphminspantree(A.*(edm_reg));
T(T>0) = T(T>0).^(-1);
[I,J,w] = find(T);
W = T+T';
L = w2l(W);

S = compute_pinv_Laplacian(L);
S = S + ones(n,n)/n;
%put necessary edm values on input vector d
[~,~,d ]= find(triu(edm_reg.*A,1));

%initialize weight vector
for i=1:n-1
    A(I(i),J(i)) = w(i);
    A(J(i),I(i)) = w(i);
end

[If,Jf,w]=find( triu(A,1));
w(w==1)=0;
if(isFull)
   edge = [If,Jf]; 
end

[wout,obj_function]=coordinate_descent_Laplacian_Asilomar(d,edge, w,S, tol, niter,rule );

Wout = sparse(edge(:,1), edge(:,2), wout, n ,n);
Wout = Wout + Wout';
obj_function =nonzeros(obj_function);
end
%computes pseudo inverse of combinatorial Laplacian, when it is a tree complexity is O(n^2)
function X = compute_pinv_Laplacian(L)
%assume L is the combLap of a tree, or a connected graph
%approx min degree permutation
n = size(L,1);
p = amd(L);

Lp = L(p,p);

%L11 = Lp(1,1);  %this is a positive value
L21 = Lp(2:end,1); %this is a non positive vector
L22 = Lp(2:end,2:end); %this is an invertible M-matrix

ch = chol(L22,'lower'); %sparse cholesky satisfying ch*ch'=L22

a   = ch'\(ch\L21);         %computes inv(L22)*L21
b   = ch'\(ch\ones(n-1,1)); %
X11 = sum(b)/(1-sum(a))/n;
X21 = -b/n - a * X11;

X22 = ch'\(ch\(eye(n-1) - ones(n-1,n-1)/n - L21*X21'));

X = [X11, X21'; X21, X22];

%X is the pinv of Lp
[~,ip] = sort(p,'ascend'); 
X = X(ip,ip);

X = full((X + X')/2);


end
%Adjacency to Laplacian
function L=w2l(W)
L=diag(sum(W))-W+diag(diag(W));
end