clear all;

function result = sigmoid(z)
result = 1/(exp(-z)+1);
endfunction

function result = applysigmoid(X)
result = arrayfun(@sigmoid,X);
endfunction

function result = gradient(X, y, lam, beta)
p = applysigmoid(X*beta);
result = transpose(p-y)*X + 2*lam*transpose(beta);
endfunction

function result = hessian(X, y, lam, beta)
p = applysigmoid(X*beta);
result = transpose(X)*diag(p.*(1-p))*X + 2*eye(columns(X));
endfunction

n=5;
d=3;
X = randn(n,d)
y = randi(2,n,1)-1
lam = 0
beta = rand(d,1)
gradient(X,y,lam,beta)
hessian(X,y,lam,beta)