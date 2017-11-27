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

