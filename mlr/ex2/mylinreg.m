clear;
clc;

% load data
load('dataLinReg2D.txt');

% decompose loaded data into X and Y
n = size(dataLinReg2D, 1);
X = dataLinReg2D(:,1:2);
Y = dataLinReg2D(:,3);

%prepend 1 (constant)
X = [ones(n,1),X];

%compute beta
%beta = inv(X'*X)*X'*Y;
XTX = X'*X;
lam = 0.01;
reg = lam*eye(rows(XTX));
beta = inv(XTX + reg)*X'*Y;

%error
Ypred = X*beta;
err = Y-Ypred;
e = (err'*err)/rows(err)


%prepare for display
[a b] = meshgrid(-2:0.1:2, -2:0.1:2);
Xgrid = [ones(length(a(:)),1), a(:), b(:)];
Ygrid = Xgrid*beta;
Ygrid = reshape(Ygrid, size(a));

%plot
scatter3(dataLinReg2D(:,1), dataLinReg2D(:,2), dataLinReg2D(:,3));
surface(a,b,Ygrid);
