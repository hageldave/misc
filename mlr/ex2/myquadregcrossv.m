clear;
clc;

% load data
load('dataQuadReg2D_noisy.txt');
data = dataQuadReg2D_noisy;

function e = error4regularization(lam, data)
  % decompose loaded data into X and Y
  n = size(data, 1);
  X = data(:,1:2);
  Ycomplete = data(:,3);
  
  %make features
  X1 = X(:,1);
  X2 = X(:,2);
  Xcomplete = [ones(n,1),X, X1.*X1, X1.*X2, X2.*X2];

  %cross validation
  K = 5;
  e = 0;
  for k = 0:(K-1)
    %make X Y for training and validation
    X = [];
    Y = [];
    Xv = [];
    Yv = [];
    for i = 1:rows(Xcomplete)
      if mod(i,K) == k
        Xv = [Xv;Xcomplete(i,:)];
        Yv = [Yv;Ycomplete(i)];
      else
        X = [X;Xcomplete(i,:)];
        Y = [Y;Ycomplete(i)];
      end
    end
    
    %compute beta
    XTX = X'*X;
    %lam = 41;
    reg = lam*eye(rows(XTX));
    reg(1,1) = 0;
    beta = inv(XTX + reg)*X'*Y;
    
    %error
    Ypred = Xv*beta;
    err = Yv-Ypred;
    e += (err'*err)/rows(err);
  end
  e = e/K;
endfunction

toOptimize = @(lam) error4regularization(lam, data);
optimalLambda = fminsearch(toOptimize, 30)
err = error4regularization(optimalLambda, data)

%%compute beta
%%beta = inv(X'*X)*X'*Y;
%XTX = X'*X;
%lam = 0.01;
%reg = lam*eye(rows(XTX));
%beta = inv(XTX + reg)*X'*Y;
%
%%error
%Ypred = X*beta;
%err = Y-Ypred;
%e = err'*err
%
%
%%prepare for display
%[a b] = meshgrid(-3:0.1:3, -3:0.1:3);
%a_ = a(:);
%b_ = b(:);
%%xgrid with features
%Xgrid = [ones(length(a_),1), a_, b_, a_.*a_, a_.*b_, b_.*b_];
%Ygrid = Xgrid*beta;
%Ygrid = reshape(Ygrid, size(a));
%
%%plot
%scatter3(data(:,1), data(:,2), data(:,3));
%surface(a,b,Ygrid);
