clear;
clc;

% load data
data = load('data2Class.txt');

% decompose loaded data into X and Y
n = size(data, 1);
X = data(:,1:2);
Y = data(:,3);

linfeat = @(x) [ones(rows(x),1),x];
quadfeat = @(x) [ones(rows(x),1),x, x(:,1).*x(:,2), x(:,1).*x(:,1), x(:,2).*x(:,2)];
sinfeat = @(x) [ones(rows(x),1),x, sin(x(:,1)), cos(x(:,1)), sin(x(:,2)), cos(x(:,2))];
X = sinfeat(X);


m = columns(X);
lambda = 0.1;
sigmoid = @(x) 1 ./(1+exp(-x));
p = @(x,beta) sigmoid(x*beta);
grad = @(beta) X'*(p(X,beta)-Y) + 2*lambda*eye(m)*beta;
hess = @(beta) X'*diag(p(X,beta).*(1-p(X,beta)))*X;

% newton iteration
beta = zeros(m,1);
for i = 1:10
  beta = beta - inverse(hess(beta))*grad(beta);
end


[a b] = meshgrid(-3:0.1:3, -3:0.1:3);
a_ = a(:);
b_ = b(:);
ab = [a_,b_];
%xgrid with features
Xgrid = sinfeat(ab);
Ygrid = sigmoid(Xgrid*beta);
Ygrid = reshape(Ygrid, size(a));

%plot
hold on;
contour(a,b, Ygrid, [0.5 0.5]);
contour(a,b, Ygrid, [0.2 0.2]);
contour(a,b, Ygrid, [0.8 0.8]);
colors = zeros(n,3);
class0 = Y == 0;
class1 = Y == 1;
colors(class0, 3) = 1;
colors(class1, 1) = 1;
scatter(data(:,1), data(:,2),6,colors,'x');
%axis([-inf inf -inf inf -2 2])
%scatter3(data(:,1), data(:,2), data(:,3));
%surface(a,b,Ygrid);