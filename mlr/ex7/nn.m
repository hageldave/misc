clear;
clc;
more off;

sigmoid = @(x) 1 ./(1+exp(-x));
lossfn = @(pred,ytrue) max(0, 1-pred.*ytrue);

data = load('data2Class_adjusted.txt');
X = data(:,1:3);
Y = data(:,4);

n = rows(X);
m = columns(X);

l = [m,100,1];
numlayers = columns(l)-1;
layers = cell(1,numlayers);
for i = 1:numlayers
  % random matrix of dims li x li-1
  layers{i} = rand(l(i+1),l(i))*2-1;
end

function [f,xes] = fwd(x, layers, xes, activationFN)
  numlayers = columns(layers);
  z=0;
  for lay = 1:numlayers
    % keep track of x input per layer
    xes{lay} = x;
    % compute z then compute x
    z = layers{lay}*x;
    x = activationFN(z);
  end
  f=z;
end

function gradients = bwd(delta,layers,xes,gradients)
  numlayers = columns(layers);
  for lay = numlayers:-1:1 % backwards loop
    x = xes{lay};
    gradient = delta'*x';
    delta = (delta*layers{lay}).*(x.*(1-x))';
    gradients{lay} += gradient;
  end
end

gradients = cell(1,numlayers);

total_loss = 10;
iter = 0;
while total_loss > 0.1
  for i = 1:numlayers
    % init gradients to zero
    gradients{i} = zeros(l(i+1),l(i));
  end
  losses = [];
  for i = 1:n
    % forward
    xes = cell(1,numlayers);
    x = X(i,:)';
    [f,xes] = fwd(x,layers,xes, sigmoid);
    % compute loss
    loss = lossfn(f,Y(i));
    losses = [losses;loss];
    delta = -Y(i)*[1-f*Y(i) > 0];
    % backward
    gradients = bwd(delta,layers,xes,gradients);
  end
  % gradient descent step
  for lay = 1:numlayers
    layers{lay} = layers{lay}-0.05*gradients{lay};
  end
  % calc loss
  total_loss = sum(losses)/n
  iter+=1
end

% viz
space = linspace(-4,4,100);
[x2,x3] = meshgrid(space,space);
x1 = ones(size(x2));
Xgrid = [x1(:),x2(:),x3(:)];
F = zeros(rows(Xgrid),1);
for i = 1:rows(Xgrid)
  x = Xgrid(i,:)';
  f = fwd(x,layers,cell(1,numlayers),sigmoid);
  F(i) = sigmoid(f);
end
F = reshape(F,size(x2));
hold on;
scatter(X(:,2)(Y==1),X(:,3)(Y==1), 'x');
scatter(X(:,2)(Y==-1),X(:,3)(Y==-1), 'o');
contour(x2,x3,F, [0.5,0.5]);
contour(x2,x3,F, [0.1,0.9], '--');
