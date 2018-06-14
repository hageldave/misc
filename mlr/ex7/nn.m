clear;
clc;
more off;

sigmoid = @(x) 1 ./(1+exp(-x));

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

gradients = cell(1,numlayers);
for i = 1:numlayers
  % init gradients to zero
  gradients{i} = zeros(l(i+1),l(i));
end

for i = 1:n
  xes = cell(1,numlayers);
  % forward
  x = X(i,:)';
  z = 0;
  for lay = 1:numlayers
    xes{lay} = x;
    z = layers{lay}*x;
    x = sigmoid(z);
  end
  f=z;
  loss = max(0, 1-f*Y(i));
  delta = -Y(i)*[1-f*Y(i) > 0];
  % backward
  for lay = numlayers:-1:1
    x = xes{lay};
    gradient = delta'*x';
    delta = (delta*layers{lay}).*(x.*(1-x))';
    gradients{lay} += gradient;
  end
end



