clear;
clc;

files = dir("yalefaces");
X = [];
shape = 0;
for i = 3:rows(files)
  name = files(i).name;
  filename = strcat("yalefaces/",name);
  img = imread(filename);
  shape = size(img);
  X = [X; img(:)'];
end
X = double(X);
n = rows(X);

% compute mean
mu = X(1,:);
for i = 2:n
  mu += X(i,:);
end
mu = mu.*(1/n);

%show mean face
%imshow(uint8(reshape(mu, shape)))

%center data
Xc = X - ones(n,1)*mu;

%compute svd anfd get projection
[U, S, V] = svd(Xc, "econ");
p = 60;
V = V(:,1:p);
Z = Xc*V;
size(Z)