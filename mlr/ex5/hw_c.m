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

% compute mean
mu = X(1,:);
for i = 2: rows(X)
  mu += X(i,:);
end
mu = mu.*(1/rows(X));

%show mean face
%imshow(uint8(reshape(mu, shape)))

[U, S, V] = svd(X, "econ");
size(V)
size(X)