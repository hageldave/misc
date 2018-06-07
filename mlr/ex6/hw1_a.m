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

%K-MEANS
%%%%%%%%%

k=4;
numruns = 10;
errors = [];
allclusterfaces = [];
for run = 1:numruns
  %init cluster centers
  rand_indices = randperm(rows(X),k);
  centers = [];
  for i = 1:k
    centers = [centers;X(rand_indices(i),:)];
  end
  
  iter = 15;
  while(iter > 0)
    %compute for each image which cluster center is nearest
    %also take the opportunity and calculate the error from minimal distances
    whichcluster = zeros(rows(X),1);
    err = 0;
    for i = 1: rows(X)
      mindist=norm(X(i,:)-centers(1,:));
      whichcluster(i) = 1;
      for c = 2:k
        dist=norm(X(i,:)-centers(c,:));
        if dist < mindist
          mindist = dist;
          whichcluster(i)=c;
        end
      end
      err += mindist^2;
    end
    err /= rows(X);
    iter-=1;
    if iter == 0
      break;
    end
    %compute new cluster centers according to assignment
    for c = 1:k
      nummembers = 0;
      newcenter = zeros(1,columns(X));
      for i = 1:rows(X)
        if whichcluster(i) == c
          newcenter += X(i,:);
          nummembers += 1;
        end
      end
      newcenter=newcenter.*(1.0/nummembers);
      centers(c,:)=newcenter;
    end
  end
  
  % save cluster centers
  clusterfaces = [uint8(reshape(centers(1,:), shape))];
  for c = 2:k
    clusterfaces = [clusterfaces, uint8(reshape(centers(c,:), shape))];
  end
  errors = [errors;err];
  allclusterfaces = [allclusterfaces;clusterfaces];
end
% show best cluster centers images
[minerr,idx] = min(errors)
bestcluster = allclusterfaces((idx-1)*shape(1)+1:(idx)*shape(1),:);
imshow(bestcluster);
