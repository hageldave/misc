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
p = 20;
Vp = V(:,1:p);
Z = Xc*Vp;

%K-MEANS
%%%%%%%%%

lensquared = @(x) x*x';

allk = [];
k2err = [];

for k=4:10
  numruns = 10;
  errors = [];
  allclusterfaces = [];
  for run = 1:numruns
    %init cluster centers
    rand_indices = randperm(n,k);
    centers = [];
    for i = 1:k
      centers = [centers;Z(rand_indices(i),:)];
    end
    
    iter = 15;
    while(iter > 0)
      %compute for each image which cluster center is nearest
      %also take the opportunity and calculate the error from minimal distances
      whichcluster = zeros(n,1);
      err = 0;
      % reconstruct centers in original domain (due to PCA) for error measurement
      reconstCenters = ones(k,1)*mu + centers*Vp';
      for i = 1:n
        mindist=lensquared(Z(i,:)-centers(1,:));
        whichcluster(i) = 1;
        for c = 2:k
          dist=lensquared(Z(i,:)-centers(c,:));
          if dist < mindist
            mindist = dist;
            whichcluster(i)=c;
          end
        end
        err += lensquared(X(i,:)-reconstCenters(whichcluster(i),:));%mindist;
      end
      err /= n;
      iter-=1;
      if iter == 0
        break;
      end
      %compute new cluster centers according to assignment
      for c = 1:k
        nummembers = 0;
        newcenter = zeros(1,columns(Z));
        for i = 1:n
          if whichcluster(i) == c
            newcenter += Z(i,:);
            nummembers += 1;
          end
        end
        if nummembers > 0
          newcenter=newcenter.*(1.0/nummembers);
          centers(c,:)=newcenter;
        end
      end
    end
    
    % save cluster centers
    %clusterfaces = [uint8(reshape(centers(1,:), shape))];
    %for c = 2:k
    %  clusterfaces = [clusterfaces, uint8(reshape(centers(c,:), shape))];
    %end
    errors = [errors;err];
    %allclusterfaces = [allclusterfaces;clusterfaces];
  end
  % show best cluster centers images
  [minerr,idx] = min(errors);
  allk = [allk,k];
  k2err = [k2err,minerr];
  k
  minerr
  fflush(stdout);
  %bestcluster = allclusterfaces((idx-1)*shape(1)+1:(idx)*shape(1),:);
  %imshow(bestcluster);
end


plot(allk,k2err);
