clear;
clc;

function z = gaussian(x,mean,cov, covinv)
  z =(1/sqrt((2*pi)^3*det(cov)))*exp(-0.5*(x-mean)'*covinv*(x-mean));
end

function z = n_gaussian(x,mean,cov, covinv)
  cols = columns(x);
  z = zeros(1,cols);
  for i = 1:cols
    z(i) = gaussian(x(:,i),mean,cov,covinv);
  end
end

function P = posteriors(X,means,covs, priors)
  k = columns(priors);
  n = rows(X);
  m = columns(X);
  P = zeros(n,k);
  covinvs = [];
  for c = 1:k
    cov = covs(:,c);
    cov = reshape(cov,[m,m]);
    covinv = inv(cov);
    covinvs = [covinvs,covinv(:)];
  end
  for c = 1:k
    prior = priors(:,c);
    mean = means(:,c);
    cov = reshape(covs(:,c),[m,m]);
    covinv = reshape(covinvs(:,c),[m,m]);
    for i = 1:n
      temp = prior*gaussian(X(i,:)',mean,cov,covinv);
      sum = 0;
      for j = 1:k
        prior = priors(j);
        mean = means(:,j);
        cov = reshape(covs(:,j),[m,m]);
        covinv = reshape(covinvs(:,j),[m,m]);
        sum += prior*gaussian(X(i,:)',mean,cov,covinv);
      end
      P(i,c) = temp/sum;
    end
  end
end

gauss = @(x,mean,cov) gaussian(x,mean,cov,inv(cov));
n_gauss = @(x,mean,cov) n_gaussian(x,mean,cov,inv(cov));




X = load('mixture.txt');
n = rows(X);
m = columns(X);
k = 3;

priors = (ones(1,k)*(1/k));
rand_indices = randperm(n,k);
means = [];
covariances = [];
for c = 1:k
  means = [means, X(rand_indices(c),:)'];
  %means = [means, rand(2,1)*5];
  covariances = [covariances, eye(m,m)(:)];
end

for iter = 1:4
  gamma_ik = posteriors(X,means,covariances,priors);
  for c = 1:k
    nk = sum(gamma_ik(:,c));
    % update prior
    priors(:,c) = nk/n;
    % update mean
    mean = zeros(m,1);
    for i = 1:n
      mean += gamma_ik(i,c)*X(i,:)';
    end
    mean *= 1/nk;
    means(:,c) = mean;
    % update covariance
    cov = zeros(m,m);
    for i = 1:n
      diff = X(i,:)' - mean;
      cov += gamma_ik(i,c)*diff*diff';
    end
    cov *= 1/nk;
    covariances(:,c) = cov(:);
  end
end
priors
means
covariances

%scatter(X(:,1),X(:,2));
