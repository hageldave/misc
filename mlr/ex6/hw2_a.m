clear;
clc;

function z = gaussian(x,mean,cov, covinv)
  z =(1/sqrt(det(2*pi*cov)))*exp(-0.5*(x-mean)'*covinv*(x-mean));
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
    prior = priors(c);
    mean = means(:,c);
    cov = reshape(covs(:,c),[m,m]);
    covinv = reshape(covinvs(:,c),[m,m]);
    for i = 1:rows(X)
      temp = prior*gaussian(X(i,:)',mean,cov,covinv);
      sum = 0;
      for j = 1:k
        prior = priors(j);
        mean = means(:,j);
        cov = reshape(covs(:,c),[m,m]);
        covinv = reshape(covinvs(:,c),[m,m]);
        sum += prior*gaussian(X(i,:)',mean,cov,covinv);
      end
      P(i,k) = temp/sum;
    end
  end
end

gauss = @(x,mean,cov) gaussian(x,mean,cov,inv(cov));
n_gauss = @(x,mean,cov) n_gaussian(x,mean,cov,inv(cov));




X = load('mixture.txt');
n = rows(X);
m = columns(X);
k = 3;

priors = (ones(k,1)*(1/k))';
covariances = [eye(m)(:),eye(m)(:),eye(m)(:)];
rand_indices = randperm(n,k);
means = [];
for c = 1:k
  means = [means, X(rand_indices(c),:)'];
end

posteriors(X,means,covariances,priors);
