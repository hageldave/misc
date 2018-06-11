clear;
clc;

function z = gaussian(x,mean,cov, covinv)
  z =(1/sqrt((2*pi)*det(cov)))*exp(-0.5*(x-mean)'*covinv*(x-mean));
end

function z = gaussmix(x,means,covs,covinvs,priors)
  m = rows(means);
  z = 0;
  for c = 1:columns(priors)
    mean = means(:,c);
    cov = reshape(covs(:,c),[m,m]);
    covinv = reshape(covinvs(:,c),[m,m]);
    z += priors(c)*gaussian(x,mean,cov,covinv);
  end
end

function Z = gaussmixN(X,means,covs,priors)
  k = columns(priors);
  m = rows(means);
  covinvs = [];
  for c = 1:k
    cov = covs(:,c);
    cov = reshape(cov,[m,m]);
    covinv = inv(cov);
    covinvs = [covinvs,covinv(:)];
  end
  n = rows(X);
  Z = zeros(n,1);
  for i = 1:n
    Z(i) = gaussmix(X(i,:)',means,covs,covinvs,priors);
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
  for i = 1:n
    probs = [];
    for c = 1:k
      prior = priors(c);
      mean = means(:,c);
      cov = reshape(covs(:,c),[m,m]);
      covinv = reshape(covinvs(:,c),[m,m]);
      probs = [probs,prior*gaussian(X(i,:)',mean,cov,covinv)];
    end
    s = sum(probs);
    for c = 1:k
      P(i,c) = probs(c)/s;
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

liklhd_prev = 0;
liklhd_current = sum(log(1+gaussmixN(X,means,covariances,priors)));

iter = 0;
%for iter = 1:20
while (liklhd_current-liklhd_prev) > 0.01
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
  % update convergence criteria
  liklhd_prev = liklhd_current
  liklhd_current = sum(log(1+gaussmixN(X,means,covariances,priors)))
  iter+=1
end

%result
priors
means
covariances

% viz
gamma_ik = posteriors(X,means,covariances,priors);
classes = [];
for i = 1:n
  [mx,idx] = max(gamma_ik(i,:));
  classes = [classes; idx];
end

x = linspace(-4,3,100);
y = linspace(-3.5,1,100);
[x,y] = meshgrid(x,y);
shape = size(x);
Z = gaussmixN([x(:),y(:)],means,covariances,priors);
Z = reshape(Z,shape);
figure,hold on;
scatter(X(:,1),X(:,2),'white','x');
contourf(x,y,Z,'LineColor', 'none');

uClass = unique(classes);
pntColor = colorcube(length(uClass));
colors = gamma_ik*pntColor;
markers = ['x','o','+'];
figure,hold on;
for ind = 1:length(uClass)
    %scatter(X(:,1)(classes == uClass(ind)), X(:,2)(classes == uClass(ind)),[], pntColor(ind,:),'x')
    scatter(X(:,1)(classes == uClass(ind)), X(:,2)(classes == uClass(ind)),[], colors(classes == uClass(ind)),markers(uClass(ind)))
end
