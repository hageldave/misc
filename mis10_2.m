clear all;
clc;

function [result,evals,searches,trajec] = gradientdescent(f,df,x,tol)
  a = 1;
  incr_a = 1.2;
  decr_a = 0.5;
  ls = 0.01;
% -------------
  evals = [f(x)];
  searches = [];
  trajec = [x];
% -------------
  do
    gr = df(x);
    delta = -gr/norm(gr);
    searches = [searches,0];% meta info
    while (f(x+a*delta) > f(x)+ls*transpose(gr)*(a*delta))
      evals = [evals,f(x+a*delta)];% meta info
      searches(columns(searches))++;% meta info
      a = decr_a*a;
    endwhile
    evals = [evals,f(x+a*delta)];% meta info
    
    x = x+a*delta;
    trajec = [trajec,x];% meta info
    a = incr_a*a;
  until (norm(a*delta) < tol)
  result = x;
endfunction

function [result,evals,searches,trajec] = newton(f,df,hf,x,tol)
  a = 1;
  incr_a = 1.2;
  decr_a = 0.5;
  ls = 0.01;
  incr_l = decr_l = 1;
% -------------
  evals = [f(x)];
  searches = [];
  trajec = [x];
% -------------
  do
    hess = hf(x);
    gr = df(x);
    l = 0.001;%max(eig(hess))*1.1;
    delta = - inv(hess+l*eye(rows(hess)))*gr;
    searches = [searches,0];% meta info
    while (f(x+a*delta) > f(x)+ls*transpose(gr)*(a*delta))
      evals = [evals,f(x+a*delta)];% meta info
      searches(columns(searches))++;% meta info
      a = decr_a*a;
    endwhile
    evals = [evals,f(x+a*delta)];% meta info
    
    x = x+a*delta;
    trajec = [trajec,x];% meta info
    a = incr_a*a;
  until (norm(a*delta) < tol)
  result = x;
endfunction

function [result,evals,searches,trajec] = gaussnewton(f,df,x,tol)
  hf = @(x) 2*transpose(df(x))*df(x);
  [result,evals,searches,trajec] = newton(f,df,hf,x,tol);
endfunction

function result = func2D(x1,x2,f)
  z = ones(rows(x1),columns(x1));
  for i = 1:rows(x1)
    for j = 1:columns(x1)
      z(i,j) = f([x1(i,j);x2(i,j)]);
    endfor
  endfor
  result = z;
endfunction

a=4;
c=3;

phi = @(x) [sin(a*x(1));sin(a*c*x(2));2*x(1);2*c*x(2)];
f = @(x) transpose(phi(x))*phi(x);
dphi = @(x) [cos(a*x(1))*a,0,2,0;0,cos(2*a*c*x(2))*2*a*c,0,2*c];
df = @(x) 2*dphi(x)*phi(x);

start = [rand();rand()];

gradientdescent(f,df,start,0.001);

[r_gd,evals_gd,searches_gd,trajec_gd] = gradientdescent(f,df,start,0.0001);
[r_gn,evals_gn,searches_gn,trajec_gn] = gaussnewton(f,df,start,0.0001);


ax = subplot(2,3,1);
plot(ax, 0:columns(evals_gd)-1,evals_gd);
title(ax,'function value grad.desc.');

ax = subplot(2,3,2);
plot(ax, 0:columns(searches_gd)-1,searches_gd);
title(ax,'line searches grad.desc.');

ax = subplot(2,3,4);
plot(ax, 0:columns(evals_gn)-1,evals_gn);
title(ax,'function value gaussnewt');

ax = subplot(2,3,5);
plot(ax, 0:columns(searches_gn)-1,searches_gn);
title(ax,'line searches gaussnewt');

ax = subplot(2,3,3);
domain = linspace(-1.1,1.1,100);
[x,y] = meshgrid(domain,domain);
z = func2D(x,y,f);
contour(ax, x,y,z, 30);
line(ax, trajec_gd(1:1,:),trajec_gd(2:2,:));
title(ax,'search trajectory grad.desc.');

ax = subplot(2,3,6);
domain = linspace(-1.1,1.1,100);
[x,y] = meshgrid(domain,domain);
z = func2D(x,y,f);
contour(ax, x,y,z, 30);
line(ax, trajec_gn(1:1,:),trajec_gn(2:2,:));
title(ax,'search trajectory gaussnewt');
