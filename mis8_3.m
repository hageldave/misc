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

function result = func2D(x1,x2,f)
  z = ones(rows(x1),columns(x1));
  for i = 1:rows(x1)
    for j = 1:columns(x1)
      z(i,j) = f([x1(i,j);x2(i,j)]);
    endfor
  endfor
  result = z;
endfunction


f = @(x) transpose(x)*x;
df = @(x) 2*x;

gradientdescent(f,df,10,0.001);

cond = @(c,i,dim) c^((i-1)/(dim-1));
C = diag([cond(10,1,2),cond(10,2,2)]);
f_sq = @(x) transpose(x)*C*x;
df_sq = @(x) 2*C*x;
f_hole = @(x) 1 - exp(-f_sq(x));
df_hole = @(x) -exp(-f_sq(x))*-df_sq(x);

[r_sq,evals_sq,searches_sq,trajec_sq] = gradientdescent(f_sq,df_sq,[1;1],0.001);
[r_hole,evals_hole,searches_hole,trajec_hole] = gradientdescent(f_hole,df_hole,[1;1],0.001);


ax = subplot(2,3,1);
plot(ax, 0:columns(evals_sq)-1,evals_sq);
title(ax,'function value f_s_q');

ax = subplot(2,3,2);
plot(ax, 0:columns(searches_sq)-1,searches_sq);
title(ax,'line searches f_s_q');

ax = subplot(2,3,4);
plot(ax, 0:columns(evals_hole)-1,evals_hole);
title(ax,'function value f_h_o_l_e');

ax = subplot(2,3,5);
plot(ax, 0:columns(searches_hole)-1,searches_hole);
title(ax,'line searches f_h_o_l_e');

ax = subplot(2,3,3);
domain = linspace(-1.1,1.1,100);
[x,y] = meshgrid(domain,domain);
z = func2D(x,y,f_sq);
contour(ax, x,y,z, 30);
line(ax, trajec_sq(1:1,:),trajec_sq(2:2,:));
title(ax,'search trajectory f_s_q');

ax = subplot(2,3,6);
domain = linspace(-1.1,1.1,100);
[x,y] = meshgrid(domain,domain);
z = func2D(x,y,f_hole);
contour(ax, x,y,z);
line(ax, trajec_hole(1:1,:),trajec_hole(2:2,:));
title(ax,'search trajectory f_h_o_l_e');
