clear all;
clc;

function [result,evals,searches] = gradientdescent(f,df,x,tol)
  a = 1;
  incr_a = 1.2;
  decr_a = 0.5;
  ls = 0.01;
% -------------
  evals = [f(x)];
  searches = [];
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
    a = incr_a*a;
  until (norm(a*delta) < tol)
  result = x;
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

[r_sq,evals_sq,searches_sq] = gradientdescent(f_sq,df_sq,[1;1],0.001);
[r_hole,evals_hole,searches_hole] = gradientdescent(f_hole,df_hole,[1;1],0.001);


ax1 = subplot(2,2,1);
plot(ax1, 0:columns(evals_sq)-1,evals_sq);
title(ax1,'function value f_s_q');

ax2 = subplot(2,2,2);
plot(ax2, 0:columns(searches_sq)-1,searches_sq);
title(ax2,'line searches f_s_q');

ax3 = subplot(2,2,3);
plot(ax3, 0:columns(evals_hole)-1,evals_hole);
title(ax3,'function value f_h_o_l_e');

ax4 = subplot(2,2,4);
plot(ax4, 0:columns(searches_hole)-1,searches_hole);
title(ax4,'line searches f_h_o_l_e');
