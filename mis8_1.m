clear all;
clc;

function result = fsq(x,l)
result = transpose(x)*l*x;
endfunction

function result = fsq2D(x1,x2,l)
  z = ones(rows(x1),columns(x1));
  for i = 1:rows(x1)
    for j = 1:columns(x1)
      z(i,j) = fsq([x1(i,j);x2(i,j)],l);
    endfor
  endfor
  result = z;
endfunction

l=diag([0,1]);
domain = linspace(-2,2,100);
[x,y] = meshgrid(domain,domain);
z = fsq2D(x,y,l);
surf(domain,domain,z);