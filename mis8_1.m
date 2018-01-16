clear all;
clc;

function result = fsq(x,l)
result = transpose(x)*l*x;
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

l=diag([1,10]);
domain = linspace(-2,2,100);
[x,y] = meshgrid(domain,domain);
z = func2D(x,y,@(x)fsq(x,l));
surf(domain,domain,z);