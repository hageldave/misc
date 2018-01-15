clear all;

n = 4
x = randn(n,1)
A = randn(3,n)

function result = gradientcheck (xin, fin, dfin)
eps = 10e-6;
n = rows(xin);
d = rows(fin(xin));
basis = eye(n);
J = zeros(d,n);
for i = 1:n
 b = basis(:,i)*eps;
 x1 = xin+b;
 x2 = xin-b;
 J(:,i) = (fin(x1) - fin(x2))/(2*eps);
endfor
result = norm(J-dfin(xin),Inf) < 10e-4;
endfunction


f = @(x) A*x
df = @(x) A
gradientcheck(x, f, df)



f = @(x) transpose(x)*x
df = @(x) 2*transpose(x)
gradientcheck(x, f, df)
