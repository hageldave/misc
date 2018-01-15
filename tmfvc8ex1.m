clear;
pkg load odepkg % <- tell octave to load ode package


function [tout,xout] = pathline(func, tspan, x0)
[tout,xout] = ode45(func, tspan, x0);
end

function [tout,xout] = streamline(func, tspan, x0, t0)
[tout,xout] = ode45(@(t,x) func(t0,x), tspan, x0);
% have to change tout to be all t0
tout = repmat(t0,rows(tout),1);
end

function lines = streamlines(func, tspan, x0, tRes)
lines = cell(1,tRes);
i=1;
for t=linspace(tspan(1),tspan(2),tRes)
[t,x] = streamline(func,tspan,x0, t);
lines{i++} = [t,x];
end
end

function [tout,xout] = streakline(func, tspan, x0, tRes)
tout = [];
xout = [];
for t = linspace(tspan(1),tspan(2),tRes)
[ts,xs] = pathline(func, [t,tspan(2)], x0, 10);
tout = [tout,t];
xout = [xout;xs(rows(xs),:)];
end
fliplr(tout);
end

f = @(t,y) [sin(t) ; 1]

x1=1;
x2=1;
[ts,ys] = pathline(f,[0,20],[x1;x2]);
[ts2,ys2] = streakline(f,[0,20],[x1;x2], 100);
%pathline
figure
ax1 = subplot(1,2,1);
hold on;
plot3(ax1,ts,ys(:,1),ys(:,2));
p = plot3(ax1,ts2,ys2(:,1),ys2(:,2));
set(p, 'Color', [1 0 0])
xlabel('t'); % x-axis label
ylabel('x'); % y-axis label
zlabel("y");
hold off;

%streamline
lines = streamlines(f,[0,20],[x1;x2], 100);
ax2 = subplot(1,2,2);
hold on;
for i=1:length(lines)
line = lines{i};
p = plot3(ax2,line(:,1),line(:,2),line(:,3));
% set different color for each streamline
set(p,'Color',[(1-(i/length(lines))) 0.6 (i/length(lines))]);
ax2 = subplot(1,2,2);
end
xlabel('t'); % x-axis label
ylabel('x'); % y-axis label
zlabel("y");
hold off;

%random vllt noch sinnvoler shit


%xsamples = -3:.5:3
%ysample = -3:.5:3
%n1=length(xsamples);
%n2=length(ysample);
%yp1=zeros(n2,n1);
%yp2=zeros(n2,n1);


%y1val = xsamples
%y2val = ysample
%for i=1:n1
 % for j=1:n2
  %  ypv = feval(f,t,[y1val(i);y2val(j)]);
   % yp1(j,i) = ypv(1);
    %yp2(j,i) = ypv(2);
  %end
%end
%q = quiver(y1val,y2val,yp1,yp2,'r');
%c = q.Color;
%q.Color = 'blue';

%hold on;
%plot(ys(:,1),ys(:,2))
%hold off;