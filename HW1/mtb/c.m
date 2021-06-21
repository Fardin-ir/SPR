x1=[-6:0.01:+6];
x2=[-6:0.01:+6];
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

%c1
mu = [-3 3];
sigma = [1 0 ; 0 1];
y = mvnpdf(X,mu,sigma);
y = reshape(y,length(x2),length(x1));
S = mvnrnd(mu,sigma,500);
figure(1)

[M,f] = contour(X1,X2,y);
xlabel('x1')
ylabel('x2')
title('c1')
f.LineWidth = 2;
hold on
plot(S(:,1),S(:,2),'.');
hold off

%c2
mu = [0 0];
sigma = [1 0.99 ; 0.99 1];
y = mvnpdf(X,mu,sigma);
y = reshape(y,length(x2),length(x1));
S = mvnrnd(mu,sigma,500);
figure(2)

[M,f] = contour(X1,X2,y);
xlabel('x1')
ylabel('x2')
title('c2')
f.LineWidth = 2;
hold on
plot(S(:,1),S(:,2),'.');
hold off
%c3
mu = [3 -3];
sigma = [1 0 ; 0 0.3];
y = mvnpdf(X,mu,sigma);
y = reshape(y,length(x2),length(x1));
S = mvnrnd(mu,sigma,500);
figure(3)

[M,f] = contour(X1,X2,y);
xlabel('x1')
ylabel('x2')
title('c3')
f.LineWidth = 2;
hold on
plot(S(:,1),S(:,2),'.');
hold off

