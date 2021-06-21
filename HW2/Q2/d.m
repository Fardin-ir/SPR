mu1 = [1 2];
sigma1 = [1.8 -0.7; -0.7 1.8];
mu2= [-1 -3];
sigma2 = [1.5 0.3;0.3 1.5];

S1 = [mvnrnd(mu1,sigma1,1000) ones(1000,1)];
S2 = [mvnrnd(mu2,sigma2,1000) zeros(1000,1)];
S = [S1; S2];

x1=[-6:0.01:+6];
x2=[-6:0.01:+6];
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

y1 = mvnpdf(X,mu1,sigma1);
y1 = reshape(y1,length(x2),length(x1));

y2 = mvnpdf(X,mu2,sigma2);
y2 = reshape(y2,length(x2),length(x1));

figure(10)
[M,c1] = contour(X1,X2,y1);
c1.LineWidth = 1;

hold on
%plot(S(:,1),S(:,2),'.');
[M,c2] = contour(X1,X2,y2);
c2.LineWidth = 1;

x = transpose(sym('x',[1,2]));
mu1=transpose(mu1);
W12 = (-1/2)*inv(sigma1);
W11 = sigma1\mu1;
W10 = (-1/2)*transpose(mu1)*inv(sigma1)*mu1+(-1/2)*log(det(sigma1))+log(1/2);
mu2=transpose(mu2);
W22 = (-1/2)*inv(sigma2);
W21 = sigma2\mu2;
W20 = (-1/2)*transpose(mu2)*inv(sigma2)*mu2+(-1/2)*log(det(sigma2))+log(1/2);

f1=transpose(x)*W12*x+transpose(W11)*x+W10;
f2=transpose(x)*W22*x+transpose(W21)*x+W20;
f=f1-f2;
f1=transpose(x)*W12*x+transpose(W11)*x+W10+log(3);
f2=transpose(x)*W22*x+transpose(W21)*x+W20+log(1);
f_c=f1-f2;

fimplicit(f1-f2, [-6 6 -6 6]);
hold off

pred_B = zeros(2000,1);
pred_B_c = zeros(2000,1);
for i = 1:2000
   syms x1 x2;
   pred_B(i,1) = heaviside(subs(f_c,[x1,x2],S(i,1:2)));
   pred_B_c(i,1) = heaviside(subs(f,[x1,x2],S(i,1:2)));
end

