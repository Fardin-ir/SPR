mu1 = [1 2];
sigma1 = [1.8 -0.7; -0.7 1.8];
mu2= [-1 -3];
sigma2 = [1.5 0.3;0.3 1.5];

S1 = [mvnrnd(mu1,sigma1,1000) ones(1000,1)];
S2 = [mvnrnd(mu2,sigma2,1000) zeros(1000,1)];
S = [S1; S2];
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
f = f1-f2;
mdc = transpose((mu1-mu2))*x-1/2*(transpose(mu1)*mu1-transpose(mu2)*mu2);

num_true_B = 0;
num_true_mdc = 0;
for i = 1:2000
   syms x1 x2;
   o1 = heaviside(subs(mdc,[x1,x2],S(i,1:2)));
   o2 = heaviside(subs(f,[x1,x2],S(i,1:2)));
   if o1 == S(i,3)
       num_true_B =num_true_B + 1;
   end
   if o2 == S(i,3)
       num_true_mdc =num_true_mdc + 1;
   end
end
    
1-num_true_B/2000
1-num_true_mdc/2000
