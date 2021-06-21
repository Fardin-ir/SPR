mu = [2 1];
sigma = [2 1; 1 3];

S = mvnrnd(mu,sigma,500);

x1=[-6:0.01:+6];
x2=[-6:0.01:+6];
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

y = mvnpdf(X,mu,sigma);
y = reshape(y,length(x2),length(x1));

figure(10)
[M,c] = contour(X1,X2,y);
c.LineWidth = 2;

hold on
plot(S(:,1),S(:,2),'.');
hold off
%f
sigma_h = cov(S)
mu_h = mean(S)
%g
[V,D]=eig(sigma)

inv(V)*sigma*V

inv(V)*sigma_h*V

%h
N = S*inv(sqrtm(sigma_h));

figure(3)
plot(N(:,1),N(:,2),'.');
part_h_mean = mean(N)
part_h_cov = cov(N)

%i
[V,D]=eig(sigma)
figure(4)
plotv(V,'-')

%j
[V,D]=eig(sigma_h)
temp_v = V(:,1);
V(:,1)=V(:,2);
V(:,2)=temp_v;
projected_S = (S-mu)*V;
figure(5)
plot(projected_S(:,1),projected_S(:,2),'.');
cov(projected_S);
mean(projected_S);

%k
cov_n = cov(N)
[V_N,D_N]=eig(cov(N))
