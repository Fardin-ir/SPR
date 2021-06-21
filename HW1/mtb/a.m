n = 1;
N = 500;
mu = 5;
sigma1 = 1;
sigma2 = 2;
sigma3 = 3;


X1 = mvnrnd(mu,sigma1,500);
X2 = mvnrnd(mu,sigma2,500);
X3 = mvnrnd(mu,sigma3,500);


figure(1)
subplot(3,1,1)
h1 = histogram(X1,'FaceAlpha',0.5, 'FaceColor','r','edgecolor','k','BinWidth',0.4);
xlabel('x')
ylabel('Frequency')
axis([-1,+11,0,100])
title("sigma = 1")
subplot(3,1,2)
h2 = histogram(X2,'FaceAlpha',0.5, 'FaceColor','g','edgecolor','k','BinWidth',0.4);
xlabel('x')
ylabel('Frequency')
axis([-1,+11,0,100])
title("sigma = 2")
subplot(3,1,3)
h3 = histogram(X3,'FaceAlpha',0.5, 'FaceColor','b','edgecolor','k','BinWidth',0.4);
xlabel('x')
ylabel('Frequency')
axis([-1,+11,0,100])
title("sigma = 3")

figure(2)
h1 = histfit(X1);
delete(h1(1))
h1(2).Color = 'r';
hold on
h2 = histfit(X2);
delete(h2(1))
h2(2).Color = 'g';
h3 = histfit(X3);
delete(h3(1))
h3(2).Color = 'b';
hold off

figure(3)
subplot(3,1,1)
plot(X1,zeros(500),'xr','MarkerSize',2)
xlabel('x')
title("sigma = 1")
axis([-1,+11,-1,1])
subplot(3,1,2)
plot(X1,zeros(500),'xb','MarkerSize',2)
xlabel('x')
title("sigma = 2")
axis([-1,+11,-1,1])
subplot(3,1,3)
plot(X1,zeros(500),'xg','MarkerSize',2)
xlabel('x')
title("sigma = 3")
axis([-1,+11,-1,1])
