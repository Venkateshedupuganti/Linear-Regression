clc;
data = csvread('BreastCancerData.csv');

x= data(1:180,6);
y = data(1:180, 9);
m = length(y);
x_mean = mean(x);
x_std = std(x);

x = [ones(m,1), ((x-x_mean)/x_std), ((x-x_mean)/x_std).^2];
theta_val = ones(3, 1);

max_Iterations = 700;
alpha = 0.007;

costFunctionCalculation(x,y,m,theta_val);

theta_val = gradientDescentCalculation(x, y,m, alpha, theta_val, max_Iterations, true, false);

fprintf('theta values after gradient decent calculation:\n');
fprintf('theta0 = %f \t theta1 = %f \t theta2 = %f \n', theta_val(1),theta_val(2),theta_val(3));


v = costFunctionCalculation(x,y,m,theta_val);
fprintf('Final cost for training data is %f\n',v);

%plot graph for linear regression on training data

figure(3)
plot(x(:,2), y, x(:,2), x*theta_val, '-r');
xlabel('Perimeter');
ylabel('Compactness');
legend('Training Data','Linear Regression');
tit = sprintf('Quadratic Regression for training data set with learning data %f',alpha);
title(tit)
hold off



% Testing

x_test = data(181:end,6);
y_test = data(181:end,9);
m1 = length(y_test);
x_test_mean = mean(x_test);
x_test_std = std(x_test);

x_td = (x_test-x_test_mean)/(x_test_std);
n = size(x_td,1);
pred = [ones(n,1), x_td, x_td.^2] * theta_val;
figure(4)
plot(x_td, y_test, x_td, pred, '-r');
xlabel('Perimeter');
ylabel('Compactness');
legend('Testing Data','Quadratic Regression');
tit = sprintf('Quadratic Regression for testing data set with learning data %f',alpha);
title(tit)
hold off

x_t = [ones(m1,1),x_td,x_td.^2];
v = costFunctionCalculation(x_t,y_test,m1,theta_val);
fprintf('Final cost for testing data is %f\n',v);

