clc;
data = csvread('BreastCancerData.csv');

x1 = data(1:140,6);
x2 = data(1:140,7);
x3 = data(1:140,3);
y = data(1:140, 9);
m = length(y);

x1_mean = mean(x1);
x1_std = std(x1);

x2_mean = mean(x2);
x2_std = std(x2);

x3_mean = mean(x3);
x3_std = std(x3);

x = [ones(m,1), ((x1-x1_mean)/x1_std), ((x2-x2_mean)/x2_std), ((x3-x3_mean)/x3_std)];
theta_val = ones(4, 1);

max_Iterations = 700;
alpha = 0.5;

costFunctionCalculation(x,y,m,theta_val);

theta_val = gradientDescentCalculation(x, y,m,alpha, theta_val, max_Iterations, true, true);

fprintf('theta values after gradient decent calculation:\n');
fprintf('theta0 = %f \t theta1 = %f \t theta2 = %f\t theta3 = %f \n', theta_val(1),theta_val(2),theta_val(3),theta_val(4));

v = costFunctionCalculation(x,y,m,theta_val);
fprintf('Final cost for training data is %f\n',v);

%plot graph for linear regression with two variables on training data

figure(3)
plot(x(:,2), y, x(:,2), x*theta_val, '-r');
xlabel('Perimeter');
ylabel('Compactness');
legend('Training Data','Linear Regression with three variables');
tit = sprintf('Linear Regression with three variables for training data set with learning data %f',alpha);
title(tit)
hold off

figure(4)
plot(x(:,3), y, x(:,3), x*theta_val, '-r');
xlabel('Area');
ylabel('Compactness');
legend('Training Data','Linear Regression with three variables');
tit = sprintf('Linear Regression with three variables for training data set with learning data %f',alpha);
title(tit)
hold off

figure(5)
plot(x(:,4), y, x(:,4), x*theta_val, '-r');
xlabel('Radius');
ylabel('Compactness');
legend('Training Data','Linear Regression with three variables');
tit = sprintf('Linear Regression with three variables for training data set with learning data %f',alpha);
title(tit)
hold off




% Testing

x1_test = data(141:end,6);
y_test = data(141:end,9);
x2_test = data(141:end,7);
x3_test = data(141:end,3);
m1 = length(y_test);
x1_test_mean = mean(x1_test);
x1_test_std = std(x1_test);
x1_td = (x1_test-x1_test_mean)/x1_test_std;
n = size(x1_td,1);

x2_test_mean = mean(x2_test);
x2_test_std = std(x2_test);
x2_td = (x2_test-x2_test_mean)/x2_test_std;

x3_test_mean = mean(x3_test);
x3_test_std = std(x3_test);
x3_td = (x3_test-x3_test_mean)/x3_test_std;

pred = [ones(n,1), x1_td, x2_td, x3_td] * theta_val;
figure(6)
plot(x1_td, y_test, x1_td, pred,'-O');
xlabel('Perimeter');
ylabel('Compactness');
legend('Testing Data','Linear Regression with variable x1(perimeter)');
tit = sprintf('Linear Regression with three variables for testing data set with learning data %f',alpha);
title(tit)
hold off

figure(7)
plot(x2_td, y_test, x2_td, pred,'-O');
xlabel('Area');
ylabel('Compactness');
legend('Testing Data','Linear Regression with variable x2(Area)');
tit = sprintf('Linear Regression with three variables for testing data set with learning data %f',alpha);
title(tit)
hold off

figure(8)
plot(x3_td, y_test, x3_td, pred,'-O');
xlabel('Radius');
ylabel('Compactness');
legend('Testing Data','Linear Regression with variable x2(Area)');
tit = sprintf('Linear Regression with three variables for testing data set with learning data %f',alpha);
title(tit)
hold off


x_t = [ones(m1,1),x1_td,x2_td,x3_td];
v = costFunctionCalculation(x_t,y_test,m1,theta_val);
fprintf('Final cost for testing data is %f\n',v);
