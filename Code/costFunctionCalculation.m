function cost = costFunctionCalculation( x,y,m,theta_val)
         %Cost function for linear regression.
         %costval = costFunctionCalculation( x,y,theta_val) that finds the theta values that fits the given data.  

cost = 0;
summation = sum(((x * theta_val)-y).^2); % calculating the sum of square errors
cost = (1/(2*m)) * summation;   % calculating the theta value.
end

