function [theta_val, cost_val] = gradientDescentCalculation(x,y,m,alpha,theta_val,max_Iterations,isMultivarient,isExtraVariable)

cost_val = zeros(max_Iterations, 1);

 for i = 1: max_Iterations
    x0 = x(:, 1);
    x1 = x(:, 2);

    theta_val(1) = theta_val(1)-(alpha * (1/m) * (((x*theta_val) - y)' * x0));
    theta_val(2) = theta_val(2)-(alpha * (1/m) * (((x*theta_val) - y)' * x1));

     if(isMultivarient)
        x2 = x(:,3);
        theta_val(3) = theta_val(3)-((alpha/m)*((x * theta_val) - y)' * x2);
        temp1(i) = theta_val(3);
     end
     
     if(isExtraVariable)
         x3 = x(:,4);
         theta_val(4) = theta_val(4)-((alpha/m)*((x * theta_val) - y)' * x3);
         temp2(i) = theta_val(4);
     end    
     
     cost_val(i) = costFunctionCalculation(x,y,m,theta_val);
     temp(i) = theta_val(2);
     temp3(i) = theta_val(1);
 end
  
 
 figure(1);
 plot(1:max_Iterations, cost_val, '-');
 xlabel('Number of Iterations');
 ylabel('Cost Function');
 tit = sprintf('Cost function when learning rate %f',alpha);
 title(tit)
 hold off
 
 figure(9)
 plot(temp3, cost_val, '-');
 xlabel('theta0');
 ylabel('Cost Function');
 tit = sprintf('theta0 vs cost function');
 title(tit)
 hold off
 
 figure(10)
 plot(temp, cost_val, '-');
 xlabel('theta1');
 ylabel('Cost Function');
 tit = sprintf('theta1 vs cost function');
 title(tit)
 hold off
 
 if(isMultivarient)
     figure(11)
     plot(temp1, cost_val, '-');
     xlabel('theta2');
     ylabel('Cost Function');
     tit = sprintf('theta2 vs cost function');
     title(tit)
     hold off
 end 
 
 if(isExtraVariable)
     figure(12)
     plot(temp2, cost_val, '-');
     xlabel('theta3');
     ylabel('Cost Function');
     tit = sprintf('theta3 vs cost function');
     title(tit)
     hold off
 end 
 
end

