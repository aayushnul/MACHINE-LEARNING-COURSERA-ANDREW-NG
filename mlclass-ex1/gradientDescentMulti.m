function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
fe = size(X,2);
tmp=zeros(size(theta));
for iter = 1:num_iters,
    for j=1:fe,
        sum=0;
        for i=1:m,
            hx=0; %compute htheta xi
            for k=1:fe,
             hx = hx+theta(k)*X(i,k);
            end;
            sum = sum+(hx-y(i))*X(i,j);
        end;
        tmp(j)=theta(j)-alpha*(sum/m);
    end;
    for j=1:fe, %update
        theta(j) = tmp(j);
    end;

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
