function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
        fe=size(X,2);
        sum=0;
        for i=1:m,
            hx=0; %compute htheta xi
            for k=1:fe,
             hx = hx+theta(k)*X(i,k);
            end;
            sum = sum+(hx-y(i))^2;
        end;
        J = (sum/(2*m));
        
        




% =========================================================================

end
