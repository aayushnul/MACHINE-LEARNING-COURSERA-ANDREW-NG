function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%lambda part
fe=length(theta);
sumJ=0;
for i=1:m
     g=sigmoid(theta'*(X(i,1:fe))');
     sumJ=sumJ+(-y(i)*log(g)-(1-y(i))*log(1-g));
end
sum=0;
for j=2:fe %no regularization for theta1
    sum=sum+(theta(j))^2;
end
sum=(sum*lambda)/2;
J=(sumJ+sum)/m;
for j=1:fe
     for i=1:m
        g=sigmoid(theta'*(X(i,1:fe))');
        grad(j)=grad(j)+(g-y(i))*X(i,j);
     end
    if (j==1)   %no regularization
     grad(j)=grad(j)/m;
    else
    grad(j)=(grad(j)+lambda*theta(j))/m;
end








% =============================================================

end
