function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%size(X) 12x2 1st colum-bias term
%size(y)  12x1
%size(theta) 2x1
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%%(1/2m)square(log(h(x)))-y) +(lambda/2m)sum(square(theta))

sig=theta'*X'; %1x2 x 2 x12 -- 1x12
sig=sig'; %12x1 vector
%size(sig) 
error=(sig-y);
sq_error=power(error,2);
summation=sum(sq_error);
summation=(1/(2*m))*summation;
%lambda/2m sum(square of theta j=1:n)
%overfitting term

theta_square=power(theta,2);
sumtheta_square=sum(theta_square)-theta_square(1);
regterm=(lambda/(2*m))*sumtheta_square;
%fprintf('cost function\n')
%J
J=summation+ regterm;

x_i=X(:,1);
grad(1)=sum(error.*x_i)/m;

for i=2:size(theta)
x_i=zeros(m);
x_i=X(:,i);
grad(i)=(sum(error.*x_i)/m)+((lambda/m)*theta(i));
endfor














% =========================================================================

grad = grad(:);

end
