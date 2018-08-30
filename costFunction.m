function [J, grad] = costFunction(theta, X, y)


% Initialize some useful values
m = length(y); % number of training examples

z=X*theta;
h=sigmoid(z);
l=(-y)'*log(h)-(1-y)'*log(1-h);
J=(1/m)*sum(l);		% sum will only work for a column, will not sum over a row

grad=1/m*((X'*h-X'*y)');

end
