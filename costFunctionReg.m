function [J, grad] = costFunctionReg(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples

z=X*theta;
h=sigmoid(z);

J = ( (1 / m) * sum(-y'*log(sigmoid(X*theta)) - (1-y)'*log( 1 - sigmoid(X*theta))) ) + (lambda/(2*m))*sum(theta(2:length(theta)).*theta(2:length(theta))) ;

k=length(theta)-1;
n=length(theta);
grad(1)=1/m.*(sum(X'(1,:)*h-X'(1,:)*y));
for j=2:n
	grad(j)=(1/m).*(sum(X'(j,:)*h-X'(j,:)*y)+lambda.*theta(j,1));
end

end
