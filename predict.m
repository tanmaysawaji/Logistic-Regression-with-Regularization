function p = predict(theta, X)

m = size(X, 1); % Number of training examples

z=X*theta;
h=sigmoid(z);
for k=1:m
  if h(k)>=0.5;
	  p(k)=1;
  else
	  p(k)=0;
  end

end
