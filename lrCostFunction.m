function [J, grad] = lrCostFunction(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples


n=length(theta);
z=X*theta;
h=sigmoid(z);
logisf=(-y)'*log(h)-(1-y)'*log(1-h);
thetas=theta(2:n,1);
J=((1/m).*sum(logisf))+(lambda/(2*m)).*sum(thetas.^2);

% Regularized
temp=theta;
temp(1)=0;
grad=(1/m).*(X'*(h-y)+lambda.*temp);








% =============================================================

grad = grad(:);

end
