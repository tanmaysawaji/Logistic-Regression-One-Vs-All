function [all_theta] = oneVsAll(X, y, num_labels, lambda)

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

for c = 1:num_labels

% Set initial theta
initial_theta=zeros(n+1,1);
%initial_theta=zeros(num_labels, n + 1);

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Run fmincg to obtain the optimal theta
% This function will return theta and the cost 
[theta]=fmincg(@(t)(lrCostFunction(t,X, (y == c), lambda)), initial_theta, options);

% Updating theta to overall all_theta
if (c==1)
	history_theta=theta';
else
	history_theta=[history_theta; theta'];
end

end

all_theta=history_theta;

end
