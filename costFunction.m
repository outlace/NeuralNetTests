function j = costFunction( X, y, theta1 )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

a1 = [X ones(4,1)];
z1 = a1 * theta1; %4x3 * 3x1 = 4x1
a2 = sigmoid(z1); %4x1
j = (1/4)*sum(a2 - y)^2;
end

