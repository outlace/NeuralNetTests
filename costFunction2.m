function J = costFunction2( X, y, thetaVec )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
theta1 = reshape(thetaVec(1:9), 3,3);
theta2 = reshape(thetaVec(10:end), 4, 1);
m = size(X,1);

a1 = [X ones(4,1)];
z2 = a1 * theta1; %4x3 * 3x2 = 4x2
a2 = [sigmoid(z2) ones(4,1)]; %4x3
z3 = a2 * theta2; %4x3 * 3x1 = 4x1
a3 = sigmoid(z3);
j = (1/2)*sum(a3 - y)^2;
J = 0;
for n = 1:m
    yn = y(n,:)';
    a3n = a3(n,:);
    J = J + ( -yn'*log(a3n) - (1-yn)'*log(1-a3n) );
end
J = (1/m) * J;

end