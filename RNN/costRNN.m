function [J grad] = costRNN(thetaVec, X)

% Setup some useful variables
m = size(X, 1);
J = 0;
theta1 = reshape(thetaVec(1:24), 6, 4);
theta2 = reshape(thetaVec(25:end), 5, 1);
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));
hid_last = zeros(4, 1);
results = [];
for j = 1:(size(X,1)-1) %for every training element
    y = X(j+1,:); %expected output, the next element in the sequence
    context = sigmoid(hid_last);
    a1 = [X(j,:); context; 1]; %add bias, context units to input layer; 3x1
    z2 = theta1' * a1; %4x6 x 6x1
    a2 = [sigmoid(z2); 1]; %output hidden layer; 3x1
    hid_last = a2(1:end-1,1);
    z3 = theta2' * a2; %1x1
    a3 = sigmoid(z3);
    results(j) = a3;
    %calculate delta errors
    d3 = (a3 - y);
    d2 = (theta2 * d3) .* (a2 .* (1 - a2));
    %accumulate gradients
    theta1_grad = theta1_grad + (d2(1:4, :) * a1')'; 
    theta2_grad = theta2_grad + (d3 * a2')';
end
for n = 1:(m-1)
    a3n = results(n)';
    yn = X(n+1,:)';
    J = J + ( -yn'*log(a3n) - (1-yn)'*log(1-a3n) );
end
% cost function with regularization:
%reg = (lambda / (2*m)) * (sum((Theta1(size(Theta1,1)+1:end)).^2) + sum((Theta2(size(Theta2,1)+1:end)).^2));
J = (1/m) * J;
%J = J + reg;

% Unroll gradients
grad = [theta1_grad(:) ; theta2_grad(:)];


end