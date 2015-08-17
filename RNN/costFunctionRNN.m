function J = costFunctionRNN( X, thetaVec )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
theta1 = reshape(thetaVec(1:24), 6, 4);
theta2 = reshape(thetaVec(25:end), 5, 1);
hid_last = zeros(4, 1);
results = zeros(11,1);
m = size(X,1);
J = 0;
%forward propagation
for j = 1:(size(X,1)-1) %for every training element
    y = X(j+1,:); %expected output, the next element in the sequence
    context = sigmoid(hid_last);
    a1 = [X(j,:); context; 1]; %add bias, context units to input layer; 3x1
    z2 = theta1' * a1; %2x1
    a2 = [sigmoid(z2); 1]; %output hidden layer; 3x1
    hid_last = a2(1:end-1,1);
    z3 = theta2' * a2; %1x1
    a3 = sigmoid(z3);
    results(j) = a3;
end
for n = 1:(m-1)
    a3n = results(n)';
    yn = X(n+1,:)';
    J = J + ( -yn'*log(a3n) - (1-yn)'*log(1-a3n) );
end
J = (1/m) * J;

end

