X = [0 0;0 1;1 0;1 1];
y = [0;0;1;1];
theta1 = randn(3,1);
epochs = 50;
theta1_grad = zeros(3,1);
alpha = 0.1;
disp(costFunction(X, y, theta1));
for t = 1:epochs
    for i = 1:size(X,1)
        a1 = [X(i, :), 1]; %add bias
        z1 = a1 * theta1;
        a2 = sigmoid(z1);
        a2_delta = (a2 - y(i));
        theta1_grad = theta1_grad + (a1' * a2_delta);
    end;
    theta1 = theta1 - alpha*theta1_grad;
end;
disp(costFunction(X, y, theta1));

a1 = [X ones(4,1)];
z1 = a1 * theta1; %4x3 * 3x1 = 4x1
a2 = sigmoid(z1); %4x1
disp(a2)