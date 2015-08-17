X = [0;0;0;0;1;1;1;0;1;1;1;0]; %12x1
numIn = 1;
numHid = 2;
numOut = 1;
%theta1 = ( 0.5 * sqrt ( 6 / ( numIn + numHid) ) * randn( numIn + numHid + 1, numHid ) );
%theta2 = ( 0.5 * sqrt ( 6 / ( numHid + numOut ) ) * randn( numHid + 1, numOut ) );
epsilon_init = 0.12;
theta1 = rand(numIn + numHid + 1, numHid) * 2 * epsilon_init - epsilon_init;
theta2 = rand( numHid + 1, numOut ) * 2 * epsilon_init - epsilon_init;
theta1_grad = zeros(numIn + numHid + 1, numHid);
theta2_grad = zeros(numHid + 1, numOut);
epochs = 1000;
alpha = 0.01; %learning rate
epsilon = 0.0001; %momentum rate
thetaVec = [theta1(:);theta2(:)];
hid_last = zeros(numHid, 1);
last_change1 = zeros(numIn + numHid + 1, numHid);
last_change2 = zeros(numHid + 1, numOut);

for j = 1:(size(X,1)-1) %for every training element
    y = X(j+1,:); %expected output, the next element in the sequence
    context = sigmoid(hid_last);
    a1 = [X(j,:); context; 1]; %add bias, context units to input layer; 3x1
    z2 = theta1' * a1; %2x1
    a2 = [sigmoid(z2); 1]; %output hidden layer; 3x1
    hid_last = a2(1:end-1,1);
    z3 = theta2' * a2; %1x1
    a3 = sigmoid(z3);
    %calculate delta errors
    d3 = (a3 - y);
    d2 = (theta2 * d3) .* (a2 .* (1 - a2));
    %accumulate gradients
    theta1_grad = theta1_grad + (d2(1:numHid, :) * a1')'; 
    theta2_grad = theta2_grad + (d3 * a2')';
end
theta1_change = alpha * theta1_grad + epsilon * last_change1;
theta2_change = alpha * theta2_grad + epsilon * last_change2;
theta1 = theta1 - theta1_change;
theta2 = theta2 - theta2_change;
%for momentum term
last_change1 = theta1_change;
last_change2 = theta2_change;
%reset gradients
theta1_grad = zeros(numIn + numHid + 1, numHid);
theta2_grad = zeros(numHid + 1, numOut);
%compute cost function
thetaVec_ = [theta1(:);theta2(:)];
err = costRNN(X, thetaVec_);
%  value to see how more training helps.
options = optimset('MaxIter', 150);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costRNN(X, thetaVec_);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, thetaVec_, options);

