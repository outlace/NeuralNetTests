%X = [0;1;1;1;0;1;1;1;0;0;0;0;0;1;1;1;0;1;1;1;0;0;0;0;0;1;1;1;0;1;1;1;0;0;0;0;0;1;1;1;0;1;1;1;0;0;0;0;0;1;1;1;0;1;1;1;0]; %12x1
%X = [1;1;0;1;1;0;1;1;0;1;1;0];
X = [0;0;1;1;0];
Y = [0;0;1;0;1];
numIn = 1;
numHid = 4;
numOut = 1;
theta1 = ( 1 * sqrt ( 6 / ( numIn + numHid) ) * randn( numIn + numHid + 1, numHid ) );
theta2 = ( 1 * sqrt ( 6 / ( numHid + numOut ) ) * randn( numHid + 1, numOut ) );
theta1_grad = zeros(numIn + numHid + 1, numHid);
theta2_grad = zeros(numHid + 1, numOut);
epochs = 10000;
alpha = 0.01;
epsilon = 0.00;
thetaVec_ = [theta1(:);theta2(:)];
disp('Initial Cost:');
disp(costFunctionRNN(X, thetaVec_));
minErr = 1e-1;
hid_last = zeros(numHid, 1);
last_change1 = zeros(numIn + numHid + 1, numHid);
last_change2 = zeros(numHid + 1, numOut);
m = size(X,1);
for i = 1:epochs
    %forward propagation
    s = 1;%randi([1 (m-1)]);
    for j = s:(m-1) %for every training element
        %y = X(j+1,:); %expected output, the next element in the sequence
        y = Y(j,:);
        context = sigmoid(hid_last);
        x1 = X(j,:);
        a1 = [x1; context; 1]; %add bias, context units to input layer; 3x1
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
    theta1_change = alpha * (1/m)*theta1_grad + epsilon * last_change1;
    theta2_change = alpha * (1/m)*theta2_grad + epsilon * last_change2;
    theta1 = theta1 - theta1_change;
    theta2 = theta2 - theta2_change;
    last_change1 = theta1_change;
    last_change2 = theta2_change;
    %reset gradients
    theta1_grad = zeros(numIn + numHid + 1, numHid);
    theta2_grad = zeros(numHid + 1, numOut);
    %compute cost function
    thetaVec_ = [theta1(:);theta2(:)];
    err = costFunctionRNN(X, thetaVec_);
    if mod(i, 10) == 0
        disp(err);
    end
end
runRNN(thetaVec_);
disp(sprintf('Error at end: %d', err));

