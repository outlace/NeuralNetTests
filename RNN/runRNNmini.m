%X = [0;0;0;0;1;1;1;0;1;1;1;0;0;0;0;0;1;1;1;0;1;1;1;0;0;0;0;0;1;1;1;0;1;1;1;0];
X = [0;0;1;1;0];
Y = [0;0;1;0;1];
numIn = 1;
numHid = 4;
numOut = 1;
%  value to see how more training helps.
%'MaxIter', 60000, 'TolFun', 1e-11, 'MaxFunEvals',1000,'LargeScale', 'off', 'GradObj', 'on'
options = optimset('TolX', 1e-19);
options = optimset(options, 'TolFun', 1e-19);
options = optimset(options, 'MaxIter', 50);
theta1 = ( 0.5 * sqrt ( 6 / ( numIn + numHid) ) * randn( numIn + numHid + 1, numHid ) );
theta2 = ( 0.5 * sqrt ( 6 / ( numHid + numOut ) ) * randn( numHid + 1, numOut ) );
epsilon_init = 0.12;
%theta1 = rand(numIn + numHid + 1, numHid) * 2 * epsilon_init - epsilon_init;
%theta2 = rand( numHid + 1, numOut ) * 2 * epsilon_init - epsilon_init;
thetaVec_ = [theta1(:);theta2(:)];
% Create "short hand" for the cost function to be minimized
costFunction = @(p) costRNN(p, X);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost, info] = fmincg(costFunction, thetaVec_, options);
%disp(sprintf('Cost: %d',cost));
theta1 = reshape(nn_params(1:24), 6, 4);
theta2 = reshape(nn_params(25:end), 5, 1);
results = [];
%exp_y = [];
hid_last = zeros(numHid, 1);
Xt = [1;0;1;1;1;0];
for j = 1:(size(Xt,1)) %for every training element
    %y = X(j+1,:); %expected output, the next element in the sequence
    context = sigmoid(hid_last);
    a1 = [Xt(j,:); context; 1]; %add bias, context units to input layer; 3x1
    z2 = theta1' * a1; %2x1
    a2 = [sigmoid(z2); 1]; %output hidden layer; 3x1
    hid_last = a2(1:end-1,1);
    z3 = theta2' * a2; %1x1
    a3 = sigmoid(z3);
    results(j) = a3;
    %exp_y(j) = y;
end

disp(sprintf('Results:'))
disp(round(results));