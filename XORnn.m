X = [0 0;0 1;1 0;1 1];
y = [0;1;1;0];
numIn = 2;
numHid = 3;
numOut = 1;
theta1 = ( 0.5 * sqrt ( 6 / ( numIn + numHid) ) * randn( numIn + 1, numHid ) );
theta2 = ( 0.5 * sqrt ( 6 / ( numHid + numOut ) ) * randn( numHid + 1, numOut ) );
epochs = 12000;
theta1_grad = zeros(numIn + 1, numHid);
theta2_grad = zeros(numHid + 1, numOut);
alpha = 0.05;
thetaVec = [theta1(:);theta2(:)];
minErr = 10e-11;
%disp(costFunction2(X, y, thetaVec));
for t = 1:epochs
    for i = 1:size(X,1)
        a1 = [X(i, :), 1]; %add bias; 1x3
        z2 = a1 * theta1; %1x3 * 3x2 = 1x2
        a2 = [sigmoid(z2) 1]; %1x3
        z3 = a2 * theta2; %1x3 * 3x1 = 1x1
        a3 = sigmoid(z3);

        delta3 = (a3 - y(i));
        delta2 = (theta2 * delta3) .* (a2 .* (1 - a2))';
        %(delta3 * theta2(2:end,:)') .* sigmoidGradient(z2);; %3x1
                    %3x1 * 1x1 = 3x1 .* 1x3 = 3x1
        %
        theta1_grad = theta1_grad + (delta2(1:numHid, :) * a1)';
                            %3x2  +  2x1 * 1x3 = 2x3     
        theta2_grad = theta2_grad + (delta3 * a2)';
                                    %1x1 * 1x3 = 1x3
        
    end;
    if t == 1
        gradVec = [theta1_grad(:); theta2_grad(:)];
        disp(reshape(gradVec, 1 , numel(gradVec)));
        gradChkVec = gradientCheck(X, y, thetaVec);
        disp(reshape(gradChkVec, 1, numel(gradChkVec))); 
    end
    theta1 = theta1 - alpha*theta1_grad;
    theta2 = theta2 - alpha*theta2_grad;
    theta1_grad = zeros(numIn + 1, numHid);
    theta2_grad = zeros(numHid + 1, numOut);
    thetaVec_ = [theta1(:);theta2(:)];
    err = costFunction2(X, y, thetaVec_);
    disp(err);
    if err < minErr
        disp('Done!');
        disp(err);
        break;
    end
end;
thetaVec = [theta1(:);theta2(:)];
disp(costFunction2(X, y, thetaVec));
%gradVec = [theta1_grad(:); theta2_grad(:)];
%disp(reshape(gradVec, 1 , numel(gradVec)));
%gradChkVec = gradientCheck(X, y, thetaVec);
%disp(reshape(gradChkVec, 1, numel(gradChkVec))); 
a1 = [X ones(4,1)];
z2 = a1 * theta1; %4x3 * 3x2 = 4x2
a2 = [sigmoid(z2) ones(4,1)]; %4x3
z3 = a2 * theta2; %4x3 * 3x1 = 4x1
a3 = sigmoid(z3);
disp(a3)