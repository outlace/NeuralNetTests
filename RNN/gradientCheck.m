function gradApprox =  gradientCheck(X, y, theta)
%theta is unrolled all thetas

n = numel(theta); %3 total weights
EPSILON = 10e-06;
gradApprox = zeros(size(theta));
for i = 1:n
    thetaPlus = theta;
    thetaPlus(i) = thetaPlus(i) + EPSILON;
    thetaMinus = theta;
    thetaMinus(i) = thetaMinus(i) - EPSILON;
    gradApprox(i) = (costFunction2(X, y, thetaPlus) - costFunction2(X, y, thetaMinus))/(2*EPSILON);

end;


end