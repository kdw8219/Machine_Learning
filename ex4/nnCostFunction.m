function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%



%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


X = [ ones(m,1) X ];
a1= sigmoid(X * Theta1');

a1 = [ ones(m, 1) a1 ];
a2 = sigmoid(a1 * Theta2');

%a2 is result of feed propagation
%10 5000

theta1_mod = Theta1(:,2:end);
theta2_mod = Theta2(:,2:end);

%size(a2) = 5000 10
%size(y) = 5000 1
%num_labels 10
%m = 5000

y_modifed = zeros(num_labels, m); %10 5000

for i=1:m
  y_modifed(y(i),i) = 1; %Size of result vector was not mapped. so resizing it
end;

fnc = (-y_modifed').*log(a2)-(1-y_modifed').*log(1-a2);
J = sum(sum(fnc))./m + (lambda/2).*sum(sum(sum(theta1_mod.^2))+sum(sum(theta2_mod.^2)))./m;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


for i=1:m,
  %step 1 - feedforward
  
  %  size(Theta1) 25, 401
  %  size(Theta2) 10, 26
  
  a1 = X(i,:); % each training set may be 1, 25
  z2 = Theta1 * a1';
  a2= sigmoid(z2); %25, 1

  a2 = [ 1; a2 ]; % 26, 1
  z3 = Theta2 * a2; % [10, 26][26, 1]
  a3 = sigmoid(z3); % [10, 1] thing is left
  
  %step 2 & 3 - make delta

  %size(y_modifed) % 10, 5000
  
  z2 = [1; z2];

  last_delta = a3 - y_modifed(:,i); %y_modifed can be helpful...
  front_delta = (Theta2'*last_delta).*sigmoidGradient(z2);
  
  %step 4 - Theta_grad calculation
  front_delta = front_delta(2:end);
  
  %size(last_delta) % 10 1
  %size(front_delta) % 25 1
  
  Theta2_grad = Theta2_grad + last_delta*a2';
  Theta1_grad = Theta1_grad + front_delta*a1; % * [1,25]
  
  %size(Theta2_grad) % 10 26
  %size(Theta1_grad) % 25 401
  
  
end

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

%Regularization.............
Theta2_grad(:, 2:end) = Theta2_grad(:,2:end)./m + (lambda/m)*Theta2(:, 2:end);
Theta2_grad(:, 1) = Theta2_grad(:,1)./m;

Theta1_grad(:, 2:end) = Theta1_grad(:,2:end)./m + (lambda/m)*Theta1(:, 2:end);
Theta1_grad(:, 1) = Theta1_grad(:,1)./m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
