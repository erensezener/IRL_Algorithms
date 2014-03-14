% Construct feature matrix
function F = npbfirlgenfeatmtrx(X, U)

% X             : representing contruction process of feature matrix
%   T (M x 1)   : vector indicating used base features
%   Z (M x K)   : IBP matrix indicating which base features are used in conjunction
%   A (M x K)   : matrix indicating negation of base feature
% U   (|S| x M) : base feature matrix over state space
% F   (|S| x K) : constructed feature matrix over state space

% F(:,k): the newly constructed feature as conjunction of base features
% if Z(m,k) == 1 && T(m) == 1 
%    if A(m,k) == 1, then F(:,k) = F(:,k) and U(:,m)
%    if A(m,k) == 0, then F(:,k) = F(:,k) and ~U(:,m)

K = size(X.Z, 2);                               % # of features
F = zeros(size(U, 1), K);                       % feature matrix
Y = bsxfun(@times, X.Z.*(2.*X.A - 1), X.T);     % temporal matrix (M x K)
for k = 1:K
    posU = U(:, Y(:, k) == 1);                  % positively chosen base features
    negU = U(:, Y(:, k) == -1);                 % negatively chosen base features
    if ~isempty(posU) && ~isempty(negU)         % make k-th conjunction
        F(:, k) = all(posU, 2) & ~any(negU, 2);
    elseif ~isempty(posU) && isempty(negU)
        F(:, k) = all(posU, 2);
    elseif isempty(posU) && ~isempty(negU)
        F(:, k) = ~any(negU, 2);
    end
end

end
