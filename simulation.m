clear all
close all
clc

% load 3D_exp1_1e-4_2e-3

% load data
% 
% for i = 1 : len
%     if(Sun_Sensor_Available(i))
%         ref = Sun_Sensor_J2000_Pos(i, 1 : 3);
%         ref = ref ./ norm(ref);
%         r = ref';
%         break;
%     end
% end
% 
dim = 3;
N = 1;
M = 1;

v_err = 1e-5;
h_err = 1e-4;

% r = randn(dim, N);
% A = randn(dim, dim, M);
% A = A * A' - eye(dim);

b = zeros(dim, N);
B = zeros(dim, dim, M);

% R = orthonormalize(randn(dim, dim));

A = [
 -0.91489, -0.37563, -0.14788;
 -0.37563,  0.65793,  0.65271;
 -0.14788,  0.65271, -0.74304];

R = [
 0.0056824, -0.85853,  0.51272;
   0.38732, -0.47082, -0.79266;
   0.92193,  0.20309,  0.32985];
R = orthonormalize(R);

r = [0.93976;
  -0.31362;
  -0.13596];

for i = 1 : N
    r(:, i) = r(:, i) ./ norm(r(:, i));
    b(:, i) = R * r(:, i) + v_err * randn(dim, 1);
    b(:, i) = b(:, i) ./ norm(b(:, i));
end

for i = 1 : M
    A(:, :, i) = orthonormalize(A(:, :, i));
    B(:, :, i) = R * A(:, :, i) * R' + h_err * randn(dim, dim);
    B(:, :, i) = orthonormalize(B(:, :, i));
end

% N = 1;
% M = 1;
% dim = 3;
% 
% A = [
%  -0.91489, -0.37563, -0.14788;
%  -0.37563,  0.65793,  0.65271;
%  -0.14788,  0.65271, -0.74304];
% 
% R = [
%  0.0056824, -0.85853,  0.51272;
%    0.38732, -0.47082, -0.79266;
%    0.92193,  0.20309,  0.32985];
% R = orthonormalize(R);
% 
% r = [0.93976;
%   -0.31362;
%   -0.13596];


P = b;
Q = r;

H = zeros(dim * dim, dim * dim);
for i = 1 : M
    H = H + (kron(A(:, :, i), eye(dim)) - kron(eye(dim), B(:, :, i)'))' * ...
            (kron(A(:, :, i), eye(dim)) - kron(eye(dim), B(:, :, i)'));
end

x = pinv(H + kron(Q', eye(dim, dim))' * kron(Q', eye(dim, dim))) * kron(Q', eye(dim, dim))' * reshape(P, [dim * N, 1]);
RR = reshape(x, [dim, dim]);

R_err = norm(orthonormalize(RR) - R, 'inf')

% save 3D_exp1_1e-4_2e-3

