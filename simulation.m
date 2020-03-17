% Source codes of 
% Wu, J. (2020) Unified Attitude Determination Problem from Vector 
%               Observations and Hand-eye Measurements, 
%               IEEE Transactions on Aerospace and Electronic Systems.
%
% Author: Jin Wu
% E-mail: jin_wu_uestc@hotmail.com

clear all
close all
clc

dim = 3;
N = 1;
M = 1;

v_err = 1e-5;
h_err = 1e-4;

r = randn(dim, N);
A = randn(dim, dim, M);

b = zeros(dim, N);
B = zeros(dim, dim, M);

R = orthonormalize(randn(dim, dim));

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

