clear all
close all
clc

format short g

base1 = randn(4, 1);
base2 = randn(4, 1);

len = 20000;
dt = 1 / 1000;
time = dt * (1 : len);
quaternion_true = zeros(len, 4);
accelerometer = zeros(len, 3);
gyroscope = zeros(len, 3);
magnetometer = zeros(len, 3);
ar = [0; 0; 1];
mr = [0.7; 0.1; -0.6];
mr = mr ./ norm(mr);
acc_noise = 0e-2;
mag_noise = 0e-4;
for i = 1 : len
    quat = sin(base1 * i * dt + base2) + 0.00 * randn(4, 1);
    quat = quat ./ norm(quat);
    quaternion_true(i, :) = quat';
    R = quat2dcm(quat');
    mr_ = mr + 0e-4 * randn(3, 1);
    mr_ = mr_ ./ norm(mr_);
    
    accelerometer(i, :) = (R * ar)' + acc_noise * randn(1, 3);
    magnetometer(i, :) = (R * mr_)' + mag_noise * randn(1, 3);
    if(i > 1)
        dR = R - last_R;
        S = - dR * R';
        u = [- S(2, 3), S(1, 3), - S(1, 2)];
        gyroscope(i, :) = u;
    end
    last_R = R;
end

q = quaternion_true(1, :);
quat = zeros(len, 4);
for i = 1 : len
    dq = 0.5 * quatmultiply(q, [0, gyroscope(i, :)]);
    q = q + dq;
    q = q ./ norm(q);
    quat(i, :) = q;
end

figure(1);
subplot(2, 2, 1);
plot(time, quat(:, 1), 'LineWidth', 1); hold on
plot(time, quaternion_true(:, 1), 'LineWidth', 1); hold off
xlim([0 max(time)]);
xlabel('Time (s)');
ylabel('q_0');

subplot(2, 2, 2);
plot(time, quat(:, 2), 'LineWidth', 1); hold on
plot(time, quaternion_true(:, 2), 'LineWidth', 1); hold off
xlim([0 max(time)]);
xlabel('Time (s)');
ylabel('q_1');

subplot(2, 2, 3);
plot(time, quat(:, 3), 'LineWidth', 1); hold on
plot(time, quaternion_true(:, 3), 'LineWidth', 1); hold off
xlim([0 max(time)]);
xlabel('Time (s)');
ylabel('q_2');

subplot(2, 2, 4);
plot(time, quat(:, 4), 'LineWidth', 1); hold on
plot(time, quaternion_true(:, 4), 'LineWidth', 1); hold off
xlim([0 max(time)]);
xlabel('Time (s)');
ylabel('q_3');


len_ = len;


Sigmas_x_ekf = zeros(7, 7, len_);
Sigmas_x_ukf = zeros(7, 7, len_);
state_ekf = zeros(len_, 7);
state_ukf = zeros(len_, 7);
q_proposed = quaternion_true(1, :)';
quaternion_analytical = zeros(len, 4);

N = 3;
M = 2;
is_symmetric = false;

v_err = 3e-1;
h_err = 3e-1;

r = randn(3, N);
A = randn(3, 3, M);

b = zeros(3, N);
B = zeros(3, 3, M);

gyro_noise = 1e-7;
bias_noise = 1e-3;
bias_true = randn(1, 3) * bias_noise;

x_ekf = [1; 0; 0; 0; 0; 0; 0];
x_ukf = [1; 0; 0; 0; 0; 0; 0];

global AA BB bb rr angular

filter_ekf = extendedKalmanFilter(...
    'StateTransitionFcn', @process, ...
    'MeasurementFcn', @measurement, ...
    'State', x_ekf, ...
    'StateCovariance', blkdiag(5e-1 * eye(4), 1e-6 * eye(3)), ...
    'ProcessNoise', blkdiag(1e-4 * eye(4), 1e-10 * eye(3)), ...
    'MeasurementNoise', blkdiag(v_err^2 * eye(3 * N), h_err^2 * eye(9 * M)));

filter_ukf = unscentedKalmanFilter(...
    'StateTransitionFcn', @process, ...
    'MeasurementFcn', @measurement, ...
    'State', x_ukf, ...
    'StateCovariance', blkdiag(5e-1 * eye(4), 1e-6 * eye(3)), ...
    'ProcessNoise', blkdiag(1e-4 * eye(4), 1e-10 * eye(3)), ...
    'MeasurementNoise', blkdiag(v_err^2 * eye(3 * N), h_err^2 * eye(9 * M)));

for j = 1 : len_
    
    meas = [];
    R = quat2dcm(quaternion_true(j, :));
    for i = 1 : N
        r(:, i) = r(:, i) ./ norm(r(:, i));
        b(:, i) = R * r(:, i) + v_err * randn(3, 1);
        b(:, i) = b(:, i) ./ norm(b(:, i));
        meas = [meas; zeros(3, 1)];
    end

    for i = 1 : M
        if(is_symmetric)
            A(:, :, i) = randn(3, 3);
            A(:, :, i) = A(:, :, i) * A(:, :, i)' - randn(1, 1) * eye(3);
        else
            A(:, :, i) = orthonormalize(A(:, :, i));
        end
    
        if(is_symmetric)
            noise = randn(3, 3);
            B(:, :, i) = R * A(:, :, i) * R' + h_err * noise * noise';
        else
            B(:, :, i) = R * A(:, :, i) * R' + h_err * randn(3, 3);
        end
    
        if(~is_symmetric)
            B(:, :, i) = orthonormalize(B(:, :, i));
        end
        
        meas = [meas; zeros(9, 1)];
    end
    
    AA = A;
    BB = B;
    bb = b;
    rr = r;
    angular = gyroscope(j, :) + gyro_noise * randn(1, 3) + bias_true;
    
    [x_ekf_, Sigma_x_ekf_] = predict(filter_ekf);
    [x_ekf, Sigma_x_ekf] = correct(filter_ekf, meas);
    Sigmas_x_ekf(:, :, j) = Sigma_x_ekf;
    x_ekf = x_ekf ./ norm(x_ekf);
    state_ekf(j, :) = x_ekf';
    
    [x_ukf_, Sigma_x_ukf_] = predict(filter_ukf);
    [x_ukf, Sigma_x_ukf] = correct(filter_ukf, meas);
    Sigmas_x_ukf(:, :, j) = Sigma_x_ukf;
    x_ukf = x_ukf ./ norm(x_ukf);
    state_ukf(j, :) = x_ukf';
    
    
    P = b;
    Q = r;

    H = zeros(9);
    for i = 1 : M
        H = H + (kron(A(:, :, i), eye(3)) - kron(eye(3), B(:, :, i)'))' * ...
            (kron(A(:, :, i), eye(3)) - kron(eye(3), B(:, :, i)'));
    end

    x = pinv(H + kron(Q', eye(3))' * kron(Q', eye(3))) * kron(Q', eye(3))' * reshape(P, [3 * N, 1]);

    RR1 = orthonormalize(reshape(x, [3, 3]));
    RR2 = orthonormalize(reshape(- x, [3, 3]));

    R_err1 = norm(RR1 - R, 'inf');
    R_err2 = norm(RR2 - R, 'inf');

    if(R_err1 < R_err2)
        R_err = R_err1;
        RR = RR1;
    else
        R_err = R_err2;
        RR = RR2;
    end
    
    q = dcm2quat(RR)';
    q = sign(q' * q_proposed) * q;
    q_proposed = q;
    
    quaternion_analytical(j, :) = q_proposed';
    
    if(mod(j, 50) == 0)
        j
        x_ekf'
        x_ukf'
        q_proposed'
        [quaternion_true(j, :), bias_true]
        
    end
end

if(sign(state_ekf(1000, 1)) ~= sign(quaternion_true(1000, 1)))
    state_ekf(:, 1 : 4) = - state_ekf(:, 1 : 4);
end

if(sign(state_ukf(1000, 1)) ~= sign(quaternion_true(1000, 1)))
    state_ukf(:, 1 : 4) = - state_ukf(:, 1 : 4);
end

if(sign(quaternion_analytical(1000, 1)) ~= sign(quaternion_true(1000, 1)))
    quaternion_analytical(:, 1 : 4) = - quaternion_analytical(:, 1 : 4);
end

[e1, e2, e3] = quat2angle(quaternion_true, 'XYZ');
euler_true = [e1, e2, e3] * 180 / pi;
[e1, e2, e3] = quat2angle(quaternion_analytical, 'XYZ');
euler_analytical = [e1, e2, e3] * 180 / pi;
[e1, e2, e3] = quat2angle(state_ekf(:, 1 : 4), 'XYZ');
euler_ekf = [e1, e2, e3] * 180 / pi;
[e1, e2, e3] = quat2angle(state_ukf(:, 1 : 4), 'XYZ');
euler_ukf = [e1, e2, e3] * 180 / pi;

time_ = (1 : len_) * dt;

figure(2);
subplot(3, 1, 1);
plot(time_, euler_analytical(:, 1), '.-', 'LineWidth', 1); hold on
plot(time_, euler_ekf(:, 1), '.-', 'LineWidth', 1); hold on
plot(time_, euler_ukf(:, 1), '-', 'LineWidth', 1); hold on
plot(time_, euler_true(1 : len_, 1), '--', 'LineWidth', 1); hold off
xlim([min(time_), max(time_)]);
xlabel('Time (s)');
ylabel('Roll (deg)');
legend('Proposed - Analytical', 'Proposed - EKF', 'Proposed - UKF', 'Reference');

subplot(3, 1, 2);
plot(time_, euler_analytical(:, 2), '.-', 'LineWidth', 1); hold on
plot(time_, euler_ekf(:, 2), '.-', 'LineWidth', 1); hold on
plot(time_, euler_ukf(:, 2), '-', 'LineWidth', 1); hold on
plot(time_, euler_true(1 : len_, 2), '--', 'LineWidth', 1); hold off
xlim([min(time_), max(time_)]);
xlabel('Time (s)');
ylabel('Pitch (deg)');
legend('Proposed - Analytical', 'Proposed - EKF', 'Proposed - UKF', 'Reference');

subplot(3, 1, 3);
plot(time_, euler_analytical(:, 3), '.-', 'LineWidth', 1); hold on
plot(time_, euler_ekf(:, 3), '.-', 'LineWidth', 1); hold on
plot(time_, euler_ukf(:, 3), '-', 'LineWidth', 1); hold on
plot(time_, euler_true(1 : len_, 3), '--', 'LineWidth', 1); hold off
xlim([min(time_), max(time_)]);
xlabel('Time (s)');
ylabel('Yaw (deg)');
legend('Proposed - Analytical', 'Proposed - EKF', 'Proposed - UKF', 'Reference');



figure(3);
subplot(2, 2, 1);
plot(time_, quaternion_analytical(:, 1), '.-', 'LineWidth', 1); hold on
plot(time_, state_ekf(:, 1), '.-', 'LineWidth', 1); hold on
plot(time_, state_ukf(:, 1), '-', 'LineWidth', 1); hold on
plot(time_, quaternion_true(1 : len_, 1), '--', 'LineWidth', 1); hold off
xlim([min(time_), max(time_)]);
xlabel('Time (s)');
ylabel('$q_0$', 'Interpreter', 'LaTeX');
title('$q_0$', 'Interpreter', 'LaTeX');
legend('Proposed - Analytical', 'Proposed - EKF', 'Proposed - UKF', 'Reference');

subplot(2, 2, 2);
plot(time_, quaternion_analytical(:, 2), '.-', 'LineWidth', 1); hold on
plot(time_, state_ekf(:, 2), '.-', 'LineWidth', 1); hold on
plot(time_, state_ukf(:, 2), '-', 'LineWidth', 1); hold on
plot(time_, quaternion_true(1 : len_, 2), '--', 'LineWidth', 1); hold off
xlim([min(time_), max(time_)]);
xlabel('Time (s)');
ylabel('$q_1$', 'Interpreter', 'LaTeX');
title('$q_1$', 'Interpreter', 'LaTeX');
legend('Proposed - Analytical', 'Proposed - EKF', 'Proposed - UKF', 'Reference');

subplot(2, 2, 3);
plot(time_, quaternion_analytical(:, 3), '.-', 'LineWidth', 1); hold on
plot(time_, state_ekf(:, 3), '.-', 'LineWidth', 1); hold on
plot(time_, state_ukf(:, 3), '-', 'LineWidth', 1); hold on
plot(time_, quaternion_true(1 : len_, 3), '--', 'LineWidth', 1); hold off
xlim([min(time_), max(time_)]);
xlabel('Time (s)');
ylabel('$q_2$', 'Interpreter', 'LaTeX');
title('$q_2$', 'Interpreter', 'LaTeX');
legend('Proposed - Analytical', 'Proposed - EKF', 'Proposed - UKF', 'Reference');

subplot(2, 2, 4);
plot(time_, quaternion_analytical(:, 4), '.-', 'LineWidth', 1); hold on
plot(time_, state_ekf(:, 4), '.-', 'LineWidth', 1); hold on
plot(time_, state_ukf(:, 4), '-', 'LineWidth', 1); hold on
plot(time_, quaternion_true(1 : len_, 4), '--', 'LineWidth', 1); hold off
xlim([min(time_), max(time_)]);
xlabel('Time (s)');
ylabel('$q_3$', 'Interpreter', 'LaTeX');
title('$q_3$', 'Interpreter', 'LaTeX');
legend('Proposed - Analytical', 'Proposed - EKF', 'Proposed - UKF', 'Reference');



figure(4);
subplot(1, 3, 1);
plot(time_, state_ekf(:, 5) / dt, '.-', 'LineWidth', 1); hold on
plot(time_, state_ukf(:, 5) / dt, '-', 'LineWidth', 1); hold on
plot(time_, bias_true(1) * ones(len_, 1) / dt, '--', 'LineWidth', 1); hold off
xlim([min(time_), max(time_)]);
xlabel('Time (s)');
ylabel('$\omega_{x, {\rm{bias}}} \ \rm{(rad / s)}$', 'Interpreter', 'LaTeX', 'FontSize', 15);
legend('EKF', 'UKF', 'Reference');

subplot(1, 3, 2);
plot(time_, state_ekf(:, 6) / dt, '.-', 'LineWidth', 1); hold on
plot(time_, state_ukf(:, 6) / dt, '-', 'LineWidth', 1); hold on
plot(time_, bias_true(2) * ones(len_, 2) / dt, '--', 'LineWidth', 1); hold off
xlim([min(time_), max(time_)]);
xlabel('Time (s)');
ylabel('$\omega_{y, {\rm{bias}}} \ \rm{(rad / s)}$', 'Interpreter', 'LaTeX', 'FontSize', 15);
legend('EKF', 'UKF', 'Reference');

subplot(1, 3, 3);
plot(time_, state_ekf(:, 7) / dt, '.-', 'LineWidth', 1); hold on
plot(time_, state_ukf(:, 7) / dt, '-', 'LineWidth', 1); hold on
plot(time_, bias_true(3) * ones(len_, 3) / dt, '--', 'LineWidth', 1); hold off
xlim([min(time_), max(time_)]);
xlabel('Time (s)');
ylabel('$\omega_{z, {\rm{bias}}} \ \rm{(rad / s)}$', 'Interpreter', 'LaTeX', 'FontSize', 15);
legend('EKF', 'UKF', 'Reference');


function x = process(x_)
    global angular
    angular_ = angular - x_(5 : 7)';
    q = [x_(1), x_(2), x_(3), x_(4)];
    dq = 0.5 * quatmultiply(q, [0, angular_]);
    q = q + dq;
    q = q ./ norm(q);
    x = [q'; x_(5 : 7)];
end

function z = measurement(q)
global AA BB bb rr

q0 = q(1);  q1 = q(2);  q2 = q(3);  q3 = q(4);

s = size(bb);
N = s(2);
s = size(AA);
M = s(3);

z = [];

for k = 1 : N
    for i = 1 : 3
        str = sprintf('b%d = bb(%d, k); r%d = rr(%d, k);', i, i, i, i);
        eval(str);
    end

    z_vec = [
      r1*(q0^2 + q1^2 - q2^2 - q3^2) + r2*(2*q0*q3 + 2*q1*q2) - r3*(2*q0*q2 - 2*q1*q3);
      r2*(q0^2 - q1^2 + q2^2 - q3^2) - r1*(2*q0*q3 - 2*q1*q2) + r3*(2*q0*q1 + 2*q2*q3);
      r3*(q0^2 - q1^2 - q2^2 + q3^2) + r1*(2*q0*q2 + 2*q1*q3) - r2*(2*q0*q1 - 2*q2*q3);
        ];
    z = [z; [b1; b2; b3] - z_vec];
end
   

for k = 1 : M
    for i = 1 : 3
        for j = 1 : 3
            str = sprintf('a%d%d = AA(%d, %d, k); b%d%d = BB(%d, %d, k);', i, j, i, j, i, j, i, j);
            eval(str);
        end
    end
    
    z_hand_eye = [
        a11*(q0^2 + q1^2 - q2^2 - q3^2) - b11*(q0^2 + q1^2 - q2^2 - q3^2) + a12*(2*q0*q3 + 2*q1*q2) - a13*(2*q0*q2 - 2*q1*q3) + b21*(2*q0*q3 - 2*q1*q2) - b31*(2*q0*q2 + 2*q1*q3);
        a21*(q0^2 + q1^2 - q2^2 - q3^2) - b21*(q0^2 - q1^2 + q2^2 - q3^2) + a22*(2*q0*q3 + 2*q1*q2) - a23*(2*q0*q2 - 2*q1*q3) - b11*(2*q0*q3 + 2*q1*q2) + b31*(2*q0*q1 - 2*q2*q3);
        a31*(q0^2 + q1^2 - q2^2 - q3^2) - b31*(q0^2 - q1^2 - q2^2 + q3^2) + a32*(2*q0*q3 + 2*q1*q2) - a33*(2*q0*q2 - 2*q1*q3) + b11*(2*q0*q2 - 2*q1*q3) - b21*(2*q0*q1 + 2*q2*q3);
        a12*(q0^2 - q1^2 + q2^2 - q3^2) - b12*(q0^2 + q1^2 - q2^2 - q3^2) - a11*(2*q0*q3 - 2*q1*q2) + a13*(2*q0*q1 + 2*q2*q3) + b22*(2*q0*q3 - 2*q1*q2) - b32*(2*q0*q2 + 2*q1*q3);
        a22*(q0^2 - q1^2 + q2^2 - q3^2) - b22*(q0^2 - q1^2 + q2^2 - q3^2) - a21*(2*q0*q3 - 2*q1*q2) + a23*(2*q0*q1 + 2*q2*q3) - b12*(2*q0*q3 + 2*q1*q2) + b32*(2*q0*q1 - 2*q2*q3);
        a32*(q0^2 - q1^2 + q2^2 - q3^2) - b32*(q0^2 - q1^2 - q2^2 + q3^2) - a31*(2*q0*q3 - 2*q1*q2) + a33*(2*q0*q1 + 2*q2*q3) + b12*(2*q0*q2 - 2*q1*q3) - b22*(2*q0*q1 + 2*q2*q3);
        a13*(q0^2 - q1^2 - q2^2 + q3^2) - b13*(q0^2 + q1^2 - q2^2 - q3^2) + a11*(2*q0*q2 + 2*q1*q3) - a12*(2*q0*q1 - 2*q2*q3) + b23*(2*q0*q3 - 2*q1*q2) - b33*(2*q0*q2 + 2*q1*q3);
        a23*(q0^2 - q1^2 - q2^2 + q3^2) - b23*(q0^2 - q1^2 + q2^2 - q3^2) + a21*(2*q0*q2 + 2*q1*q3) - a22*(2*q0*q1 - 2*q2*q3) - b13*(2*q0*q3 + 2*q1*q2) + b33*(2*q0*q1 - 2*q2*q3);
        a33*(q0^2 - q1^2 - q2^2 + q3^2) - b33*(q0^2 - q1^2 - q2^2 + q3^2) + a31*(2*q0*q2 + 2*q1*q3) - a32*(2*q0*q1 - 2*q2*q3) + b13*(2*q0*q2 - 2*q1*q3) - b23*(2*q0*q1 + 2*q2*q3);
        ];
    z = [z; z_hand_eye];
end

end