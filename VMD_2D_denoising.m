%% 2D VMD去噪完整流程
clc; clear; close all;

%% 1. 数据载入
% 加载数据（请替换为您的实际数据路径）
origSignal = load('part4.mat');
origSignal = double(origSignal.data);
noise = load('D:\Deep\FFTUNet_Project\本文方法\data\noise_mat_npy_data\mat\snr_6.mat');
noise = double(noise.noise_data);

cleanSignal = origSignal;
noisySignal = noise;

%% 2. 显示原始和带噪声信号
vmin = -1;
vmax = 1;
figure;
subplot(2,1,1); 
imagesc(cleanSignal); 
colormap(gray); 
caxis([vmin vmax]); 
colorbar; 
title('原始信号');

subplot(2,1,2); 
imagesc(noisySignal); 
colormap(gray); 
caxis([vmin vmax]); 
colorbar; 
title('带噪声信号');

%% 3. 执行2D VMD分解
% 设置VMD参数
% [K, alpha, tau] = auto_params(noisySignal);
K = 4;          % 模态数（建议3-6）
alpha = 50;   % 带宽控制参数（建议1000-5000）
tau = 0.05;      % 时间步长
tol = 1e-6;     % 收敛容差
maxIter = 100;  % 最大迭代次数

% 执行VMD分解
[u, omega] = VMD_2D(noisySignal, alpha, K, tau, tol, maxIter);

%% 4. 显示分解结果
figure('Name','VMD模态分解结果');
for k = 1:K
    subplot(1,K,k);
    imagesc(u(:,:,k));
    title(['IMF ',num2str(k)]);
    colormap(gray);
    caxis([vmin vmax]);
end

%% 5. 设计模态选择策略（替代频域滤波器）
% modal_energy = squeeze(sum(sum(u.^2, 1), 2));
% energy_ratio = modal_energy / sum(modal_energy);
% retain_modes = energy_ratio > 0.05;  % 保留能量>10%的模态

% 策略2：手动选择模态（示例选择前3个）
retain_modes = [1 0 0 0];  % 对K=6的情况

%% 6. 重构信号
% denoisedSignal = sum(u(:,:,retain_modes), 3);
denoisedSignal = sum(u(:,:,logical(retain_modes)), 3);
SNR_original = calc_SNR(origSignal,noisySignal);
SNR_filtered = calc_SNR(origSignal,denoisedSignal);
fprintf('原始信噪比: %.2f dB\n', SNR_original);
fprintf('去噪后信噪比: %.2f dB\n', SNR_filtered);

%% 7. 显示最终结果
figure;
subplot(3,1,1); 
imagesc(cleanSignal); 
colormap(gray); 
caxis([vmin vmax]); 
colorbar;  
title('原始信号');

subplot(3,1,2); 
imagesc(noisySignal); 
colormap(gray); 
caxis([vmin vmax]); 
colorbar;  
title('带噪声信号' );

subplot(3,1,3); 
imagesc(denoisedSignal); 
colormap(gray); 
caxis([vmin vmax]); 
colorbar;  
title('VMD去噪后' );

%% 8. 时域波形对比
rowToShow = min(500, size(cleanSignal,1));  % 选择显示行
figure;
plot(cleanSignal(rowToShow,:), 'b', 'LineWidth', 1.5); hold on;
plot(noisySignal(rowToShow,:), 'r', 'LineWidth', 0.5);
plot(denoisedSignal(rowToShow,:), 'g', 'LineWidth', 1.5);
legend('原始信号', '带噪声信号', '去噪后信号');
title(['第', num2str(rowToShow), '行信号对比']);
xlabel('样本点'); ylabel('幅值');

%% 核心函数 =====================================================
function [u, omega] = VMD_2D(f, alpha, K, tau, tol, maxIter)
% 2D VMD实现
[M, N] = size(f);

% 对称扩展
f = padarray(f, [floor(M/2) floor(N/2)], 'symmetric');
[Mext, Next] = size(f);

% 初始化频率
omega_x = (0.5/K)*(1:2:2*K);
omega_y = (0.5/K)*(1:2:2*K);
omega = [omega_x; omega_y];

% 频域网格
[x, y] = meshgrid((0:Next-1)/Next, (0:Mext-1)/Mext);

% FFT变换
f_hat = fft2(f);
u_hat = zeros(Mext, Next, K);
lambda_hat = zeros(Mext, Next);


% 使用自定义初始化
% 主循环
for n = 1:maxIter
    u_hat_prev = u_hat;
    sum_uk = zeros(Mext, Next);
    
    for k = 1:K
       
        % 更新模态
        weight = 1./(alpha*((x - omega_x(k)).^2 + (y - omega_y(k)).^2) + 1);
        u_hat(:,:,k) = (f_hat - sum_uk + lambda_hat/2) .* weight;

        % 更新中心频率
        u_temp = ifft2(u_hat(:,:,k));
        omega_x(k) = sum(x(:).*abs(u_temp(:)))/sum(abs(u_temp(:)));
        omega_y(k) = sum(y(:).*abs(u_temp(:)))/sum(abs(u_temp(:)));

        sum_uk = sum_uk + u_hat(:,:,k);
    end
    
    % 更新拉格朗日乘子
    lambda_hat = lambda_hat + tau*(f_hat - sum_uk);
    
    % 检查收敛
    if n > 1 && norm(u_hat(:) - u_hat_prev(:)) < tol
        break;
    end
end

% 裁剪回原始大小
u = zeros(M, N, K);
for k = 1:K
    temp = real(ifft2(u_hat(:,:,k)));
    u(:,:,k) = temp(floor(M/2)+1:floor(M/2)+M, floor(N/2)+1:floor(N/2)+N);
end
omega = [omega_x; omega_y];
end
function [K, alpha, tau] = auto_params(noisySignal)
    % 估计噪声标准差（假设信号低频部分主要为有用信号）
    noise_std = std(noisySignal(:) - movmean(noisySignal(:), 50));
    signal_power = var(noisySignal(:));
    est_SNR = 10*log10(signal_power/noise_std^2);
    
    if est_SNR > 10  % 低噪声
        K = 4;
        alpha = 1000;
        tau = 0.2;
    elseif est_SNR > 5  % 中等噪声
        K = 5;
        alpha = 2000;
        tau = 0.1;
    else  % 高噪声
        K = 6;
        alpha = 3000;
        tau = 0.05;
    end
    fprintf('估计SNR=%.1f dB, 自动参数: K=%d, alpha=%d, tau=%.2f\n',...
            est_SNR, K, alpha, tau);
end
function snr = calc_SNR(clean, noisy)
% 计算信噪比
signal_power = norm(clean(:))^2;
noise_power = norm(noisy(:)-clean(:))^2;
snr = 10*log10(signal_power/noise_power);
end