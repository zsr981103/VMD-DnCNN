% VMD去噪
clc
clear all
close all

%%

origSignal = load('part4.mat');
origSignal = double(origSignal.data);
noise = load('D:\Deep\FFTUNet_Project\本文方法\data\noise_mat_npy_data\mat\snr_-9.mat');
% noise = load('Land_2_6_shot.mat');
noise = double(noise.noise_data);




%% 处理每个震道的数据

[m, n] = size(noise);  % 获取数据尺寸
% denoise = zeros(m, n);  % 初始化去噪后的信号矩阵

K = 5;  % VMD分解层数

denoised_signal = zeros(m, n);
tic;
for i = 1:n  % 对每一条震道进行去噪
    % 提取当前震道的信号
    signal = noise(:, i);
    
    % 变分模态分解（VMD）
    % K = 4;  % 分解层数
    [u, residual] = vmd(signal, 'NumIMF', K);  % 对当前震道进行VMD
    u = u';
    
    % 重构去噪信号
    % denoise = sum(u(3,:)) + residual';
    denoise = sum(u(2:5,:));
    % 硬阈值处理
    % threshold = 0.05 * max(abs(denoise));
    % o = find(abs(denoise) > threshold);
    % denoise(o) = 0;

    % 计算去噪后的信号
    denoised_signal(:, i) = denoise;  % 将去噪后的震道信号存储到 denoised_signal
end
elapsed_time = toc;
fprintf('去噪所用时间: %.4f 秒\n', elapsed_time);
y = denoised_signal;
%% SNR & RMSE 计算
origin=origSignal;
signal_power = sum(origin(:).^2);  % 原始信号的能量

% 计算 MSE（均方误差）
mse_before = sum((noise(:) - origin(:)).^2);   % 噪声信号与原始信号的误差
mse_after  = sum((y(:) - origin(:)).^2); % 去噪后信号与原始信号的误差

% 计算 SNR（信噪比）
snr_before = 10 * log10(signal_power / mse_before);
snr_after  = 10 * log10(signal_power / mse_after);

% 计算 RMSE（均方根误差）
rmse_before = sqrt(mse_before / numel(origin));
rmse_after  = sqrt(mse_after / numel(origin));

% 打印结果
fprintf('SNR before: %.2f dB, RMSE before: %.4f\n', snr_before, rmse_before);
fprintf('SNR after : %.2f dB, RMSE after : %.4f\n', snr_after, rmse_after);

%%
% b_value = num2str(round(snr_before, 2), '%.2f'); % 保留两位小数
% a_value = num2str(round(snr_after, 2), '%.2f');  % 保留两位小数
% file_name = sprintf('Wavelet_denoise_b%s_a%s.mat', b_value, a_value);

%% 绘制剖面图（震道-时间采样点）
figure;
vmin = -1;
vmax = 1;
% 原始信号
subplot(2,2,1)
imagesc(origSignal); colormap(gray); caxis([vmin vmax]); colorbar;
title('原始信号');
xlabel('震道');
ylabel('时间采样点');
set(gca, 'YDir', 'reverse'); % 让时间向下增加


% 含噪的信号
subplot(2,2,2)
imagesc(noise); colormap(gray); caxis([vmin vmax]); colorbar;
title('含噪的信号');
xlabel('震道');
ylabel('时间采样点');
set(gca, 'YDir', 'reverse');
% 
% % 去噪后的信号
subplot(2,2,3)
imagesc(y); colormap(gray); caxis([vmin vmax]); colorbar;
title('去噪后的信号');
xlabel('震道');
ylabel('时间采样点');
set(gca, 'YDir', 'reverse');

% 去掉的噪声
subplot(2,2,4)
imagesc(noise-y); colormap(gray); caxis([vmin vmax]); colorbar;
title('去掉的噪声');
xlabel('震道');
ylabel('时间采样点');
set(gca, 'YDir', 'reverse');

sgtitle('去噪效果（震道 vs 时间采样点）');