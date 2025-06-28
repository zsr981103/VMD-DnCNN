% Test script for 2D-VMD
% Authors: Konstantin Dragomiretskiy and Dominique Zosso
% {konstantin,zosso}@math.ucla.edu
% http://www.math.ucla.edu/~{konstantin,zosso}
% Initial release 2014-03-17 (c) 2014
%
% When using this code, please do cite our papers:
% -----------------------------------------------
% K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE Trans.
% on Signal Processing, 62(3):531-544, 2014. DOI:10.1109/TSP.2013.2288675
%
% K. Dragomiretskiy, D. Zosso, Two-Dimensional Variational Mode
% Decomposition, IEEE Int. Conf. Image Proc. (submitted). Preprint
% available here: ftp://ftp.math.ucla.edu/pub/camreport/cam14-16.pdf
%

%% preparations

close all;
clc;
clear all;


% Sample data
% origSignal = load('part4.mat');
% origSignal = load('D:\Deep\FFTUNet_Project\本文方法\data\noise_mat_npy_data\part1_4_npy_mat\2007BP_part2_11shot.mat');
% origSignal = origSignal.data;

% texture = load('D:\Deep\FFTUNet_Project\data_and_result\Mobil_Avo_Viking_Graben_Line_12真实海洋波\mat\Sea_2_10_shot.mat');
% f = texture.data;
% texture =load('D:\Deep\FFTUNet_Project\本文方法\data\noise_mat_npy_data\mat\snr_0.mat');
% texture = load("walk_5_shot.mat");
% texture = load("D:\Deep\EMDUNet\2DVMDCNN\data\data_mat_npy_sgy\mat\snr_-1.mat");
file_path = "D:\Deep\FFTUNet_Project\本文方法\data\field_data\Sea_0_1_shot.mat";
% snr_-9.970544992434437.mat，snr_-5.013031825339793.mat
texture = load(file_path);
% texture = load('D:\Deep\FFTUNet_Project\本文方法\data\noise_mat_npy_data\part1_4_npy_mat\snr_0.0382144709989505.mat');
% f = texture.noise_data;
f = texture.data;
% cleanSignal = origSignal;
noisySignal = f;
% texture = load('texture.mat');
% f = texture.f;
% 显示原始和带噪声信号
vmin = -1;
vmax = 1;
figure;
% subplot(2,1,1); imagesc(cleanSignal); colormap(gray); caxis([vmin vmax]); colorbar; title('原始信号');
subplot(2,1,2); imagesc(noisySignal); colormap(gray); caxis([vmin vmax]); colorbar; title('带噪声信号');
% parameters:
alpha = 220;       % bandwidth constraint
tau = 0;         % Lagrangian multipliers dual ascent time step
K = 3;              % number of modes
DC = 0;             % includes DC part (first mode at DC)
init = 0;           % initialize omegas randomly, may need multiple runs!
tol = 1e-7;      % tolerance (for convergence)

% snr_est = SNR_original;
% if snr_est < -10
%     alpha = 1500; K = 6; retain_modes = max(1,K-3):K;
% elseif snr_est < -5
%     alpha = 350; K = 3; retain_modes = 2:2;
% elseif snr_est < -1
%     alpha = 300; K = 3; retain_modes = 1:2;
% elseif snr_est < 3
%     alpha = 250; K = 3; retain_modes = 1:3;
% else
%     alpha = 175; K = 3; retain_modes = 1:3;
% end
% fprintf('参数 a: %.4f  k: %.4f  retain_modes: %.4f\n', alpha,K,retain_modes);
%% run actual 2D VMD code
tic;
[u, u_hat, omega] = VMD_2D(f, alpha, tau, K, DC, init, tol);
filteredSignal = sum(u(:,:, 1:2), 3); % 自适应保留模态
toc;
elapsed_time = toc-tic;
fprintf('去噪所用时间: %.4f 秒\n', elapsed_time);
save('VMD_2D_denoise.mat', 'filteredSignal');
VMD_K = u(:, :, 1:3);  % 取全部模态
[~, file_name, ~] = fileparts(file_path);  % file_name = 'snr_-1'
file_name = char(file_name);               % 强制转为字符型
save_name = ['VMD_K_', file_name, '.mat']; % 此时 save_name 是 char 类型
save(save_name, 'VMD_K');
% save('VMD_K.mat', 'VMD_K');
%% 计算信噪比评估去噪效果
% originalPower = sum(cleanSignal(:).^2);
% noisePower = sum((noisySignal(:)-cleanSignal(:)).^2);
% filteredNoisePower = sum((filteredSignal(:)-cleanSignal(:)).^2);
% 
% SNR_original = 10*log10(originalPower/noisePower);
% SNR_filtered = 10*log10(originalPower/filteredNoisePower);
% % 计算 RMSE（均方根误差）
% rmse_before = sqrt(noisePower / numel(cleanSignal));
% rmse_after  = sqrt(filteredNoisePower / numel(cleanSignal));
% % 打印结果
% fprintf('SNR before: %.2f dB, RMSE before: %.4f\n', SNR_original, rmse_before);
% fprintf('SNR after : %.2f dB, RMSE after : %.4f\n', SNR_filtered, rmse_after);
% % 显示结果
% figure;
% % subplot(3,1,1); imagesc(cleanSignal); colormap(gray); caxis([vmin vmax]); colorbar; title('原始信号');
% subplot(3,1,2); imagesc(noisySignal); colormap(gray); caxis([vmin vmax]); colorbar; title('带噪声信号');
% subplot(3,1,3); imagesc(filteredSignal); colormap(gray); caxis([vmin vmax]); colorbar; title('去噪后信号');
%% Visualization
% figure('Name', 'Input image');
% imagesc(f);
% colormap gray;
% axis equal;
% axis off;
% 
% figure('Name', 'Input spectrum');
% imagesc(abs(fftshift(fft2(f))));
% colormap gray;
% axis equal;
% axis off;
% hold on;
% colors = 'rbycmg';
% % for k = 1:size(omega,3)
% %     plot( size(f,2)*(0.5+omega(:,1,k)), size(f,1)*(0.5+omega(:,2,k)), colors(k) );
% % end
% 
% for k=1:size(u,3)
%     figure('Name', ['Mode #' num2str(k)]);
%     imagesc(u(:,:,k));
%     % colormap gray;
%     colormap(gray);
%     caxis([vmin vmax]);
%     colorbar;
%     % axis equal;
%     % axis off;
% end
% % 显示分解得到的各模态
% for k = 1:K
%     figure;
%     imagesc(u(:,:,k));
%     colormap(gray);
%     title(['VMD 模态 ', num2str(k)]);
%     xlabel('震道'); ylabel('采样点');
% end
% 
% 
% figure('Name', 'Reconstructed composite');
% imagesc(sum(u,3));
% colormap gray;
% axis equal;
% axis off;

