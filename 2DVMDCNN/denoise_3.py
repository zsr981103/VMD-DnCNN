# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 14:16:16 2021
网络预测，带IMF
@author: 666
"""
import numpy as np
import torch
from DnCNN import *
from UNet import *
from torch import nn
import torch
import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
from get_patches import *
###数据和模型载入###
# 数据载入


# path = 'data/noise_mat_npy_data/part1_4_npy_mat/2007BP_part3_11shot.sgy'
# path = 'D:\Deep\FFTUNet_Project\data_and_result\Mobil_Avo_Viking_Graben_Line_12真实海洋波\sgy\Sea_2_10_shot.sgy'
# origin, nSample,extent_time = get_info_seg(path)
# sigma = 1000
# noise = np.random.normal(0, sigma / 255.0, origin.shape)
# noise_data = origin + noise
# noise_data = np.load('data/noise_mat_npy_data/npy/snr_6.npy')
# path = 'data/data_mat_npy_sgy/part1_4_npy_mat/2007BP_part4_11shot.sgy'
path = 'data/field_data/Sea_0_1_shot.sgy'
# noise_data, nSample, extent_time = get_info_seg(path)
# noise_data_original = np.load('data/data_mat_npy_sgy/part1_4_npy_mat/snr_5.965954873508071.npy')
# noise_data_original = np.load('data/data_mat_npy_sgy/npy/snr_6.npy')
noise_data = get_mat('data/data_mat_npy_sgy/VMD_noise/VMD_K/VMD_K_Sea_0_1_shot.mat')
# noise_data_original = noise_data
# origin = noise_data
# origin = np.load('data/record_result/sea/DnCNN/denoise_result.npy')
origin = get_mat('data/record_result/sea/VMD_2D/VMD_2D_denoise.mat')
origin_t, nSample, extent_time = get_info_seg(path)
noise_data_original = origin_t
# print(calculate_snr(origin,noise_data))
# noise_data = origin
vmd_data = True
if vmd_data == True:
    print("使用VMD分解数据")
    # origin = np.expand_dims(origin, axis=-1)
    clean_data = origin
    clean_data = np.expand_dims(clean_data, axis=-1)
    # clean_data = np.expand_dims(clean_data, axis=-1)
    # noise_data, clean_data = extract_paired_patches_3d(clean_data=clean_data, noise_data=noise_data,
    #                                                    patch_length=256, stride=128)
    noise_patches, origin_patches = predict_data_extract_paired_patches_3d(clean_data=clean_data, noise_data=noise_data,
                                                       patch_length=256, stride=128)

else:
    noise_patches, origin_patches = predict_data_extract_paired_patches(noise_data=noise_data,clean_data=origin,patch_length=256,stride=128)

# 模型参数载入
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = ResNet18().to(device)
B, C, T = noise_patches.shape

# model = UNet(in_channels=data_in_channel).to(device)
model = DnCNN(in_channels=C+1).to(device)
# weights_path = "D:\Deep\FFTUNet_Project\data_and_result\Shot_normalization\\DnCNN\sigma700\model.pth"
weights_path = "D:\Deep\VMD-DnCNN\\2DVMDCNN\data\\record_result\VMDDnCNN\\model.pth"
# weights_path = "data/results/model.pth"
# weights_path = "data/Bayesian/run_alpha_1.000_beta_0.054/model.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))

###数据预处理###

# 归一化
def normalization(data, _range):
    return data / _range


# 预测集归一化
range_p = np.max(np.abs(noise_patches))
range_o = np.max(np.abs(origin_patches))
range_p = max(range_p, range_o)
# if np.any(range_o > range_p):
#     range_p = np.maximum(range_p, range_o)  # 元素级最大值，防止广播问题
p_norm = normalization(noise_patches, range_p)
o_norm = normalization(origin_patches, range_p)


# np.save('range_p', range_p)
#
# 格式转换
# p_norm = noise_patches
# o_norm = origin_patches
p_data = torch.from_numpy(p_norm)
p_data = p_data.type(torch.FloatTensor)
o_data = torch.from_numpy(o_norm)
o_data = o_data.type(torch.FloatTensor)



if vmd_data == True:
    p_data = torch.cat([p_data,o_data], dim=1)


###数据去噪###
# 网络预测
train_start_time = time.time()
model.eval()
with torch.no_grad():
    output = model(p_data.to(device))["out"]
    # output = torch.squeeze(output).cpu().detach().numpy()
    output = output.cpu().detach().numpy()

# 数据重排和反归一化
output = output * range_p
# print("11111")
print(output.shape)
# 假设
data_size = 256
stride = 128
useful_start = 64  # 每个片段中使用的起始位置
useful_end = 192   # 每个片段中使用的结束位置
useful_len = useful_end - useful_start  # 中间有效长度128

# 降噪结果重建
# total_batch, _, data_size = output.shape  # 例如 output.shape = (4000, 1, 256)
total_batch, _, _data_size = output.shape  # 例如 output.shape = (4000, 1, 256)
n_samples, n_traces,C = noise_data.shape
segments_per_trace = total_batch // n_traces  # 每条震道用了多少个batch，应该是5
# num_segments = n_samples // data_size  # 完整的小块数 (4个256)
# remaining_samples = n_samples % data_size  # 如果有剩余部分
# 计算期望长度（用于后续重建）
expected_len = (segments_per_trace - 1) * stride + data_size

# 恢复后的数据将会是 (1151, 800) 形状
# reconstructed_data = np.zeros((n_samples, n_traces))  # 初始化一个空的数组来存放恢复后的震道数据
# 初始化重建数据与叠加次数（用于平均）
reconstructed_data = np.zeros((expected_len, n_traces))



for i in range(n_traces):
    start_batch = i * segments_per_trace
    end_batch = (i + 1) * segments_per_trace
    trace_batches = output[start_batch:end_batch, 0, :]  # shape: (segments_per_trace, 256)

    for j, segment in enumerate(trace_batches):
        seg_start = j * stride
        is_first = (j == 0)
        is_last = (j == segments_per_trace - 1)

        if is_first:
            # 第一个patch，使用前192个点
            patch_part = segment[:192]  # 0～192
            write_start = 0
            write_end = 192
        elif is_last:
            # 最后一个patch，使用后192～256的64个点
            patch_part = segment[64:]  # 192～256
            write_start = seg_start + 64
            write_end = seg_start + 256
            if write_end > expected_len:  # 越界保护
                patch_part = patch_part[:expected_len - write_start]
                write_end = expected_len
        else:
            # 中间patch，使用中间部分64～192
            patch_part = segment[64:192]
            write_start = seg_start + 64
            write_end = seg_start + 192

        # 写入重建矩阵（直接替换，无需平均）
        reconstructed_data[write_start:write_end, i] = patch_part


# 裁剪成原始长度（n_samples）
reconstructed_data = reconstructed_data[:n_samples, :]


train_end_time = time.time()
print(f"Time: {train_end_time - train_start_time:.4f}")
print(f"Rmse: {calculate_rmse(origin,reconstructed_data):.4f}")
print(f"Snr: {calculate_snr(origin,reconstructed_data):.4f}")
# print("denoist time:"+train_end_time-train_start_time)
# print("rmse:"+calculate_rmse(origin,reconstructed_data))
# print("snr:"+calculate_snr(origin,reconstructed_data))
#
# print(calculate_snr(torch.from_numpy(origin).type(torch.FloatTensor),torch.from_numpy(noise_data).type(torch.FloatTensor)))
# print(calculate_snr(torch.from_numpy(origin).type(torch.FloatTensor),torch.from_numpy(reconstructed_data).type(torch.FloatTensor)))
# plot_seismic_npy(noise_data,extent_time,show=True)
# plot_seismic_npy(origin,extent_time,show=True)
plot_seismic_npy(origin,extent_time,show=True)
plot_seismic_npy(reconstructed_data,extent_time,show=True)
plot_seismic_npy((noise_data_original-reconstructed_data),extent_time,show=True)
# plot_seismic_f_k_npy(noise_data,show=True)
# plot_seismic_f_k_npy(origin,show=True)
# plot_seismic_f_k_npy(reconstructed_data,show=True)
# plot_seismic_f_k_npy((noise_data_original-reconstructed_data),show=True)
# np.save('denoise_result', reconstructed_data)