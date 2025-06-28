# import signal
import math
import scipy.io as sio
from scipy.signal import convolve2d as conv2
from scipy import signal
from scipy.signal.windows import triang
from scipy.signal import convolve2d as conv2
import pywt
import numpy as np
import segyio
import torch
from matplotlib import pyplot as plt

def get_dx_from_segy(seg_file_path):
    """
    从 SEGY 文件中提取检波点（GroupX）的空间采样间隔 dx（单位：米）

    参数:
        seg_file_path (str): SEGY 文件路径

    返回:
        dx (float): 空间采样间隔（单位：米）
    """
    with segyio.open(seg_file_path, "r", ignore_geometry=True) as f:
        group_x = f.attributes(segyio.TraceField.GroupX)[:]  # 获取所有检波点X坐标
        print(group_x[:10])  # 打印前 10 个接收点位置
        dxs = np.diff(group_x)  # 相邻检波点之间的间距
        dx = np.median(dxs)     # 使用中位数避免异常值影响
        print(f"空间采样间隔 dx = {dx} 米")
    return dx
def fk_spectra(data, dt, dx, L=6):
    """
    f-k(频率-波数)频谱分析
    :param data: 二维的地震数据
    :param dt: 时间采样间隔
    :param dx: 道间距
    :param L: 平滑窗口
    :return: S(频谱结果), f(频率范围), k(波数范围)
    """
    print(data.shape)
    data = np.array(data)
    [nt, nx] = data.shape  # 获取数据维度
    # 计算nk和nf是为了加快傅里叶变换速度,等同于nextpow2
    i = 0
    while (2 ** i) <= nx:
        i = i + 1
    nk = 4 * 2 ** i
    j = 0
    while (2 ** j) <= nt:
        j = j + 1
    nf = 4 * 2 ** j
    S = np.fft.fftshift(abs(np.fft.fft2(data, (nf, nk))))  # 二维傅里叶变换
    H1 = np.hamming(L)
    # 设置汉明窗口大小，汉明窗的时域波形两端不能到零，而海宁窗时域信号两端是零。从频域响应来看，汉明窗能够减少很近的旁瓣泄露
    H = (H1.reshape(L, -1)) * (H1.reshape(1, L))
    S = signal.convolve2d(S, H, boundary='symm', mode='same')  # 汉明平滑
    S = S[nf // 2:nf, :]
    f = np.arange(0, nf / 2, 1)
    f = f / nf / dt
    k = np.arange(-nk / 2, nk / 2, 1)
    k = k / nk / dx
    return S, k, f
# def fk_spectra_1(data, dt, dx, L=6):
#     """
#     f-k(频率-波数)频谱分析
#     :param data: 二维的地震数据
#     :param dt: 时间采样间隔
#     :param dx: 道间距
#     :param L: 平滑窗口
#     :return: S(频谱结果), f(频率范围), k(波数范围)
#     """
#     data = np.array(data)
#     [nt, nx] = data.shape  # 获取数据维度
#     # 计算nk和nf是为了加快傅里叶变换速度,等同于nextpow2
#     i = 0
#     while (2 ** i) <= nx:
#         i = i + 1
#     nk = 4 * 2 ** i
#     j = 0
#     while (2 ** j) <= nt:
#         j = j + 1
#     nf = 4 * 2 ** j
#     S = np.fft.fftshift(abs(np.fft.fft2(data, (nf, nk))))  # 二维傅里叶变换
#     H1 = np.hamming(L)
#     # 设置汉明窗口大小，汉明窗的时域波形两端不能到零，而海宁窗时域信号两端是零。从频域响应来看，汉明窗能够减少很近的旁瓣泄露
#     H = (H1.reshape(L, -1)) * (H1.reshape(1, L))
#     S = signal.convolve2d(S, H, boundary='symm', mode='same')  # 汉明平滑
#     S = S[nf // 2:nf, :]
#     # f = np.arange(0, nf / 2, 1)
#     # f = f / nf / dt
#     f = np.fft.fftfreq(nf, dt)
#     f = f[:nf // 2]  # 只保留正频率
#     # k = np.arange(-nk / 2, nk / 2, 1)
#     # k = k / nk / dx
#     k = np.fft.fftfreq(nk, dx)
#     k = np.fft.fftshift(k)  # 因为 S 也做了 fftshift
#     return S, k, f
def plot_seismic_f_k_npy(seismic_data, save=False, save_path=None, show=False):
    dx = 125
    dt = 0.008
    S, k, f = fk_spectra(seismic_data, dt, dx)
    S[S <= 0] = 1e-10  # 避免对数域中的零值或负值
    amplitude_db = 10 * np.log10(S)


    plt.figure(figsize=(6, 6))
    plt.pcolormesh(k, f, amplitude_db, shading='auto', cmap='viridis', vmin=0, vmax=100)
    # plt.colorbar(label='Amplitude (dB)')
    plt.colorbar()
    plt.xlabel('k [c/m]')
    plt.ylabel('f [Hz]')
    # 倒转 y 轴，使得低频在底部，高频在顶部
    plt.gca().invert_yaxis()
    x_ticks = np.linspace(k.min(), k.max(), 3)  # 生成 5 个等间距的刻度
    plt.xticks(x_ticks)  # 设置 x 轴刻度
    # plt.title('f-k Spectrum')
    plt.subplots_adjust(left=0.18, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    # plt.show()
    # 你可以通过调整 k 和 f 的范围来放大图像
    # k_min, k_max = -0.01025, 0  # 设置你希望显示的波数范围
    # f_min, f_max = 0, 55  # 设置你希望显示的频率范围

    # 绘制频谱图
    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(k, f, amplitude_db, shading='auto', cmap='viridis', vmin=0, vmax=100)
    #
    # # 放大显示感兴趣的区域
    # plt.xlim(k_min, k_max)  # 限制 x 轴（波数 k）的范围
    # plt.ylim(f_min, f_max)  # 限制 y 轴（频率 f）的范围
    # plt.colorbar()
    # plt.xlabel('k [c/m]')
    # plt.ylabel('f [Hz]')
    # # 设置 x 轴刻度
    # x_ticks = np.linspace(k_min, k_max, 3)  # 生成 3 个等间距的刻度
    # plt.xticks(x_ticks)  # 设置 x 轴刻度
    # plt.subplots_adjust(left=0.18, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📷 Saved figure: {save_path}")
    elif show:
        plt.show()
    plt.close()
def plot_seismic_tensor(seis_tensor,extent_time, save=False, save_path=None, show=True):
    plt.figure(figsize=(4.5, 6))
    plt.imshow(seis_tensor, cmap='gray', extent=extent_time, aspect='auto', vmin=-1, vmax=1)
    # plt.colorbar(label='')
    plt.xlabel('Trace')
    plt.ylabel('Time(ms)')
    plt.title('')
    # 调整边距和间距
    plt.subplots_adjust(left=0.18, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📷 Saved figure: {save_path}")
    elif show:
        plt.show()
    plt.close()
def plot_seismic_npy(path_file,extent_time,save=False,save_path=None, show=False):
    dataset_t = path_file
    seis_tensor = torch.tensor(dataset_t)
    plot_seismic_tensor(seis_tensor, extent_time, save=save, save_path=save_path, show=show)
def calculate_snr(target_v, output_v):
    # 如果是Tensor，转为numpy
    if isinstance(output_v, torch.Tensor):
        output_v = output_v.detach().cpu().numpy()
    if isinstance(target_v, torch.Tensor):
        target_v = target_v.detach().cpu().numpy()

    # flatten后整体计算能量
    origSignal = target_v.flatten()
    errorSignal = (target_v - output_v).flatten()

    signal_power = np.sum(origSignal ** 2)
    noise_power = np.sum(errorSignal ** 2)

    # 避免除零错误
    if noise_power == 0:
        return float('inf')

    snr = 10 * math.log10(signal_power / noise_power)
    return snr
def calculate_rmse(origin, predicted):
    # 加载数据
    print(origin.shape)
    print(predicted.shape)

    # 确保数据形状一致
    if origin.shape != predicted.shape:
        raise ValueError("Origin and predicted signals must have the same shape.")

    # 计算均方根误差（RMSE）
    mse = np.mean((predicted - origin) ** 2)
    rmse = np.sqrt(mse)

    return rmse
def get_info_seg(seg_file_path):
    with segyio.open(seg_file_path, 'r', ignore_geometry=True) as f:
        # 读取所有地震道数据
        f.mmap()
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        nTrace = f.tracecount
        nSample = f.bin[segyio.BinField.Samples]
        startT = 0
        deltaT = f.bin[segyio.BinField.Interval]
        print("     Number of Trace   = %d" % (nTrace))
        print("     Number of Samples = %d" % (nSample))
        print("     Start Samples     = %d" % (startT))
        print("     Sampling Rate     = %d" % (deltaT))
        data = np.asarray([np.copy(trace) for trace in f.trace])
    data = data.T
    time_length = (nSample*deltaT)/1000.0
    extent_time = [0, nTrace, time_length, 0]
    return data, nSample,extent_time
def get_mat(mat_file_path):
    dataset_p = sio.loadmat(mat_file_path)
    print(dataset_p.keys())
    keys = list(dataset_p.keys())
    last_key = keys[-1]
    dataset_p = dataset_p[last_key]  # 假设文件中存有字符变量是matrix，
    return dataset_p

def predict_data_extract_paired_patches(noise_data, clean_data, patch_length=256, stride=128):
    """
    从噪声和干净数据中滑窗提取一一对应的 patch。
    输入：
        noise_data: shape = (1151, 16000)
        clean_data: shape = (1151, 16000)
    输出：
        noise_patches: shape = (N, 1, patch_length)
        clean_patches: shape = (N, 1, patch_length)
    """
    n_samples, n_traces = noise_data.shape
    noise_patches = []
    clean_patches = []

    # std_thresh = 0
    for trace in range(n_traces):
        start = 0
        while start + patch_length <= n_samples:
            n_patch = noise_data[start:start+patch_length, trace]
            c_patch = clean_data[start:start+patch_length, trace]


            noise_patches.append(n_patch[np.newaxis, :])
            clean_patches.append(c_patch[np.newaxis, :])
            start += stride

        # 若最后一段不够 patch_length，则从尾部往前截取完整 patch
        if start < n_samples:
            # end = n_samples
            # start_last = max(end - patch_length, 0)
            #
            # n_patch = noise_data[start_last:end, trace]
            # c_patch = clean_data[start_last:end, trace]
            #
            # noise_patches.append(n_patch[np.newaxis, :])
            # clean_patches.append(c_patch[np.newaxis, :])

            n_remain = noise_data[start:, trace]
            c_remain = clean_data[start:, trace]
            pad_len = patch_length - len(n_remain)

            n_padded = np.pad(n_remain, (0, pad_len), mode='constant')
            c_padded = np.pad(c_remain, (0, pad_len), mode='constant')

            noise_patches.append(n_padded[np.newaxis, :])
            clean_patches.append(c_padded[np.newaxis, :])

    return (
        np.array(noise_patches),  # shape: (N, 1, patch_length)
        np.array(clean_patches)
    )
def extract_paired_patches(noise_data, clean_data, patch_length=256, stride=128):
    """
    从噪声和干净数据中滑窗提取一一对应的 patch。
    输入：
        noise_data: shape = (1151, 16000)
        clean_data: shape = (1151, 16000)
    输出：
        noise_patches: shape = (N, 1, patch_length)
        clean_patches: shape = (N, 1, patch_length)
    """
    n_samples, n_traces = noise_data.shape
    noise_patches = []
    clean_patches = []
    std_thresh = 1e-3
    # std_thresh = 0
    for trace in range(n_traces):
        start = 0
        while start + patch_length <= n_samples:
            n_patch = noise_data[start:start+patch_length, trace]
            c_patch = clean_data[start:start+patch_length, trace]

            # 添加筛选条件
            if np.sum(c_patch) != 0 and np.std(c_patch) > std_thresh:
                noise_patches.append(n_patch[np.newaxis, :])
                clean_patches.append(c_patch[np.newaxis, :])
            start += stride

        # 补足最后一段
        if start < n_samples:
            # end = n_samples
            # start_last = max(end - patch_length, 0)
            #
            # n_patch = noise_data[start_last:end, trace]
            # c_patch = clean_data[start_last:end, trace]

            n_remain = noise_data[start:, trace]
            c_remain = clean_data[start:, trace]
            pad_len = patch_length - len(n_remain)

            n_patch = np.pad(n_remain, (0, pad_len), mode='constant')
            c_patch = np.pad(c_remain, (0, pad_len), mode='constant')

            if np.sum(c_patch) != 0 and np.std(n_patch) > std_thresh:
                noise_patches.append(n_patch[np.newaxis, :])
                clean_patches.append(c_patch[np.newaxis, :])

    return (
        np.array(noise_patches),  # shape: (N, 1, patch_length)
        np.array(clean_patches)
    )

def extract_paired_patches_3d(noise_data, clean_data, patch_length=256, stride=128):
    """
    支持通道数的版本。
    输入：
        noise_data: shape = (1151, 16000, 2)
        clean_data: shape = (1151, 16000, 1)
    输出：
        noise_patches: shape = (N, 2, patch_length)
        clean_patches: shape = (N, 1, patch_length)
    """
    n_samples, n_traces, n_channels_noise = noise_data.shape
    _, _, n_channels_clean = clean_data.shape

    noise_patches = []
    clean_patches = []
    std_thresh = 1e-3

    for trace in range(n_traces):
        start = 0
        while start + patch_length <= n_samples:
            n_patch = noise_data[start:start+patch_length, trace, :]  # (patch_len, 2)
            c_patch = clean_data[start:start+patch_length, trace, :]  # (patch_len, 1)

            n_patch = n_patch.T  # -> (2, patch_len)
            c_patch = c_patch.T  # -> (1, patch_len)

            if np.sum(c_patch) != 0 and np.std(n_patch) > std_thresh:
                noise_patches.append(n_patch)
                clean_patches.append(c_patch)

            start += stride

        # 补最后一段
        if start < n_samples:
            # n_remain = noise_data[start:, trace, :]  # (残长, 2)
            # c_remain = clean_data[start:, trace, :]  # (残长, 1)
            # pad_len = patch_length - len(n_remain)
            #
            # n_patch = np.pad(n_remain, ((0, pad_len), (0, 0)), mode='constant').T  # -> (2, patch_len)
            # c_patch = np.pad(c_remain, ((0, pad_len), (0, 0)), mode='constant').T  # -> (1, patch_len)
            start_last = n_samples - patch_length
            n_patch = noise_data[start_last:start_last + patch_length, trace, :].T
            c_patch = clean_data[start_last:start_last + patch_length, trace, :].T

            if np.sum(c_patch) != 0 and np.std(n_patch) > std_thresh:
                noise_patches.append(n_patch)
                clean_patches.append(c_patch)

    return (
        np.array(noise_patches),  # shape: (N, 2, patch_length)
        np.array(clean_patches)   # shape: (N, 1, patch_length)
    )


def extract_data_2d_extract_paired_patches(noise_data, clean_data, patch_length=64, stride=32):
    """
    从噪声和干净数据中提取 2D patch（时间 × 震道）。
    输出：
        noise_patches: shape = (N, 1, patch_length, patch_length)
        clean_patches: shape = (N, 1, patch_length, patch_length)
    """
    n_samples, n_traces = noise_data.shape
    noise_patches = []
    clean_patches = []

    std_thresh = 1e-3

    half_patch = patch_length // 2

    for center_trace in range(half_patch, n_traces - half_patch):
        start = 0
        while start + patch_length <= n_samples:
            # 时间窗口
            time_slice = slice(start, start + patch_length)
            # 空间窗口（震道）
            trace_slice = slice(center_trace - half_patch, center_trace + half_patch)



            n_patch = noise_data[time_slice, trace_slice]    # shape: (patch_length, patch_length)
            c_patch = clean_data[time_slice, trace_slice]

            # 筛选条件：避免全零和标准差过小的patch
            if np.sum(c_patch) != 0 and np.std(c_patch) > std_thresh:
                noise_patches.append(n_patch[np.newaxis, :, :])  # (1, patch_length, patch_length)
                clean_patches.append(c_patch[np.newaxis, :, :])
            # noise_patches.append(n_patch[np.newaxis, :, :])  # shape: (1, patch_length, patch_length)
            # clean_patches.append(c_patch[np.newaxis, :, :])
            start += stride

        # 处理最后一段（padding）
        if start < n_samples:
            n_remain = noise_data[start:, center_trace - half_patch:center_trace + half_patch]
            c_remain = clean_data[start:, center_trace - half_patch:center_trace + half_patch]

            pad_len = patch_length - n_remain.shape[0]
            n_padded = np.pad(n_remain, ((0, pad_len), (0, 0)), mode='constant')
            c_padded = np.pad(c_remain, ((0, pad_len), (0, 0)), mode='constant')

            if np.sum(c_patch) != 0 and np.std(c_patch) > std_thresh:
                noise_patches.append(n_patch[np.newaxis, :, :])
                clean_patches.append(c_patch[np.newaxis, :, :])
            # noise_patches.append(n_padded[np.newaxis, :, :])
            # clean_patches.append(c_padded[np.newaxis, :, :])

    noise_patches = np.array(noise_patches)  # shape: (N, 1, patch_length, patch_length)
    clean_patches = np.array(clean_patches)
    return noise_patches, clean_patches



def predict_data_extract_paired_patches_3d(noise_data, clean_data, patch_length=256, stride=128):
    """
    支持通道数的版本。
    输入：
        noise_data: shape = (1151, 16000, 2)
        clean_data: shape = (1151, 16000, 1)
    输出：
        noise_patches: shape = (N, 2, patch_length)
        clean_patches: shape = (N, 1, patch_length)
    """
    n_samples, n_traces, n_channels_noise = noise_data.shape
    # print(clean_data.shape)
    _, _, n_channels_clean = clean_data.shape

    noise_patches = []
    clean_patches = []
    std_thresh = 1e-3

    for trace in range(n_traces):
        start = 0
        while start + patch_length <= n_samples:
            n_patch = noise_data[start:start+patch_length, trace, :]  # (patch_len, 2)
            c_patch = clean_data[start:start+patch_length, trace, :]  # (patch_len, 1)

            n_patch = n_patch.T  # -> (2, patch_len)
            c_patch = c_patch.T  # -> (1, patch_len)

            noise_patches.append(n_patch)
            clean_patches.append(c_patch)

            # if np.sum(c_patch) != 0 and np.std(n_patch) > std_thresh:
            #     noise_patches.append(n_patch)
            #     clean_patches.append(c_patch)

            start += stride

        # 补最后一段
        if start < n_samples:
            # n_remain = noise_data[start:, trace, :]  # (残长, 2)
            # c_remain = clean_data[start:, trace, :]  # (残长, 1)
            # pad_len = patch_length - len(n_remain)
            #
            # n_patch = np.pad(n_remain, ((0, pad_len), (0, 0)), mode='constant').T  # -> (2, patch_len)
            # c_patch = np.pad(c_remain, ((0, pad_len), (0, 0)), mode='constant').T  # -> (1, patch_len)
            start_last = n_samples - patch_length
            n_patch = noise_data[start_last:start_last + patch_length, trace, :].T
            c_patch = clean_data[start_last:start_last + patch_length, trace, :].T

            noise_patches.append(n_patch)
            clean_patches.append(c_patch)

            # if np.sum(c_patch) != 0 and np.std(n_patch) > std_thresh:
            #     noise_patches.append(n_patch)
            #     clean_patches.append(c_patch)

    return (
        np.array(noise_patches),  # shape: (N, 2, patch_length)
        np.array(clean_patches)   # shape: (N, 1, patch_length)
    )

def cwt_real_imag_concat_2d(data, fs=125, totalscal=32, wavelet='cmor1.5-1.0'):
    """
    对形状为 (B, C, H, W) 的 torch.Tensor 数据进行 CWT，
    对最后一个维度（宽度方向）做CWT，
    将实部与虚部堆叠成 (B, C*2, H, W) 返回。

    参数：
        data: torch.Tensor，形状为 (B, C, H, W)
        fs: 采样频率
        totalscal: CWT尺度数量
        wavelet: 小波名称，默认 'cmor1.5-1.0'

    返回：
        torch.Tensor，形状为 (B, C*2, H, W)
    """
    assert data.ndim == 4, "输入必须是形状 (B, C, H, W)"
    B, C, S, T = data.shape

    Fc = pywt.central_frequency(wavelet)
    c = 2 * Fc * totalscal
    scales = c / np.arange(1, totalscal + 1)

    output = np.zeros((B, C * 2, S, T), dtype=np.float32)

    assert data.ndim == 4, "输入必须是形状为 (B, C, S, T)"
    B, C, S, T = data.shape

    # 构造小波尺度
    Fc = pywt.central_frequency(wavelet)
    c = 2 * Fc * totalscal
    scales = c / np.arange(1, totalscal + 1)

    # 初始化输出 (B, C*2, S, T)
    output = np.zeros((B, C * 2, S, T), dtype=np.float32)

    # 遍历每个样本、通道、震道
    for b in range(B):
        for c_idx in range(C):
            for t in range(T):
                sig = data[b, c_idx, :, t]  # shape: (S,) —— 每条震道的时间序列
                if isinstance(sig, torch.Tensor):
                    sig = sig.cpu().numpy()

                # 小波变换
                coefs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1 / fs)  # (scales, S)
                real_part = np.mean(np.real(coefs), axis=0)  # (S,)
                imag_part = np.mean(np.imag(coefs), axis=0)  # (S,)

                output[b, c_idx * 2, :, t] = real_part
                output[b, c_idx * 2 + 1, :, t] = imag_part

    return torch.from_numpy(output).float()


def cwt_real_imag_concat(data, fs=125):
    """
    对形状为 (B, C, 256) 的 torch.Tensor 数据进行 CWT，并将实部与虚部堆叠成 (B, 2, 256)

    参数：
        data: torch.Tensor，形状为 (B, 1, 256)
        wavelet: 小波函数名称，如 'cmor1.5-1.0'
        totalscal: 使用的尺度数量（即频率层级数）
        fs: 采样频率（Hz）

    返回：
        torch.Tensor，形状为 (B, 3, 256)
    """


    # wavelet = 'cmor3-3'
    assert data.ndim == 3, "输入必须是形状为 (B, 1, T) 的 Tensor"
    totalscal = 32
    wavelet = 'cmor1.5-1.0'
    B, C, T = data.shape

    Fc = pywt.central_frequency(wavelet)
    c = 2 * Fc * totalscal
    scales = c / np.arange(1, totalscal + 1)

    # 初始化输出实部和虚部数组
    # real_out = np.zeros((B, T), dtype=np.float32)
    # imag_out = np.zeros((B, T), dtype=np.float32)
    # features = np.zeros((B, 1, T), dtype=np.float32)
    # 初始化输出数组 (B, C*2, T)
    output = np.zeros((B, C * 2, T), dtype=np.float32)
    for b in range(B):
        for c_idx in range(C):
            sig = data[b, c_idx].cpu().numpy()  # shape: (T,)
            coefs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1 / fs)  # (scales, T)
            real_part = np.mean(np.real(coefs), axis=0)  # (T,)
            imag_part = np.mean(np.imag(coefs), axis=0)  # (T,)
            output[b, c_idx * 2] = real_part
            output[b, c_idx * 2 + 1] = imag_part
    # for b in range(B):
    #     sig = data[b, 0].cpu().numpy()  # shape: (256,)
    #     coefs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1/fs)  # (scales, T)
    #     real_out[b] = np.mean(np.real(coefs), axis=0)  # 平均投影到时间维度
    #     imag_out[b] = np.mean(np.imag(coefs), axis=0)
        # magnitude = np.abs(coefs)  # (scales, T)
        # mean_mag = np.mean(magnitude, axis=0)  # (T,)
        # features[b, 0] = mean_mag

    # print("--------------")
    # print(real_out.shape)
    # 堆叠成 (B, 2, T)
    # output = np.stack([real_out, imag_out], axis=1)
    # print(output.shape)
    return torch.from_numpy(output).float()
    # return torch.from_numpy(features).float()



def cwt_real_imag_concat_scales(data, fs=125):
    """
    对形状为 (B, C, T) 的 torch.Tensor 数据进行 CWT，保留每个尺度的实部和虚部，
    并将其堆叠为 (B, 2*C, T, S) 的输出。

    参数：
        data: torch.Tensor，形状为 (B, C, T)
        fs: 采样频率（Hz）

    返回：
        torch.Tensor，形状为 (B, 2*C, T, S)
    """
    assert data.ndim == 3, "输入必须是形状为 (B, C, T) 的 Tensor"
    totalscal = 64
    wavelet = 'cmor1.5-1.0'
    B, C, T = data.shape

    Fc = pywt.central_frequency(wavelet)
    c = 2 * Fc * totalscal
    scales = c / np.arange(1, totalscal + 1)

    # 输出 shape: (B, 2*C, T, S)
    output = np.zeros((B, 2 * C, T, totalscal), dtype=np.float32)

    for b in range(B):
        for c_idx in range(C):
            sig = data[b, c_idx].cpu().numpy()  # (T,)
            coefs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1 / fs)  # (S, T)

            real_part = np.real(coefs).T  # (T, S)
            imag_part = np.imag(coefs).T  # (T, S)

            output[b, c_idx * 2, :, :] = real_part
            output[b, c_idx * 2 + 1, :, :] = imag_part

    return torch.from_numpy(output).float()

# 示例用法
if __name__ == "__main__":
    # path = 'data/data_mat_npy_sgy/part1_4_npy_mat/2007BP_part1_11shot.sgy'
    #
    # noise_data_original = np.load('data/data_mat_npy_sgy/part1_4_npy_mat/snr_0.0382144709989505.npy')
    # noise_data = get_mat('data/data_mat_npy_sgy/VMD_noise/VMD_K/VMD_K_snr_0.0382144709989505.mat')
    #
    # origin, nSample, extent_time = get_info_seg(path)
    # plot_seismic_npy(noise_data, extent_time, show=True)
    # plot_seismic_f_k_npy(noise_data,show=True)
    # plot_seismic_f_k_npy(origin,show=True)
    # plot_seismic_npy((original_noise_data - reconstructed_data), extent_time, show=True)
    # plot_seismic_f_k_npy(noise_data,show=True)
    # plot_seismic_f_k_npy(origin,show=True)
    # sea_dx = 25,2007BP_dx = 125，land_dx = 25米
    vmd_noise = get_mat('data/data_mat_npy_sgy/VMD_noise/VMD_K.mat')
    clean_data, t, time = get_info_seg('data/2007BP_synthetic_train.sgy')
    clean_data = np.expand_dims(clean_data, axis=-1)
    # clean_data = np.expand_dims(clean_data, axis=-1)
    # print(vmd_noise.shape)
    # print(clean_data.shape)
    # n,c = extract_paired_patches_3d(noise_data=vmd_noise,clean_data=clean_data,patch_length=256,stride=128)
    # print(n.shape)
    # print(c.shape)
    np.random.seed(42)
    sigma = 700
    noise = np.random.normal(0, sigma / 255.0, clean_data.shape)
    # print(clean_data.shape)
    noise_data = clean_data + noise

    noise_patches, origin_patches = extract_paired_patches_3d(clean_data=clean_data, noise_data=vmd_noise,
                                                              patch_length=256, stride=128)

    print(noise_patches.shape)
    print(origin_patches.shape)




