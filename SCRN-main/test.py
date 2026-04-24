# -*- coding: utf-8 -*-
from util.My_tool1 import *
import time
import torch
if __name__ == '__main__':

    model = torch.load('/home/zhangxin/hanwen/Github/SCRN-main/trained_model/model.pth', weights_only=False)

    model.eval()  # evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()

    x = np.load('/home/zhangxin/hanwen/Github/SCRN-main/test_data/clear.npy')
    x = x.astype(np.float64)

    y = np.load('/home/zhangxin/hanwen/Github/SCRN-main/test_data/noise_and_miss.npy')
    y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])


    torch.cuda.synchronize()
    start_time = time.time()
    y_ = y_.type(torch.float32)
    y_ = y_.cuda()

    x_ = model(y_)  # inferences
    x_ = x_.view(y.shape[0], y.shape[1])
    x_ = x_.cpu()
    x_ = x_.detach().numpy().astype(np.float64)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    pre_snr = snr_(y, x)
    print("before：snr" + str(pre_snr))
    snr = snr_(x_, x)
    print("After：snr"+str(snr))

    pre_ssim = ssim_(y, x)
    print("before：ssim" + str(pre_ssim))
    ssim = ssim_(x_, x)
    print("After：ssim" + str(ssim))

# 6. 【关键修改3】绘图并保存 (替代 My_tool1.show_x_y)
    print("\nPlotting results...")
    
    # 定义保存路径
    save_dir = '/home/zhangxin/hanwen/Github/SCRN-main/plt/test'

    plt.figure(figsize=(15, 5))
    
    # 图1: Ground Truth (干净数据)
    plt.subplot(131)
    plt.imshow(x, cmap='seismic', aspect='auto')
    plt.title('Ground Truth')
    plt.colorbar()

    # 图2: Noisy Input (输入数据)
    plt.subplot(132)
    plt.imshow(y, cmap='seismic', aspect='auto')
    plt.title(f'Input (SNR={pre_snr:.2f}dB)')
    plt.colorbar()

    # 图3: Denoised Output (去噪结果)
    plt.subplot(133)
    plt.imshow(x_, cmap='seismic', aspect='auto')
    plt.title(f'Output (SNR={snr:.2f}dB)')
    plt.colorbar()

    save_path = os.path.join(save_dir, 'result_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"Result image saved to: {save_path}")
    plt.close()













