def fused_ssim(img1, img2, padding=True):
    from pytorch_msssim import ssim
    return ssim(img1, img2, data_range=1.0, size_average=True)
