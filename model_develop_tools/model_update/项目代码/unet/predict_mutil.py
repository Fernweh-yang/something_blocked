import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src import UNet

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    classes = 1  # exclude background
    weights_path = r"save_weights\CH_best_model.pth"
    img_path = "./DRIVE/test/images/01_test.tif"
    roi_mask_path = r"D:\Codes\Deep learning\unet\DRIVE\test\1st_manual\01_manual1.gif"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=3, num_classes=classes+1, base_c=64)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from PIL image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # enter evaluation mode
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不感兴趣的区域像素设置成0(黑色)
        prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("test_result.png")

        # 绘制原始图像、真实mask和预测结果
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 显示原始图像
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # 显示真实的mask图像
        true_mask = Image.open(roi_mask_path).convert('L')
        axes[1].imshow(true_mask, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')

        # 显示预测结果
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')

        # 显示绘图
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
