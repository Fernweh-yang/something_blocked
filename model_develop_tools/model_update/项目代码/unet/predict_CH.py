import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet


def time_synchronized():
    # 如果GPU可用，则等待GPU计算完毕后再返回当前时间，否则直接返回当前时间
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # 类别数量为1（不包括背景）
    weights_path = r"D:\Codes\Deep learning\unet\save_weights\CH_best_model.pth"  # 预训练权重文件的路径
    img_path = r"D:\Codes\Deep learning\unet\data\test\images\Image_01L.jpg"  # 待推理的测试图像路径
    assert os.path.exists(weights_path), f"weights {weights_path} not found."  # 检查权重文件是否存在
    assert os.path.exists(img_path), f"image {img_path} not found."  # 检查测试图像是否存在

    mean = (0.709, 0.381, 0.224)  # 图像归一化所用的均值
    std = (0.127, 0.079, 0.043)  # 图像归一化所用的标准差

    # 获取设备信息，优先使用GPU，如果不可用则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 创建模型实例
    model = UNet(in_channels=3, num_classes=classes + 1, base_c=64)

    # 加载预训练权重到模型中
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)  # 将模型移动到设备上（GPU或CPU）

    # 加载图像
    original_img = Image.open(img_path).convert('RGB')  # 以RGB格式打开测试图像

    # 定义图像预处理操作，包括转换为张量和归一化
    data_transform = transforms.Compose([
        transforms.ToTensor(),  # 将PIL图像转换为Tensor
        transforms.Normalize(mean=mean, std=std)  # 归一化处理
    ])
    img = data_transform(original_img)  # 对图像进行预处理
    # 扩展维度，使其符合模型输入的形状要求（增加一个batch维度）
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 关闭梯度计算，节省内存和加快推理速度
        # 初始化模型以便推理
        img_height, img_width = img.shape[-2:]  # 获取图像的高度和宽度
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)  # 创建一个与输入图像大小一致的空张量
        model(init_img)  # 用初始化的空张量进行一次前向传播，预热模型

        t_start = time_synchronized()  # 记录开始推理的时间
        output = model(img.to(device))  # 将图像输入模型并进行推理
        t_end = time_synchronized()  # 记录推理结束的时间
        print("inference time: {}".format(t_end - t_start))  # 打印推理时间

        prediction = output['out'].argmax(1).squeeze(0)  # 获取预测结果，并去掉多余的维度
        prediction = prediction.to("cpu").numpy().astype(np.uint8)  # 将预测结果转换为CPU上的numpy数组，并转换为uint8类型
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255

        mask = Image.fromarray(prediction)  # 将预测结果转换为PIL图像
        mask.save("test_result_test.png")  # 保存预测结果图像


if __name__ == '__main__':
    main()  # 运行主函数
