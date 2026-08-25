import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # check files
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# import os
# from PIL import Image
# import numpy as np
# from torch.utils.data import Dataset

# class CustomDataset(Dataset):
#     def __init__(self, root: str, train: bool, transforms=None):
#         super(CustomDataset, self).__init__()
#         # 根据train参数，确定是加载训练数据还是测试数据
#         self.flag = "train" if train else "test"
#         # 拼接数据集的根路径和对应的子路径（train或test）
#         data_root = os.path.join(root, self.flag)
#         # 检查路径是否存在
#         assert os.path.exists(data_root), f"path '{data_root}' does not exist."
#         self.transforms = transforms  # 图像变换操作
#         # 获取图像文件名列表，仅保留扩展名为.jpg的文件
#         img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".jpg")]
#         # 生成图像路径列表
#         self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
#         # 生成mask1路径列表，文件名格式为"图像名_1stHO.png"
#         self.mask1_list = [os.path.join(data_root, "mask1", i.split(".")[0] + "_1stHO.png")
#                            for i in img_names]
#         # 生成mask2路径列表，文件名格式为"图像名_2ndHO.png"
#         self.mask2_list = [os.path.join(data_root, "mask2", i.split(".")[0] + "_2ndHO.png")
#                            for i in img_names]

#         # 检查mask1和mask2文件是否存在
#         for i in self.mask1_list + self.mask2_list:
#             if os.path.exists(i) is False:
#                 raise FileNotFoundError(f"file {i} does not exist.")

#     def __getitem__(self, idx):
#         # 打开第idx个图像并转换为RGB格式
#         img = Image.open(self.img_list[idx]).convert('RGB')
#         # 打开对应的mask1和mask2并转换为灰度图（L模式）
#         mask1 = Image.open(self.mask1_list[idx]).convert('L')
#         mask2 = Image.open(self.mask2_list[idx]).convert('L')

#         # 将mask1和mask2转换为二值化的numpy数组（0和1）
#         mask1 = np.array(mask1) / 255
#         mask2 = np.array(mask2) / 255

#         # 如果需要可以组合mask1和mask2，这里只使用mask1
#         # mask = np.clip(mask1 + mask2, a_min=0, a_max=1)
#         mask = mask1
#         mask = Image.fromarray(mask)  # 将numpy数组转换回PIL图像

#         # 如果定义了transforms操作，则对图像和mask进行变换
#         if self.transforms is not None:
#             img, mask = self.transforms(img, mask)

#         # 返回处理后的图像和对应的mask
#         return img, mask

#     def __len__(self):
#         # 返回数据集的大小，即图像列表的长度
#         return len(self.img_list)

#     @staticmethod
#     def collate_fn(batch):
#         # 将一个batch中的图像和标签分别打包
#         images, targets = list(zip(*batch))
#         # 将图像打包为一个张量，缺失的部分用0填充
#         batched_imgs = cat_list(images, fill_value=0)
#         # 将标签打包为一个张量，缺失的部分用255填充
#         batched_targets = cat_list(targets, fill_value=255)
#         return batched_imgs, batched_targets

# def cat_list(images, fill_value=0):
#     # 计算批次中每个维度的最大值
#     max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
#     # 生成包含所有图像的张量，形状为（批次数量, 最大尺寸）
#     batch_shape = (len(images),) + max_size
#     # 创建一个用fill_value填充的新张量
#     batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
#     # 将每个图像复制到对应的张量中，保持原有图像尺寸
#     for img, pad_img in zip(images, batched_imgs):
#         pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
#     return batched_imgs  # 返回打包好的张量


