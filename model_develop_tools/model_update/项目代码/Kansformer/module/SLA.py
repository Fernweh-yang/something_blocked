import torch
import torch.nn as nn
from einops import rearrange  # 导入 rearrange 函数，用于重排张量

# 定义简化版的线性注意力类，继承自 nn.Module
class SimplifiedLinearAttention(nn.Module):
    r"""
    参数:
        dim (int): 输入通道数。
        window_size (tuple[int]): 窗口的高和宽。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool, 可选): 如果为 True，则为查询、键和值添加一个可学习的偏置。默认值：True
        qk_scale (float | None, 可选): 如果设置，则覆盖默认的 qk scale，即 head_dim ** -0.5。
        attn_drop (float, 可选): 注意力权重的丢弃比率。默认值：0.0
        proj_drop (float, 可选): 输出的丢弃比率。默认值：0.0
    """

    # 初始化函数，定义了注意力模块的基本结构
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        self.dim = dim  # 输入通道数
        self.window_size = window_size  # 窗口大小，分别为高和宽
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个注意力头的维度

        self.focusing_factor = focusing_factor  # 聚焦因子
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 线性变换，用于生成查询、键和值
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力权重的 dropout
        self.proj = nn.Linear(dim, dim)  # 最终的线性变换
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的 dropout

        self.softmax = nn.Softmax(dim=-1)  # 定义 Softmax 函数用于计算权重

        # 深度可分离卷积 (Depthwise Convolution)，用于捕捉局部特征
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        # 可学习的相对位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))

        # 打印初始化信息，便于调试
        print('Linear Attention window{} f{} kernel{}'.format(window_size, focusing_factor, kernel_size))

    # 前向传播函数
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入特征，形状为 (num_windows*B, N, C)
            mask: (0/-inf) 掩码，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None
        """
        B, N, C = x.shape

        # # 获取输入的形状信息
        # B, C ,H ,W = x.shape  # B是批次大小，C是通道数，H和W是高度和宽度
        # N = H*W  # N是图像的总像素数
        # x = x.view(B,N,C)  # 调整输入形状为 (B, N, C)

        # 对 x 进行线性变换，并重新排列得到查询、键和值
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)  # 将查询、键和值分开
        k = k + self.positional_encoding  # 为键添加相对位置编码

        kernel_function = nn.ReLU()  # 使用 ReLU 作为激活函数
        q = kernel_function(q)
        k = kernel_function(k)

        # 对查询、键和值进行多头分解
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]  # 获取形状信息

        # 使用混合精度训练，保证数值精度并优化计算效率
        with torch.cuda.amp.autocast(enabled=False):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

            # 计算注意力权重
            z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
            if i * j * (c + d) > c * d * (i + j):  # 根据输入规模选择不同的计算路径
                kv = torch.einsum("b j c, b j d -> b c d", k, v)  # 计算键值对的内积
                x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)  # 结合查询计算最终结果
            else:
                qk = torch.einsum("b i c, b j c -> b i j", q, k)  # 查询和键的点积
                x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)  # 结合查询计算输出

        # 将特征图恢复为二维形状
        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)  # 重排为 (B, C, W, H)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")  # 经过卷积后重排回 (B, N, C)
        x = x + feature_map  # 将卷积结果与注意力结果相加

        # 恢复多头后的形状
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)  # 线性变换
        x = self.proj_drop(x)  # 输出 dropout
        # print(x.shape) # torch.Size([4, 1024, 64])
        # x = x.permute(0,2,1).view(B,C,H,W)  # 恢复输入的原始形状

        return x  # 返回最终输出


# 调试部分，主程序入口
if __name__ == "__main__":
    # x = torch.randn(4,64,32,32)  # 生成一个随机张量作为输入，形状为 (4, 64, 32, 32)
    # sla_attention = SimplifiedLinearAttention(64,[32,32],8)  # 实例化注意力模块，通道数为64，窗口大小为32x32，8个注意力头

    x = torch.randn(4,32*32,64)  # 可以替换的另一种输入格式示例
    sla_attention = SimplifiedLinearAttention(64,[32,32],8)
    out = sla_attention(x)  # 进行前向传播
    print(out.shape)  # 打印输出的形状，应该是 torch.Size([4, 64, 32, 32])
