import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= 1. 基础物理增强算子 =================

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.ReflectionPad2d(padding),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0),
        nn.PReLU()
    )

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x  # Apply channel attention
        x = self.sa(x) * x  # Apply spatial attention
        return x

class MSResidualBlock(nn.Module):
    """多尺度残差块：双分支不同感受野"""

    def __init__(self, channels):
        super(MSResidualBlock, self).__init__()
        self.c1 = channels // 2
        self.c2 = channels - self.c1
        self.path1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, self.c1, kernel_size=3),
            nn.PReLU()
        )
        self.path2 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(channels, self.c2, kernel_size=3, dilation=2),
            nn.PReLU()
        )
        self.fuse = nn.Conv2d(self.c1 + self.c2, channels, kernel_size=1)

    def forward(self, x):
        res = x
        x1 = self.path1(x)
        x2 = self.path2(x)
        out = self.fuse(torch.cat([x1, x2], dim=1))
        return out + res



class SBARB(nn.Module):
    """
    光谱波段自适应重标定块 (Spectral Band-Adaptive Recalibration Block)
    优化点：引入双池化感知机制 (Avg + Max Pooling)
    """

    def __init__(self, channels):
        super(SBARB, self).__init__()
        self.base = MSResidualBlock(channels)

        # 自适应计算 k (基于 ECA 原理)
        import math
        t = int(abs((math.log(channels, 2) + 1) / 2))
        k = t if t % 2 else t + 1

        # 光谱门控：捕获双重全局统计量
        self.spectral_gate = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.base(x)
        b, c, h, w = feat.size()

        # 1. 提取双重池化特征
        y_avg = F.adaptive_avg_pool2d(feat, 1).view(b, 1, c)  # 全局背景
        y_max = F.adaptive_max_pool2d(feat, 1).view(b, 1, c)  # 显著特征

        # 2. 特征融合与权重生成
        # 将两种统计量融合，增强对极端光谱响应的捕捉能力
        y = y_avg + y_max
        weights = self.spectral_gate(y).view(b, c, 1, 1)

        return feat * weights

# ================= 2. 核心模块化组件 =================


#
class STPN(nn.Module):
    """风格迁移预测网络：整合 SOC 与 Decoder"""

    def __init__(self, out_channels, hs_bands):
        super(STPN, self).__init__()
        self.soc = SOC_Module(out_channels)
        self.decoder = nn.Sequential(
            MSResidualBlock(out_channels),
            nn.Conv2d(out_channels, hs_bands, kernel_size=1)
        )

    def forward(self, content_feat, style_feat):
        fused_feat = self.soc(content_feat, style_feat)
        return self.decoder(fused_feat)


class TLEM(nn.Module):
    """Temporal Latent Evolution Module (TLEM)"""

    def __init__(self, channels):
        super(TLEM, self).__init__()
        self.conv_init = conv_block(channels*2,channels)
        self.cbam = CBAM(channels)
        self.change_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.evolver = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            # nn.Conv2d(channels, channels, kernel_size=1),
            MSResidualBlock(channels)
        )

    def forward(self, latent, ms_feature):
        map_init = self.conv_init(
        torch.cat([latent, ms_feature], dim=1)
        )

        map = self.cbam(map_init)

        attention = self.change_gate(map)

        latent_update = latent*(1-attention)+ms_feature*attention

        latent_t = self.evolver(
            torch.cat([latent_update, ms_feature], dim=1)
        )

        return latent_t, attention



class HFDR(nn.Module):

    """
    HS空间高保真融合
    输入:
        f12: T2预测HS
        f1 : T1 HS
        map: 时间变化权重

    输出:
        HS2
    """

    def __init__(self, hs_bands):

        super().__init__()


        self.refine = nn.Sequential(

            nn.Conv2d(
                hs_bands+1,
                hs_bands,
                1
            ),

            MSResidualBlock(hs_bands),

            MSResidualBlock(hs_bands),

            nn.Conv2d(
                hs_bands,
                hs_bands,
                1
            )
        )


    def forward(
        self,
        f12,
        f1,
        map
    ):

        # 时间变化区域使用T2结果
        # 稳定区域保留T1光谱

        fused = (
            f12*map
            +
            f1*(1-map)
        )


        out = self.refine(
            torch.cat(
                [
                    fused,
                    map
                ],
                dim=1
            )
        )


        return out

class TemporalModeling(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.extract = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            MSResidualBlock(channels),
            CBAM(channels),
            MSResidualBlock(channels)
        )

    def forward(self, f_ms1, f_ms2):

        delta_t = self.extract(
            torch.cat([f_ms1, f_ms2], dim=1)
        )

        return delta_t


class SOC_Module(nn.Module):
    """
    用一阶轻量化调制代替二阶协方差。
    只捕捉光谱的全局分布（均值和标准差），而不是复杂的二阶相关性。
    """
    def __init__(self, channels):
        super(SOC_Module, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 用 HS 的风格生成缩放 (Gamma) 和偏移 (Beta)
        self.to_gamma_beta = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.PReLU(),
            nn.Conv2d(channels, channels * 2, 1)
        )
        self.refine = MSResidualBlock(channels)

    def forward(self, content_feat, style_feat):
        # 1. 提取光谱风格的全局特征
        style_stats = self.pool(style_feat)
        gamma_beta = self.to_gamma_beta(style_stats)
        gamma, beta = torch.split(gamma_beta, style_feat.size(1), dim=1)

        gamma = torch.tanh(gamma) * 0.5 #缩放
        beta = torch.tanh(beta) * 0.1 #偏移

        # 2. 对内容特征进行一阶调制：y = x * gamma + beta
        # 这种方式比二阶矩阵乘法温和得多，能保留原始的边缘强度
        out = content_feat * (1 + gamma) + beta
        
        return self.refine(out) + content_feat


#     """
#     轻量、标准的风格编码器
#     用于从 HS/MS 图像中提取纯粹的风格特征图，提供一阶统计量（均值与标准差）
#     """
#     def __init__(self, in_channels, out_channels):
#         super(AdaINStyleEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.PReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.PReLU()
#         )

#     def forward(self, x):
#         return self.encoder(x)

class SpectralModeling(nn.Module):
    """
    学习 MS 到 HS 的光谱补全关系

    不是光谱变化，
    而是生成高光谱先验
    """

    def __init__(self, channels):

        super().__init__()

        self.ms_align = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                1
            ),

            MSResidualBlock(channels)

        )


        self.extract = nn.Sequential(

            nn.Conv2d(
                channels*2,
                channels,
                1
            ),

            MSResidualBlock(channels),

            CBAM(channels),

            MSResidualBlock(channels)

        )


    def forward(
        self,
        fc_m1,
        fc_h1
    ):

        ms_feat = self.ms_align(fc_m1)


        spectral_prior = self.extract(

            torch.cat(
                [
                    ms_feat,
                    fc_h1
                ],
                dim=1
            )

        )


        return spectral_prior



class LatentStateFusion(nn.Module):
    """
    T1时刻地表隐状态构建

    输入:
        fc_m1 : MS1空间结构特征
        fc_h1 : HS1光谱内容特征
        delta_s : MS-HS光谱映射先验

    输出:
        latent0 : T1完整时空谱隐状态
    """

    def __init__(self, channels):
        super().__init__()

        self.fuse = nn.Sequential(

            nn.Conv2d(
                channels*2,
                channels,
                1
            ),

            nn.PReLU(),

            MSResidualBlock(channels),

            CBAM(channels),

            MSResidualBlock(channels)
        )


    def forward(
        self,
        fc_m1,
        # fc_h1,
        delta_s
    ):

        latent0 = self.fuse(
            torch.cat(
                [
                    fc_m1,
                    # fc_h1,
                    delta_s
                ],
                dim=1
            )
        )

        return latent0


class LatentDecoder(nn.Module):

    """
    隐状态解码器

    latent state -> HS image
    """

    def __init__(
        self,
        channels,
        hs_bands
    ):

        super().__init__()


        self.decoder = nn.Sequential(

            MSResidualBlock(channels),

            MSResidualBlock(channels),

            nn.Conv2d(
                channels,
                hs_bands,
                1
            )

        )


    def forward(self,latent):

        return self.decoder(latent)

# class HighFreq(nn.Module):

#     def __init__(self,channels):

#         super().__init__()

#         self.avg = nn.AvgPool2d(3,1,padding=1)

#     def forward(self,x):

#         return x-self.avg(x)

# ================= 3. 主模型架构 =================

class DC_STSF(nn.Module):
    def __init__(self, ms_bands=4, hs_bands=151, out_channels=151):
        super(DC_STSF, self).__init__()

        # 1. 统一内容编码器 (CE)
        self.content_encoder = nn.Sequential(
            conv_block(ms_bands, out_channels),
            MSResidualBlock(out_channels),
            MSResidualBlock(out_channels)
        )

        self.content_encoder1 = nn.Sequential(
            conv_block(hs_bands, out_channels),
            MSResidualBlock(out_channels),
            MSResidualBlock(out_channels)
        )
        self.hs_encoder = nn.Sequential(

            conv_block(
                hs_bands,
                out_channels
            ),

            MSResidualBlock(out_channels),

            MSResidualBlock(out_channels)

        )


        # 3. 功能模块
        # self.stpn = STPN(out_channels, out_channels)
        self.tlem = TLEM(out_channels)
        self.hfdr = HFDR(hs_bands)
        self.refine = nn.Conv2d(out_channels, hs_bands, 1)

        self.temporal_model = TemporalModeling(out_channels)
        self.spectral_model = SpectralModeling(out_channels)
        self.latent_fusion = LatentStateFusion(out_channels)
        self.decoder = LatentDecoder(
            out_channels,
            hs_bands
        )
        self.detail = nn.Sequential(
            nn.Conv2d(ms_bands, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        
                        

    def forward(self, MS1, HS1, MS2):
        # 对齐 HS1 分辨率
        if HS1.shape[-1] != MS1.shape[-1]:
            HS1 = F.interpolate(HS1, size=MS1.shape[-2:], mode='bicubic', align_corners=True)

        # ---- T1 STSF ----
        fc_m1 = self.content_encoder(MS1)
        fc_h1 = self.hs_encoder(HS1)
        fc_m2 = self.content_encoder(MS2)

        # fs_m1 = self.style_encoder(MS1)
        # fs_h1 = self.style_encoder1(HS1)
        
        # --------------------------
        # 2. 空谱隐状态构建
        # --------------------------
        spectral_prior = self.spectral_model(
            fc_m1,
            fc_h1
        )
        # latent0 = self.latent_fusion(
        #     fc_m1,
        #     fc_h1,
        #     spectral_prior
        # )
        latent1 = self.latent_fusion(
            fc_m1,
            spectral_prior
        )
        # --------------------------
        # 3. T1 HS重建
        # --------------------------

        F1 = self.decoder(
            latent1
        )



        # --------------------------
        # 4. 时间状态演化
        # --------------------------

        delta_t = self.temporal_model(
            fc_m1,
            fc_m2
        )


        latent2,change_map = self.tlem(
            latent1,
            delta_t
        )



        # --------------------------
        # 5. T2 HS生成
        # --------------------------

        F2_temp = self.decoder(
            latent2
        ) #151


        # --------------------------
        # 6. 高频细节融合
        # --------------------------

        F2 = self.hfdr(
            F2_temp,
            F1,
            change_map
        )


        return F2,F1,latent1,latent2