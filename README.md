# Digital Twin Studio

**简体中文** | [English](README.en.md)

面向真实三维场景的交互式、波束感知无线电地图展示软件，基于 [Sionna RT](https://github.com/NVlabs/sionna-rt) 和 [sionna-rt-gui](https://github.com/NVlabs/sionna-rt-gui) 开发。

![Radio Map 演示](media/radio-map-preview.png)

[观看演示视频](media/radio-map-demo.mp4) · [快速开始](docs/QUICKSTART.md) · [场景来源](docs/SCENE_SOURCES.md) · [版权与许可](THIRD_PARTY_NOTICES.md)

本项目用于科研展示，尚不是经过现场校准的网络规划产品。 **目前尚未接入课题组训练模型的输出**；覆盖计算来自 Sionna RT，电磁材料采用演示假设。

## 功能

- 加载香港理工大学校园、旺角城区和锦田乡村三个场景。
- 放置基站，通过 **Aim** 点击目标，或从球面上的 294 个离散 beam 中直接选择。
- 选择 beam 后，自动计算六个高度的水平 RSS 切片。
- 在两列结果面板查看全部切片，在三维场景中叠加六层，或单独展示一个高度。
- 切换 **Fast** 缓存光照与 **Original** 视觉光追；视角交互时使用快速预览。
- 提供独立的接收机路线工作区；随附视频只展示 Radio Map 部分。

## 当前默认参数

| 参数 | 默认值 |
|---|---|
| 水平网格 | 3 m × 3 m |
| 蒙特卡洛预算 | 每个高度、每个所选基站发射 8,000,000 samples |
| 高度 | 根据场景几何确定六个均匀的模型 Z 高度 |
| 传播机制 | 直达与镜面反射，最大深度 5 |
| 计算完成后的展示 | 六个高度全部叠加 |
| 视觉渲染 | 默认 Fast；Original 最多累计 512 samples/pixel |

**8M 不是每个格子的采样数。** 多批线性路径增益估计按样本数加权合并，然后转换为 RSS（dBm）。它与视觉渲染采样数是两套不同的参数；达到预算不等于已证明收敛或准确。

六层切片是三维场的离散采样，并非连续求解的完整体积。高度采用场景坐标，不能统一解释为距当地地面的高度。当前开阔天空掩膜排除顶层表面以下的格子，包括桥下空间。深蓝色可能包括没有得到正功率估计的格子，不代表真实物理信号为零。

## Windows 安装与启动

当前本地验证环境为 Windows、64 位 Python 3.12、支持 CUDA 的 NVIDIA 显卡及有效驱动。其他平台与纯 CPU 模式尚未针对本定制版本验证。

```powershell
git clone https://github.com/stevenyyzhang/Digital-Twin-Software.git
cd Digital-Twin-Software
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
.\start.bat
```

仓库不附带 Python、显卡驱动或虚拟环境。首次启动会生成本地光照缓存，耗时可能更长。使用说明与故障排查见 [QUICKSTART.md](docs/QUICKSTART.md)。

## 文件结构

```text
sionna-rt-gui/            应用源码与相关测试
polyu-stylized/           校园必要资产、OSM 来源与数据许可说明
mong-kok/                城市必要资产与数据许可说明
kam-tin/                 乡村必要资产与数据许可说明
docs/                    使用说明、场景来源与演示录制说明
media/                   Radio Map 视频与预览图
tools/                   文件检查与交互录屏工具
polyu-stylized-config.yaml
requirements-tested.txt
setup.ps1
start.ps1
start.bat
```

只保留被运行时引用的场景资源，以及数据许可核查所需的 OSM 来源文件。不包含 Blender 中间工程、官方原始模型压缩包、历史版本、缓存、开发截图和 Python 运行环境。仓库提供可直接加载的场景资产，不包含从官方原始下载完整重建场景的流程。

## 来源与许可

上游应用代码采用 Apache-2.0，原许可证及版权声明已保留，见 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。 **软件许可证不覆盖场景模型数据**；每个场景都有独立的 `ASSET_LICENSE.md`。官方来源允许附条件再分发；使用这些资产时请保留来源、版权及修改说明，并遵守对应数据条款。

建筑几何及部分地形影像来自香港特别行政区政府地政总署，经 CSDI / Open3Dhk 发布。校园辅助地面、道路、绿地和水体包含 © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) 的 ODbL 数据。原始 OSM 提取文件及完整衍生辅助网格已附带，许可范围见 [校园数据声明](polyu-stylized/ASSET_LICENSE.md)。

本项目对源数据进行了坐标转换、区域裁剪、网格减面、材质分组、纹理转换、重新配色、窗带样式和显示法线处理。各场景的具体处理见其 `ASSET_LICENSE.md`。

处理后的资产是本项目制作的展示版本，不是官方原版或经测绘核实的外墙材质。原始数据版权仍归原权利人；本项目不代表政府或大学的认可。具体来源、使用条款与修改范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 及各场景的 `ASSET_LICENSE.md`。
