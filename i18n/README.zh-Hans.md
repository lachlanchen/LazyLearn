[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# LazyPhysics and Chemistry

[![Site](https://img.shields.io/badge/website-learn.lazying.art-0a7ea4?style=for-the-badge&logo=githubpages&logoColor=white)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Learning-16a34a?style=for-the-badge&logo=target&logoColor=white)
![Repo Type](https://img.shields.io/badge/Repo-Type-Mixed%20Format-6b7280?style=for-the-badge)
![Docs](https://img.shields.io/badge/Docs-Static%20Microsite-0ea5e9?style=for-the-badge&logo=markdown&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-11-0f766e?style=for-the-badge&logo=googletranslate&logoColor=white)

## 📌 概览

| 关注点     | 本仓库功能                       |
| ---------- | -------------------------------- |
| 工作流类型 | 可复现的物理与化学学习工作区     |
| 输出内容   | 脚本、笔记本、生成图表和静态文档 |
| 协作模式   | 根目录实验 + 公开站点发布        |
| 翻译覆盖   | `i18n/` 下的 README 镜像文件     |

LazyPhysics and Chemistry 是 **LazyLearn** 的代码 + 笔记本部分：一个有意放慢节奏、强调可落地实践的物理与化学学习日志。持续更新的笔记、阶段性成果与待办会发布在 [learn.lazying.art](https://learn.lazying.art)（通过本仓库 `docs/` 提供服务），可复现的实验产物则保留在本仓库中，让实验始终有一个确定的可复现归属。

## 概览 🧭

### LazyLearn

- **主站：** [learn.lazying.art](https://learn.lazying.art) - 面向公开的站点，包含每周重点、待办清单和高亮内容。
- **事实来源：** 站点链接到的内容都位于 `examples/`、`comp_physics/`、`comp_physics_python/`、`multiwfn/` 或 `figures/`。
- **更新流程：** 先发布代码/笔记本，必要时重生成图表，再向 `docs/` 添加记录，使站点始终反映最新工作。

这个仓库是故意采用混合格式的，不是单一打包应用。它把可执行脚本、笔记本、参考资料和静态文档站点放在一个版本化工作区里。

## 特性 ✨

- 可在普通笔记本电脑上运行的量子示例脚本（QAOA + VQE）。
- 计算物理笔记本与辅助求解器（例如基于 Numerov 的流程）。
- 按章节提供的教科书计算物理程序 Python 移植版。
- 用于本地量子化学后处理参考的 Multiwfn 源码与说明文档包。
- 报告/幻灯片的版本化生成图（`figures/`）。
- `i18n/` 下内建多语言 README。
- `docs/` 中的静态微站（自定义域名：`learn.lazying.art`）。

## 项目结构 🗂️

### 结构说明

| 路径                   | 用途                                                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `examples/`            | 基于 Qiskit 或 PennyLane 的聚焦型 Python 脚本（QAOA + VQE）。                                                                                    |
| `comp_physics/`        | 计算物理笔记本、`numerov.py` 等辅助脚本，以及配套的数据/图像。                                                                                   |
| `comp_physics_python/` | Jos Thijssen 的 _Computational Physics_ 程序 Python 移植版，按章节组织（见 [comp_physics_python/README.md](../comp_physics_python/README.md)）。 |
| `multiwfn/`            | Multiwfn 3.8 开发者源码包及手册，供本地参考。                                                                                                    |
| `figures/`             | 报告/幻灯片和 README 使用的静态 PNG/SVG 输出。                                                                                                   |
| `figs/`                | 标志与横幅资源。                                                                                                                                 |
| `docs/`                | LazyLearn 微站内容（由 GitHub Pages 或任意静态托管提供）。                                                                                       |
| `i18n/`                | 本地化 README 文件。                                                                                                                             |

目录示例：

```text
LazyLearn/
|- README.md
|- docs/
|- i18n/
|- examples/
|- comp_physics/
|- comp_physics_python/
|- multiwfn/
|- figures/
|- figs/
|- Gaussian -> ../Gaussian/ (symlink)
|- ComputationalPhysics -> ../ComputationalPhysics/ (symlink)
|- leonardsusskind -> ../leonardsusskind/ (symlink)
`- the_theoretical_minimum -> ../the_theoretical_minimum/ (symlink)
```

> [!IMPORTANT]
> 部分顶层条目是指向仓库外目录的符号链接。编辑这些路径会影响外部目标。

## 前置条件 🧰

| 依赖                            | 说明                               |
| ------------------------------- | ---------------------------------- |
| Python 3.x                      | 根目录脚本和大多数笔记本工作所需。 |
| `pip`（或 Conda）               | 包/环境管理。                      |
| Jupyter Lab/Notebook（可选）    | 笔记本工作流需要。                 |
| Gaussian 16 + GaussView（可选） | Gaussian 流程需要。                |

## 安装 ⚙️

### 最小 Python 环境（根目录示例）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

### ✅ 快速设置清单

| 步骤 | 命令                                            | 目标                     |
| ---- | ----------------------------------------------- | ------------------------ |
| 1    | `python -m venv .venv`                          | 创建隔离环境             |
| 2    | `source .venv/bin/activate`（或系统等价命令）   | 避免依赖冲突             |
| 3    | `pip install --upgrade pip`                     | 确保包管理工具是最新的   |
| 4    | `pip install qiskit pennylane numpy matplotlib` | 安装核心实验依赖         |
| 5    | 运行 `examples/` 中任意脚本                     | 验证安装是否端到端可运行 |

`comp_physics/` 内的 Jupyter 笔记本与上述环境同用。启动方式：

```bash
jupyter lab
# or
jupyter notebook
```

### `comp_physics_python/` 的可选章节移植依赖

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## 使用 🚀

### 示例工作流

- **QAOA with Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

无需 Aer 依赖，使用纯 statevector 后端。

- **QAOA with PennyLane**

```bash
python examples/qaoa_pennylane_maxcut.py
```

使用 `default.qubit`。

- **VQE for H2**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

可复现 `figures/pennylane_h2_vqe_convergence.png`。

所有脚本都会记录中间指标，便于复用图表或扩展到新分子/新图问题。

## 计算物理笔记本 📓

`comp_physics/` 目录是工作笔记的镜像：

- `comp_physics_textbook_code/` - 从笔记本提炼出的可复用例程。
- 独立笔记本，如 `chapter1.ipynb`、`chapter2.ipynb`、`numerov.ipynb` 和 `numpy_1ddft.ipynb`。
- 主题目录（`bosonscattering/`、`lensless/`、`lightscattering/` 等）按实验保存数据和辅助脚本。

如需额外依赖，请将其记录在 `comp_physics/environments.yaml`。

## 教科书代码移植 📚

`comp_physics_python/` 是一本由经典 _Computational Physics_ Fortran 程序逐步完善的 Python 译本。章节示例：

- `ch4/`：Hartree-Fock 示例。
- `ch8/`：分子动力学求解器。
- `ch10/`：Monte Carlo 采样器。

完整章节覆盖和 CLI 命令见 [comp_physics_python/README.md](../comp_physics_python/README.md)。

## Multiwfn 参考 🔬

`multiwfn/` 存放 `Multiwfn_3.8_dev_src_Linux`、PDF 手册和快速入门指南。仓库未提交编译后二进制文件。

## 图像 🖼️

生成的 PNG/SVG 资源位于 `figures/`，用于使产物与生成脚本/笔记本同步版本化。

## 配置 🛠️

### Python 与笔记本

- 根目录脚本默认使用上面展示的虚拟环境。
- 笔记本的环境细节分散在子项目文档中；仓库根目录目前没有统一的 lockfile。

### Gaussian 运行器（符号链接路径）

`Gaussian/run_gaussian.sh` 支持：

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

行为：

- 在输入文件旁边生成 `<basename>.log`。
- 若设置 `GAUSS_SCRDIR` 则使用该变量；否则默认使用 `~/gaussian/scr`。
- 检测输入里的 `%chk=...`；若 checkpoint 存在则 GaussView 打开 `.chk`，否则打开 `.log`。
- 若可用，优先使用 `~/gaussian/gv/gview_safe.sh`，其次使用 `gview.sh`。

推荐的 GaussView 包装脚本：

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## 开发说明 🧪

### 🎬 导航地图

可用作日常工作的起点：

| 区域         | 起始位置               |
| ------------ | ---------------------- |
| 量子演示     | `examples/`            |
| 物理笔记本   | `comp_physics/`        |
| 课本移植     | `comp_physics_python/` |
| 量子化学工具 | `multiwfn/`            |
| 已发布产物   | `docs/`                |
| 图与插图     | `figures/`、`figs/`    |

### 版本控制说明

- `.gitignore` 会忽略大体积路径，包括 `books/`、外部符号链接目标（`Gaussian`、`ComputationalPhysics`、`leonardsusskind`、`the_theoretical_minimum`）以及本地产物（如 `*.chk`）。
- 为了保持克隆和更新流程轻量，建议将贡献集中在已跟踪目录。
- 更新网站时流程为：编辑 `docs/`、本地预览，再提交推送。

本地预览文档：

```bash
python -m http.server --directory docs
```

`docs/CNAME` 配置为 `learn.lazying.art`。

## 故障排查 🩺

- Gaussian 成功标志：在 `.log` 文件末尾附近出现 `Normal termination of Gaussian`。
- 如果 GaussView 在 Wayland/远程会话下启动失败，请改用 `gview_safe.sh` 并显式传入 `--gview`。
- 若出现 Gaussian scratch 错误，请检查 `GAUSS_SCRDIR` 的磁盘空间和权限。
- 若笔记本依赖漂移，请以各子项目 README 作为权威来源，并在共享前将缺失包记录到环境文件。
- 当前仓库中的 `comp_physics/environments.yaml` 似乎是占位状态；在修正前请依赖显式安装命令。

## 路线图 🛣️

- 持续扩展 `comp_physics_python/` 章节覆盖（传输矩阵、DMC/PIMC、FEM 等）。
- 统一脚本和笔记本的输出/绘图规范。
- 为关键示例增加轻量且可复现的验证检查。
- 让 `docs/` 与多语言 README 持续与新实验保持同步。

## 贡献 🤝

欢迎提交 Issue 和 Pull Request，特别是：

- 数值正确性校验与复现性改进。
- 更完善的笔记本/脚本环境规范。
- 更多教科书章节移植与 CLI 细化。
- `i18n/` 中跨语言文档的可读性提升。

在提交较大内容更新前，请将生成图表保存在 `figures/`，并确保命令可在仓库根目录运行（除非文档另有说明）。

## 许可证 📄

仓库根目录当前尚不存在 `LICENSE` 文件。正式添加许可前，请将使用/再分发权利视为未明确；在复用较大内容前请先向维护者确认。

## ❤️ Support

| Donate                                                                                                                                                                                                                                                                                                                                                     | PayPal                                                                                                                                                                                                                                                                                                                                                          | Stripe                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
