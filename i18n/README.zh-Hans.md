[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# LazyPhysics and Chemistry

[![Site](https://img.shields.io/badge/site-learn.lazying.art-0a7ea4)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-active%20learning-22c55e)
![Repo Type](https://img.shields.io/badge/repo-mixed--format-6b7280)
![Docs](https://img.shields.io/badge/docs-static%20microsite-0ea5e9)

LazyPhysics and Chemistry 是 **LazyLearn** 的“代码 + 笔记本”部分：一个刻意放慢节奏、强调实践的物理与化学学习日志。持续更新的笔记、阶段成果和 TODO 发布在 [learn.lazying.art](https://learn.lazying.art)（由本仓库 `docs/` 提供），而可运行的实验产物保留在本仓库中，确保实验始终可复现。

## 概览 🧭

### LazyLearn

- **主站：** [learn.lazying.art](https://learn.lazying.art) - 面向公开的站点，包含每周重点、待办列表和亮点。
- **事实来源：** 站点链接到的所有内容都位于 `examples/`、`comp_physics/`、`comp_physics_python/`、`multiwfn/` 或 `figures/`。
- **更新流程：** 先提交代码/笔记本，必要时重新生成图，再将更新写入 `docs/`，让站点反映最新进展。

本仓库刻意采用混合格式，而非单一打包应用。它将可执行脚本、笔记本、参考资料和静态文档站点统一纳入同一个版本化工作区。

## 特性 ✨

- 可在普通笔记本电脑上运行的量子示例脚本（QAOA + VQE）。
- 计算物理笔记本和辅助求解器（如基于 Numerov 的流程）。
- 按章节组织的教科书计算物理程序 Python 移植版。
- 供本地量子化学后处理参考的 Multiwfn 源码/手册打包。
- 用于报告/幻灯片的版本化生成图（`figures/`）。
- 内置多语言 README（位于 `i18n/`）。
- `docs/` 中的静态微站（自定义域名：`learn.lazying.art`）。

## 项目结构 🗂️

### 目录说明

| Path | Purpose |
| --- | --- |
| `examples/` | 依赖 Qiskit 或 PennyLane 的聚焦型 Python 脚本（QAOA + VQE）。 |
| `comp_physics/` | 计算物理笔记本、`numerov.py` 等辅助脚本，以及配套数据/图像。 |
| `comp_physics_python/` | Jos Thijssen 的 *Computational Physics* 程序 Python 移植版，按章节组织（见 [comp_physics_python/README.md](../comp_physics_python/README.md)）。 |
| `multiwfn/` | Multiwfn 3.8 开发者源码包及手册，供本地参考。 |
| `figures/` | 报告/幻灯片和 README 使用的静态 PNG/SVG 输出。 |
| `figs/` | Logo 与横幅素材。 |
| `docs/` | LazyLearn 微站内容（可由 GitHub Pages 或任意静态托管服务提供）。 |
| `i18n/` | 本地化 README 文件。 |

代表性布局：

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
> 若干顶层条目是指向仓库外目录的符号链接。在这些路径下编辑会影响外部目标目录。

## 前置要求 🧰

| Requirement | Notes |
| --- | --- |
| Python 3.x | 根目录脚本和大部分笔记本工作必需。 |
| `pip` (or Conda) | 用于包/环境管理。 |
| Jupyter Lab/Notebook (optional) | 笔记本工作流可选依赖。 |
| Gaussian 16 + GaussView (optional) | Gaussian 工作流可选依赖。 |

## 安装 ⚙️

### 最小 Python 环境（根目录示例）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

`comp_physics/` 中的 Jupyter 笔记本使用同一环境。启动方式：

```bash
jupyter lab
# or
jupyter notebook
```

### 可选章节移植依赖（`comp_physics_python/`）

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

无需 Aer 依赖；使用纯 statevector 后端。

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

所有脚本都会记录中间指标，便于复用绘图或扩展到新分子/新图问题。

## 计算物理笔记本 📓

`comp_physics/` 目录对应工作笔记：

- `comp_physics_textbook_code/` - 从笔记本中提炼的可复用例程。
- 独立笔记本，如 `chapter1.ipynb`、`chapter2.ipynb`、`numerov.ipynb`、`numpy_1ddft.ipynb`。
- 主题文件夹（`bosonscattering/`、`lensless/`、`lightscattering/` 等），按实验组织数据与辅助脚本。

如需额外依赖，请将其记录在 `comp_physics/environments.yaml`。

## 教科书代码移植 📚

`comp_physics_python/` 正在逐步完成经典 *Computational Physics* Fortran 程序的 Python 翻译。章节示例：

- `ch4/`: Hartree-Fock 示例。
- `ch8/`: 分子动力学求解器。
- `ch10/`: Monte Carlo 采样器。

完整章节覆盖与 CLI 命令见 [comp_physics_python/README.md](../comp_physics_python/README.md)。

## Multiwfn 参考 🔬

`multiwfn/` 保留 `Multiwfn_3.8_dev_src_Linux`、PDF 手册和快速入门指南。仓库不包含编译后二进制。

## 图像 🖼️

生成的 PNG/SVG 资源放在 `figures/`，使输出与产出脚本/笔记本一同版本化。

## 配置 🛠️

### Python 与笔记本

- 根目录脚本默认使用上文展示的虚拟环境。
- 笔记本环境细节分散在各子项目文档中；当前仓库根目录没有统一 lockfile。

### Gaussian 运行器（符号链接路径）

`Gaussian/run_gaussian.sh` 支持：

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

行为：

- 在输入文件旁生成 `<basename>.log`。
- 若设置 `GAUSS_SCRDIR` 则使用该目录；否则默认 `~/gaussian/scr`。
- 会检测输入中的 `%chk=...`；若 checkpoint 存在，GaussView 打开 `.chk`，否则打开 `.log`。
- 若可用，优先使用 `~/gaussian/gv/gview_safe.sh`，其次 `gview.sh`。

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

### 版本控制说明

- 大体积路径通过 `.gitignore` 忽略，包括 `books/`、外部符号链接目标（`Gaussian`、`ComputationalPhysics`、`leonardsusskind`、`the_theoretical_minimum`）和本地产物（如 `*.chk`）。
- 贡献建议聚焦于已跟踪目录，以保持克隆/更新流程轻量。
- 更新网站时：编辑 `docs/`，本地预览后再推送。

本地预览文档：

```bash
python -m http.server --directory docs
```

`docs/CNAME` 已配置为 `learn.lazying.art`。

## 故障排查 🩺

- Gaussian 成功判据：`.log` 末尾附近出现 `Normal termination of Gaussian`。
- 若 GaussView 在 Wayland/远程会话中启动失败，请使用 `gview_safe.sh` 并显式传入 `--gview`。
- 若出现 Gaussian scratch 错误，请检查 `GAUSS_SCRDIR` 的磁盘空间与权限。
- 若笔记本依赖漂移，请以各子项目 README 为准，并在共享前把缺失包记录到环境文件。
- 当前仓库中的 `comp_physics/environments.yaml` 看起来仍是占位状态；修正前请依赖显式安装命令。

## 路线图 🛣️

- 继续扩展 `comp_physics_python/` 的章节覆盖（传输矩阵、DMC/PIMC、FEM 等）。
- 统一脚本与笔记本的输出/绘图规范。
- 为关键示例增加轻量且可复现的验证检查。
- 让 `docs/` 与多语言 README 持续与新实验保持同步。

## 贡献 🤝

欢迎提交 Issue 和 Pull Request，特别是：

- 数值正确性检查与可复现性改进。
- 更完善的笔记本/脚本环境规范。
- 新增教科书章节移植与 CLI 细节改进。
- `i18n/` 下跨语言文档清晰度提升。

提交较大内容更新前，请将生成图表放在 `figures/`，并确保命令可在仓库根目录运行（除非另有文档说明）。

## 支持 LazyLearn ❤️

支持 LazyLearn 可帮助实验、文档与开源工具持续推进：

- 覆盖公开演示与笔记本的托管/推理/存储成本。
- 资助 EchoMind、LazyEdit 与本仓库量子/物理工具的专注开发周。
- 原型化光学 + 可穿戴项目（IdeasGlass、LightMind），反哺后续章节。
- 赞助学生、社区实验室与创作者的免费部署。

### 捐助

<div align="center">
<table style="margin:0 auto; text-align:center; border-collapse:collapse;">
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://chat.lazying.art/donate">https://chat.lazying.art/donate</a>
    </td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://chat.lazying.art/donate"><img src="../figures/donate_button.svg" alt="Donate" height="44"></a>
    </td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://paypal.me/RongzhouChen">
        <img src="https://img.shields.io/badge/PayPal-Donate-003087?logo=paypal&logoColor=white" alt="Donate with PayPal">
      </a>
    </td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400">
        <img src="https://img.shields.io/badge/Stripe-Donate-635bff?logo=stripe&logoColor=white" alt="Donate with Stripe">
      </a>
    </td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><strong>WeChat</strong></td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><strong>Alipay</strong></td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><img alt="WeChat QR" src="../figures/donate_wechat.png" width="240"/></td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><img alt="Alipay QR" src="../figures/donate_alipay.png" width="240"/></td>
  </tr>
</table>
</div>

**支援 / Donate**

- ご支援は研究・開発と運用の継続に役立ち、より多くのオープンなプロジェクトを皆さんに届ける力になります。
- 你的支持将用于研发与运维，帮助我持续公开分享更多项目与改进。
- Your support sustains my research, development, and ops so I can keep sharing more open projects and improvements.

## 许可证 📄

当前本仓库根目录尚不存在 `LICENSE` 文件。在新增许可证之前，请将使用/再分发权利视为未明确；如需复用大量内容，请先向维护者确认。
