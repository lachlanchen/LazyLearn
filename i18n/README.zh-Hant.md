[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)


# LazyPhysics and Chemistry

[![Site](https://img.shields.io/badge/website-learn.lazying.art-0a7ea4?style=for-the-badge&logo=githubpages&logoColor=white)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Learning-16a34a?style=for-the-badge&logo=target&logoColor=white)
![Repo Type](https://img.shields.io/badge/Repo-Type-Mixed%20Format-6b7280?style=for-the-badge)
![Docs](https://img.shields.io/badge/Docs-Static%20Microsite-0ea5e9?style=for-the-badge&logo=markdown&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-11-0f766e?style=for-the-badge&logo=googletranslate&logoColor=white)

## 📌 概覽

| 目標 | 本倉庫用途 |
| --- | --- |
| 工作流程類型 | 可重現的物理＋化學學習工作區 |
| 交付內容 | 腳本、筆記本、產生圖表與靜態文件 |
| 協作模式 | 根目錄實驗 + 公開站點發布 |
| 翻譯覆蓋 | `i18n/` 內的 README 鏡像檔 |

LazyPhysics and Chemistry 是 **LazyLearn** 的程式碼與筆記本部分：一份有意放慢節奏、著重實際應用的物理與化學學習誌。持續更新的筆記、成果與待辦會發布在 [learn.lazying.art](https://learn.lazying.art)（本倉庫 `docs/` 提供服務），而可執行成果則保留在此，讓每個實驗都有可重現的歸處。

## 概覽 🧭

### LazyLearn

- **主站：** [learn.lazying.art](https://learn.lazying.art) - 提供每週焦點、待辦清單與重點整理的公開網站。
- **可信來源：** 站上連結的內容都位於 `examples/`、`comp_physics/`、`comp_physics_python/`、`multiwfn/` 或 `figures/`。
- **更新流程：** 先上傳程式碼與筆記本，必要時重新產圖，再將條目加入 `docs/`，讓網站即時反映最新工作。

這個倉庫刻意採用混合格式，而非單一可打包應用。它將可執行腳本、筆記本、參考資料與靜態文件站放在同一個有版本管理的工作區中。

## 功能 ✨

- 可在一般筆電上執行的量子示範腳本（QAOA + VQE）。
- 計算物理筆記本與輔助解題器（例如基於 Numerov 的流程）。
- 教科書計算物理程式逐章節的 Python 移植版。
- 為本機量子化學後處理參考提供的 Multiwfn 原始碼與手冊套件。
- 報告／投影片用的版本化產生圖（`figures/`）。
- 內建 `i18n/` 下的多語言 README。
- `docs/` 中的靜態微型網站（自訂網域：`learn.lazying.art`）。

## 專案結構 🗂️

### 目錄內容

| 路徑 | 用途 |
| --- | --- |
| `examples/` | 使用 Qiskit 或 PennyLane 的精簡 Python 腳本（QAOA + VQE）。 |
| `comp_physics/` | 計算物理筆記本、輔助腳本（如 `numerov.py`）以及配套資料與圖表。 |
| `comp_physics_python/` | Jos Thijssen《Computational Physics》教材程式的 Python 版，按章節整理（見 [comp_physics_python/README.md](../comp_physics_python/README.md)）。 |
| `multiwfn/` | Multiwfn 3.8 開發者原始碼套件與操作手冊，供本機參考。 |
| `figures/` | 報告／投影片與 README 使用的靜態 PNG/SVG 輸出。 |
| `figs/` | 標誌與橫幅素材。 |
| `docs/` | LazyLearn 微網站內容（由 GitHub Pages 或任何靜態主機提供）。 |
| `i18n/` | 本地化 README 檔案。 |

目錄範例：

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
> 部分頂層目錄是指向倉庫外部的符號連結。編輯這些路徑下內容會同步影響外部目標。

## 前置條件 🧰

| 需求 | 說明 |
| --- | --- |
| Python 3.x | 根目錄腳本與大多數筆記本作業所需。 |
| `pip`（或 Conda） | 套件/環境管理。 |
| Jupyter Lab/Notebook（可選） | 筆記本工作流程所需。 |
| Gaussian 16 + GaussView（可選） | Gaussian 工作流程所需。 |

## 安裝 ⚙️

### 最小 Python 環境（根目錄範例）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

### ✅ 快速設定清單

| 步驟 | 指令 | 目的 |
| --- | --- | --- |
| 1 | `python -m venv .venv` | 建立獨立環境 |
| 2 | `source .venv/bin/activate`（或作業系統等效指令） | 避免套件衝突 |
| 3 | `pip install --upgrade pip` | 確保套件工具為最新 |
| 4 | `pip install qiskit pennylane numpy matplotlib` | 安裝核心實驗套件 |
| 5 | 執行 `examples/` 中任一腳本 | 驗證安裝流程是否通順 |

`comp_physics/` 內的 Jupyter 筆記本與同一個環境共用。啟動方式：

```bash
jupyter lab
# or
jupyter notebook
```

### `comp_physics_python/` 的選配章節移植套件

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## 使用 🚀

### 範例工作流程

- **QAOA with Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

無需 Aer 依賴，使用純 statevector 後端。

- **QAOA with PennyLane**

```bash
python examples/qaoa_pennylane_maxcut.py
```

使用 `default.qubit`。

- **VQE for H2**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

可重現 `figures/pennylane_h2_vqe_convergence.png`。

所有腳本都會記錄中間指標，方便你重複使用圖表或擴展到新分子/新圖論問題。

## 計算物理筆記本 📓

`comp_physics/` 目錄是工作筆記的鏡像：

- `comp_physics_textbook_code/` - 從筆記本提煉出的可重複使用程式流程。
- 獨立筆記本，如 `chapter1.ipynb`、`chapter2.ipynb`、`numerov.ipynb` 與 `numpy_1ddft.ipynb`。
- 主題資料夾（`bosonscattering/`、`lensless/`、`lightscattering/` 等）各自含有對應實驗的資料與輔助工具。

若需額外套件，請記錄於 `comp_physics/environments.yaml`。

## 教材程式移植 📚

`comp_physics_python/` 是正在建立的經典 *Computational Physics* Fortran 程式 Python 譯本。章節對照示例：

- `ch4/`：Hartree-Fock 示例。
- `ch8/`：分子動力學求解器。
- `ch10/`：蒙地卡羅取樣器。

完整章節涵蓋與 CLI 指令請參見 [comp_physics_python/README.md](../comp_physics_python/README.md)。

## Multiwfn 參考 🔬

`multiwfn/` 包含 `Multiwfn_3.8_dev_src_Linux`、PDF 手冊與快速入門指南。本倉庫未提交編譯後的二進位檔。

## 圖表 🖼️

產生出的 PNG/SVG 資產存放於 `figures/`，讓輸出與對應腳本／筆記本同步進行版本管理。

## 組態設定 🛠️

### Python 與筆記本

- 根目錄腳本預設使用上面展示的虛擬環境。
- 筆記本環境細節散見於各子專案文件；目前 repo 根目錄沒有統一的 lockfile。

### Gaussian 執行器（符號連結路徑）

`Gaussian/run_gaussian.sh` 支援：

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

行為：

- 在輸入檔旁建立 `<basename>.log`。
- 若有設定 `GAUSS_SCRDIR` 則使用該值，否則預設為 `~/gaussian/scr`。
- 偵測輸入中的 `%chk=...`；若 checkpoint 存在，GaussView 會打開 `.chk`，否則打開 `.log`。
- 若可用，優先使用 `~/gaussian/gv/gview_safe.sh`，其次 `gview.sh`。

建議的 GaussView 包裝腳本：

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## 開發筆記 🧪

### 🎬 導覽地圖

可作為日常工作的啟動點：

| 區域 | 從這裡開始 |
| --- | --- |
| 量子示範 | `examples/` |
| 物理筆記本 | `comp_physics/` |
| 教材移植 | `comp_physics_python/` |
| 量子化學工具 | `multiwfn/` |
| 已發布輸出 | `docs/` |
| 圖示與插圖 | `figures/`、`figs/` |

### 版本控制備註

- `.gitignore` 忽略大型路徑，包括 `books/`、外部符號連結目標（`Gaussian`、`ComputationalPhysics`、`leonardsusskind`、`the_theoretical_minimum`）以及本地產物（如 `*.chk`）。
- 為了保持 clone / 更新流程輕量，請將貢獻聚焦在追蹤中的資料夾。
- 網站更新流程：編輯 `docs/`、本機預覽，再推送。

本機文件預覽：

```bash
python -m http.server --directory docs
```

`docs/CNAME` 已設為 `learn.lazying.art`。

## 故障排除 🩺

- Gaussian 成功標準：`Normal termination of Gaussian` 出現在 `.log` 檔尾端附近。
- 若 GaussView 在 Wayland / 遠端工作階段下啟動失敗，請使用 `gview_safe.sh` 並明確傳入 `--gview`。
- 遇到 Gaussian scratch 錯誤時，請檢查 `GAUSS_SCRDIR` 的磁碟空間與權限。
- 若筆記本套件版本偏移，請以各子專案 README 為準，並在共享前先將缺失套件寫入環境檔。
- `comp_physics/environments.yaml` 目前在此倉庫版本中似乎還是預留檔；在修正前請依賴明確安裝指令。

## 路線圖 🛣️

- 持續擴充 `comp_physics_python/` 的章節覆蓋（轉移矩陣、DMC/PIMC、FEM 等）。
- 統一腳本與筆記本輸出／繪圖規範。



## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
