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

LazyPhysics and Chemistry 是 **LazyLearn** 中程式碼與筆記的那一半：以刻意放慢節奏、重視實作的方式記錄物理與化學學習。持續更新的筆記、進展與 TODO 發布在 [learn.lazying.art](https://learn.lazying.art)（由本倉庫的 `docs/` 提供），可執行的產出則保留在此，讓每次實驗都能有可重現的落點。

## 概覽 🧭

### LazyLearn

- **主要入口：** [learn.lazying.art](https://learn.lazying.art) - 對外網站，整理每週重點、待辦清單與精選內容。
- **事實來源：** 網站所連結的內容皆位於 `examples/`、`comp_physics/`、`comp_physics_python/`、`multiwfn/` 或 `figures/`。
- **更新流程：** 先提交程式碼/筆記，需要時重產圖表，再於 `docs/` 補上條目，讓網站反映最新進度。

此倉庫刻意採用混合格式，而非單一封裝應用。它把可執行腳本、筆記本、參考資料與靜態文件網站放在同一個版本化工作區中。

## 功能特色 ✨

- 可在一般筆電執行的量子範例腳本（QAOA + VQE）。
- 計算物理筆記本與輔助求解器（例如基於 Numerov 的工作流程）。
- 依章節整理的教科書計算物理程式 Python 移植版。
- 供本地量子化學後處理參考的 Multiwfn 原始碼/手冊套件。
- 用於報告與投影片的版本化圖檔輸出（`figures/`）。
- 內建放在 `i18n/` 的多語 README。
- `docs/` 內的靜態微型網站（自訂網域：`learn.lazying.art`）。

## 專案結構 🗂️

### 這裡包含什麼

| Path | Purpose |
| --- | --- |
| `examples/` | 可用 Qiskit 或 PennyLane 執行的聚焦 Python 腳本（QAOA + VQE）。 |
| `comp_physics/` | 計算物理筆記本、`numerov.py` 等輔助腳本，以及相關資料/圖檔。 |
| `comp_physics_python/` | Jos Thijssen《Computational Physics》的 Python 移植，依章節整理（見 [comp_physics_python/README.md](comp_physics_python/README.md)）。 |
| `multiwfn/` | Multiwfn 3.8 開發者原始碼包與手冊，供本地參考。 |
| `figures/` | 報告、投影片與 README 使用的靜態 PNG/SVG 輸出。 |
| `figs/` | Logo 與 banner 素材。 |
| `docs/` | LazyLearn 微型網站內容（可由 GitHub Pages 或任一靜態主機提供）。 |
| `i18n/` | 本地化 README 檔案。 |

代表性目錄：

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
> 多個頂層項目是指向倉庫外部目錄的 symlink。於這些路徑下編輯會影響外部目標。

## 先決條件 🧰

| Requirement | Notes |
| --- | --- |
| Python 3.x | 根目錄腳本與大多數 notebook 工作必需。 |
| `pip`（或 Conda） | 套件/環境管理。 |
| Jupyter Lab/Notebook（可選） | notebook 工作流程需要。 |
| Gaussian 16 + GaussView（可選） | Gaussian 工作流程需要。 |

## 安裝 ⚙️

### 最小 Python 環境（根目錄範例）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

`comp_physics/` 內的 Jupyter notebook 使用同一套環境。啟動方式：

```bash
jupyter lab
# or
jupyter notebook
```

### 可選的章節移植相依套件（`comp_physics_python/`）

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## 使用方式 🚀

### 範例工作流程

- **使用 Qiskit 的 QAOA**

```bash
python examples/qaoa_qiskit_maxcut.py
```

不依賴 Aer；使用純 statevector 後端。

- **使用 PennyLane 的 QAOA**

```bash
python examples/qaoa_pennylane_maxcut.py
```

使用 `default.qubit`。

- **H2 的 VQE**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

會重現 `figures/pennylane_h2_vqe_convergence.png`。

所有腳本都會記錄中間指標，方便重用圖表或擴展到新分子/圖結構。

## 計算物理 notebook 📓

`comp_physics/` 目錄對應工作筆記：

- `comp_physics_textbook_code/` - 從 notebook 抽出的可重用例程。
- 獨立 notebook，例如 `chapter1.ipynb`、`chapter2.ipynb`、`numerov.ipynb` 與 `numpy_1ddft.ipynb`。
- 主題資料夾（`bosonscattering/`、`lensless/`、`lightscattering/` 等），各自包含每項實驗的資料與輔助程式。

若需額外相依套件，請記錄於 `comp_physics/environments.yaml`。

## 教科書程式移植 📚

`comp_physics_python/` 正在持續擴充，將 *Computational Physics* 經典 Fortran 程式翻譯為 Python。章節對應範例：

- `ch4/`：Hartree-Fock 範例。
- `ch8/`：分子動力學求解器。
- `ch10/`：Monte Carlo 取樣器。

完整章節涵蓋與 CLI 指令請參閱 [comp_physics_python/README.md](comp_physics_python/README.md)。

## Multiwfn 參考資料 🔬

`multiwfn/` 保存 `Multiwfn_3.8_dev_src_Linux`、PDF 手冊與快速入門指南。未提交已編譯二進位檔。

## 圖檔輸出 🖼️

產生的 PNG/SVG 素材位於 `figures/`，讓輸出可與對應腳本/notebook 一同版本化。

## 設定 🛠️

### Python 與 notebook

- 根目錄腳本預設使用上方展示的 venv。
- notebook 環境細節分散於各專案文件；目前倉庫根目錄沒有單一 lockfile。

### Gaussian runner（symlink 路徑）

`Gaussian/run_gaussian.sh` 支援：

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

行為：

- 在輸入檔旁寫入 `<basename>.log`。
- 若已設定 `GAUSS_SCRDIR` 則使用該目錄，否則預設 `~/gaussian/scr`。
- 偵測輸入中的 `%chk=...`；若 checkpoint 存在，GaussView 會開啟 `.chk`，否則開 `.log`。
- 若可用，優先使用 `~/gaussian/gv/gview_safe.sh`，其次 `gview.sh`。

建議的 GaussView wrapper：

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## 開發備註 🧪

### 版本控制備註

- `.gitignore` 會忽略大型路徑，包括 `books/`、外部 symlink 目標（`Gaussian`、`ComputationalPhysics`、`leonardsusskind`、`the_theoretical_minimum`），以及 `*.chk` 等本地產物。
- 請將貢獻集中在被追蹤的資料夾，以維持輕量化 clone/update 工作流程。
- 若更新網站：編輯 `docs/`、本地預覽，再推送。

本地預覽 docs：

```bash
python -m http.server --directory docs
```

`docs/CNAME` 已設定為 `learn.lazying.art`。

## 疑難排解 🩺

- Gaussian 成功標準：在 `.log` 結尾附近看到 `Normal termination of Gaussian`。
- 若 GaussView 在 Wayland/遠端會話啟動失敗，請使用 `gview_safe.sh` 並明確傳入 `--gview`。
- 若 Gaussian scratch 發生錯誤，請檢查 `GAUSS_SCRDIR` 的可用空間與權限。
- 若 notebook 相依套件漂移，請以各子專案 README 為準，並在分享前把缺漏套件寫入環境檔。
- 目前倉庫狀態中 `comp_physics/environments.yaml` 看起來是佔位內容；在修正前請以明確安裝指令為準。

## 路線圖 🛣️

- 持續擴充 `comp_physics_python/` 章節覆蓋（transfer matrices、DMC/PIMC、FEM 等）。
- 統一腳本與 notebook 的輸出/作圖慣例。
- 為關鍵範例加入輕量且可重複的驗證檢查。
- 讓 `docs/` 與多語 README 持續與新實驗同步。

## 貢獻 🤝

歡迎提出 issue 與 pull request，特別是：

- 數值正確性檢查與可重現性改進。
- notebook/腳本更完善的環境規格。
- 更多教科書章節移植與 CLI 改進。
- `i18n/` 各語言文件的可讀性與清晰度改善。

提交大型內容更新前，請保留 `figures/` 中的產生圖檔，並確保除非另有文件說明，指令皆可由倉庫根目錄執行。

## 支援 LazyLearn ❤️

支持 LazyLearn 能讓實驗、文件與開放工具持續前進：

- 支援公開 demo 與 notebook 所需的主機、推論與儲存成本。
- 資助 EchoMind、LazyEdit 與此處量子/物理工具的專注開發週。
- 原型化 optics + wearables（IdeasGlass、LightMind），回饋到後續章節內容。
- 贊助學生、社群實驗室與創作者的免費部署。

### Donate

<div align="center">
<table style="margin:0 auto; text-align:center; border-collapse:collapse;">
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://chat.lazying.art/donate">https://chat.lazying.art/donate</a>
    </td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://chat.lazying.art/donate"><img src="figures/donate_button.svg" alt="Donate" height="44"></a>
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
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><img alt="WeChat QR" src="figures/donate_wechat.png" width="240"/></td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><img alt="Alipay QR" src="figures/donate_alipay.png" width="240"/></td>
  </tr>
</table>
</div>

**支援 / Donate**

- ご支援は研究・開発と運用の継続に役立ち、より多くのオープンなプロジェクトを皆さんに届ける力になります。
- 你的支持將用於研發與運維，幫助我持續公開分享更多專案與改進。
- Your support sustains my research, development, and ops so I can keep sharing more open projects and improvements.

## 授權 📄

此倉庫根目錄目前沒有 `LICENSE` 檔。在新增授權條款前，請將使用/再散布權利視為未明確定義；重用大量內容前請先向維護者確認。
