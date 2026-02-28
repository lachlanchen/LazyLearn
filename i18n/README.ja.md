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

LazyPhysics and Chemistry は **LazyLearn** の「コード + ノートブック」側にあたるリポジトリです。物理と化学を、意図的にゆっくり・実践的に学ぶためのログとして運用しています。進行中のノート、進捗、TODO は [learn.lazying.art](https://learn.lazying.art)（このリポジトリの `docs/` から配信）に公開し、再実行可能な成果物はこのリポジトリで管理することで、実験を常に再現可能な状態に保っています。

## 概要 🧭

### LazyLearn

- **拠点サイト:** [learn.lazying.art](https://learn.lazying.art) - 週次フォーカス、バックログ、ハイライトを掲載する公開サイト。
- **単一の参照元:** サイトからリンクされる実体は `examples/`、`comp_physics/`、`comp_physics_python/`、`multiwfn/`、`figures/` にあります。
- **更新フロー:** まずコード/ノートブックを反映し、必要なら図を再生成し、最後に `docs/` へ記録を追加してサイトの内容を最新化します。

このリポジトリは、単一のパッケージ化アプリではなく、意図的に混在形式で構成されています。実行可能スクリプト、ノートブック、参照資料、静的ドキュメントサイトを 1 つのバージョン管理ワークスペースにまとめています。

## 特徴 ✨

- 一般的なノート PC でも動く量子サンプルスクリプト（QAOA + VQE）。
- 計算物理ノートブックと補助ソルバ（例: Numerov ベースのワークフロー）。
- 教科書の計算物理プログラムを章ごとに Python 移植。
- 量子化学ポスト処理のローカル参照用として Multiwfn のソース/マニュアル一式を同梱。
- レポート/スライド用に生成した図をバージョン管理（`figures/`）。
- `i18n/` 配下に多言語 README を内蔵。
- `docs/` に静的マイクロサイトを配置（カスタムドメイン: `learn.lazying.art`）。

## プロジェクト構成 🗂️

### 配置内容

| パス | 用途 |
| --- | --- |
| `examples/` | Qiskit または PennyLane で実行する、焦点を絞った Python スクリプト（QAOA + VQE）。 |
| `comp_physics/` | 計算物理ノートブック、`numerov.py` のような補助スクリプト、関連データ/図。 |
| `comp_physics_python/` | Jos Thijssen 著 *Computational Physics* の Python 移植（章構成。詳細は [comp_physics_python/README.md](comp_physics_python/README.md)）。 |
| `multiwfn/` | Multiwfn 3.8 開発版ソースとマニュアル（ローカル参照用）。 |
| `figures/` | レポート/スライド/README で使用する静的 PNG/SVG 出力。 |
| `figs/` | ロゴとバナー素材。 |
| `docs/` | LazyLearn マイクロサイトのコンテンツ（GitHub Pages など任意の静的ホストで配信可能）。 |
| `i18n/` | 各言語版 README。 |

代表的なレイアウト:

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
> ルート直下のいくつかの項目は、このリポジトリ外を指すシンボリックリンクです。これらのパス配下を編集すると、外部ターゲット側が変更されます。

## 前提条件 🧰

| 要件 | 備考 |
| --- | --- |
| Python 3.x | ルートのスクリプト実行および多くのノートブック作業に必要。 |
| `pip` (or Conda) | パッケージ/環境管理。 |
| Jupyter Lab/Notebook (optional) | ノートブック運用で必要。 |
| Gaussian 16 + GaussView (optional) | Gaussian ワークフローで必要。 |

## インストール ⚙️

### 最小 Python セットアップ（ルートの examples 用）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

`comp_physics/` 内の Jupyter ノートブックも同じ環境を利用します。起動コマンド:

```bash
jupyter lab
# or
jupyter notebook
```

### 章別移植コード向けの任意依存関係（`comp_physics_python/`）

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## 使い方 🚀

### 実行例ワークフロー

- **Qiskit で QAOA**

```bash
python examples/qaoa_qiskit_maxcut.py
```

Aer 依存なし。純粋な statevector バックエンドを使います。

- **PennyLane で QAOA**

```bash
python examples/qaoa_pennylane_maxcut.py
```

`default.qubit` を使用します。

- **H2 の VQE**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

`figures/pennylane_h2_vqe_convergence.png` を再現します。

すべてのスクリプトは中間メトリクスを記録するため、図の再利用や新しい分子/グラフへの拡張が容易です。

## 計算物理ノートブック 📓

`comp_physics/` ディレクトリには作業ノートが反映されています。

- `comp_physics_textbook_code/` - ノートブックから切り出した再利用可能ルーチン。
- `chapter1.ipynb`、`chapter2.ipynb`、`numerov.ipynb`、`numpy_1ddft.ipynb` などの単体ノートブック。
- `bosonscattering/`、`lensless/`、`lightscattering/` など、実験ごとのデータと補助コードを持つトピック別フォルダ。

追加依存が必要な場合は `comp_physics/environments.yaml` に記録してください。

## 教科書コード移植 📚

`comp_physics_python/` は *Computational Physics* の古典的な Fortran プログラムを Python へ移植している継続プロジェクトです。章の対応例:

- `ch4/`: Hartree-Fock の例。
- `ch8/`: 分子動力学ソルバ。
- `ch10/`: モンテカルロサンプラ。

章の網羅状況と CLI コマンドは [comp_physics_python/README.md](comp_physics_python/README.md) を参照してください。

## Multiwfn リファレンス 🔬

`multiwfn/` には `Multiwfn_3.8_dev_src_Linux`、PDF マニュアル、クイックスタートガイドを配置しています。コンパイル済みバイナリはコミットしていません。

## 図版 🖼️

生成された PNG/SVG アセットは `figures/` に置き、生成元スクリプト/ノートブックと一緒にバージョン管理しています。

## 設定 🛠️

### Python とノートブック

- ルートのスクリプトは、上記の venv を前提にしています。
- ノートブック環境の詳細は各サブプロジェクト文書に分散しており、現時点ではリポジトリルートに単一の lockfile はありません。

### Gaussian ランナー（シンボリックリンク先のパス）

`Gaussian/run_gaussian.sh` は以下をサポートします。

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

挙動:

- 入力ファイルと同じ場所に `<basename>.log` を出力。
- `GAUSS_SCRDIR` が設定されていればそれを使用し、未設定時は `~/gaussian/scr` を既定値として使用。
- 入力中の `%chk=...` を検出し、チェックポイントが存在すれば GaussView は `.chk` を開き、なければ `.log` を開く。
- 利用可能な場合、`~/gaussian/gv/gview_safe.sh` を優先し、次に `gview.sh` を使用。

推奨 GaussView ラッパー:

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## 開発メモ 🧪

### バージョン管理メモ

- `.gitignore` により、`books/`、外部シンボリックリンク先（`Gaussian`、`ComputationalPhysics`、`leonardsusskind`、`the_theoretical_minimum`）、`*.chk` などのローカル生成物を含む重いパスを除外しています。
- クローン/更新を軽量に保つため、追跡対象フォルダへの貢献に集中してください。
- サイト更新時は `docs/` を編集し、ローカルでプレビュー後に push します。

ローカル docs プレビュー:

```bash
python -m http.server --directory docs
```

`docs/CNAME` は `learn.lazying.art` に設定されています。

## トラブルシューティング 🩺

- Gaussian 成功判定: `.log` の末尾付近に `Normal termination of Gaussian` があること。
- Wayland/リモート環境で GaussView が起動しない場合は `gview_safe.sh` を使い、`--gview` を明示指定してください。
- Gaussian の scratch エラー時は `GAUSS_SCRDIR` の空き容量と権限を確認してください。
- ノートブック依存がずれた場合は、サブプロジェクト README を参照元として扱い、共有前に不足パッケージを環境ファイルへ記録してください。
- 現在のリポジトリ状態では `comp_physics/environments.yaml` はプレースホルダーの可能性があるため、修正されるまでは明示的なインストールコマンドを優先してください。

## ロードマップ 🛣️

- `comp_physics_python/` の章カバレッジを拡張（伝達行列、DMC/PIMC、FEM など）。
- スクリプト/ノートブック間で出力・プロット規約を統一。
- 主要サンプル向けに軽量で再現可能な検証チェックを追加。
- 新しい実験に合わせて `docs/` と多言語 README の整合性を維持。

## コントリビューション 🤝

Issue / Pull Request を歓迎します。特に次の領域は歓迎です。

- 数値的妥当性の確認と再現性向上。
- ノートブック/スクリプト向け環境定義の改善。
- 教科書章の追加移植と CLI 改善。
- `i18n/` における多言語ドキュメントの明確化。

大きな内容更新を送る前に、生成図は `figures/` に残し、特記がない限りコマンドはリポジトリルートから実行可能であることを確認してください。

## LazyLearn を支援 ❤️

LazyLearn への支援は、実験・ドキュメント・オープンツールの継続的な公開を支えます。

- 公開デモやノートブックのホスティング/推論/ストレージ費用をカバー。
- EchoMind、LazyEdit、本リポジトリの量子/物理ユーティリティに集中するハックウィークを推進。
- 次の章につながる光学 + ウェアラブル（IdeasGlass、LightMind）を試作。
- 学生、コミュニティラボ、クリエイター向けの無料デプロイを後押し。

### 寄付

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
- 你的支持将用于研发与运维，帮助我持续公开分享更多项目与改进。
- Your support sustains my research, development, and ops so I can keep sharing more open projects and improvements.

## ライセンス 📄

このリポジトリのルートには現在 `LICENSE` ファイルがありません。ライセンスが追加されるまでは、利用・再配布の権利は未規定として扱い、実質的な再利用の前にメンテナへ確認してください。
