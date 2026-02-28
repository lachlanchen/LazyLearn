[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# LazyPhysics と Chemistry

[![Site](https://img.shields.io/badge/website-learn.lazying.art-0a7ea4?style=for-the-badge&logo=githubpages&logoColor=white)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Learning-16a34a?style=for-the-badge&logo=target&logoColor=white)
![Repo Type](https://img.shields.io/badge/Repo-Type-Mixed%20Format-6b7280?style=for-the-badge)
![Docs](https://img.shields.io/badge/Docs-Static%20Microsite-0ea5e9?style=for-the-badge&logo=markdown&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-11-0f766e?style=for-the-badge&logo=googletranslate&logoColor=white)

## 📌 概要

| フォーカス | このリポジトリの内容 |
| --- | --- |
| ワークフローの種類 | 再現可能な物理 + 化学の学習ワークスペース |
| 提供物 | スクリプト、ノートブック、生成済み図、静的ドキュメント |
| 協働モデル | ルート実験 + 公開サイトへの公開 |
| 翻訳カバレッジ | `i18n/` の README ミラー |

LazyPhysics and Chemistry は **LazyLearn** の「コード + ノートブック」側です。これは物理・化学を「ゆっくり、実務的に」学ぶための意図的な学習ログです。
`docs/` で公開されている [learn.lazying.art](https://learn.lazying.art) には日々のノートや結果、TODO を掲載し、再現性のある実験素材はこのリポジトリに残して実験の拠点にしています。

## 概要 🧭

### LazyLearn

- **拠点:** [learn.lazying.art](https://learn.lazying.art) - 毎週の注力テーマ、バックログ、ハイライトを公開する公開サイト。
- **正本情報源:** サイトから参照されるほぼすべてが `examples/`、`comp_physics/`、`comp_physics_python/`、`multiwfn/`、`figures/` にあります。
- **更新フロー:** まずコード/ノートブックを投入し、必要ならプロットを再生成したうえで `docs/` に記録を追加して、サイトが最新成果を反映するようにします。

このリポジトリは意図的に複合形式で構成されており、単一のアプリとしてまとめるのではなく、実行可能スクリプト、ノートブック、参考資料、静的ドキュメントを1つのバージョン管理空間にまとめています。

## 特徴 ✨

- ノートPCでも動作する QAOA + VQE の量子例題スクリプト。
- 計算物理のノートブックと補助ソルバー（例: Numerov ベースのワークフロー）。
- 章ごとの Python 移植版として、計算物理の教科書サンプルを実装。
- Multiwfn のソースとマニュアルを束ねた、量子化学後処理向けのローカル参照資料。
- レポートやスライド向けの版管理済み図版 (`figures/`)。
- `i18n/` 下の組み込み多言語 README。
- 静的なミニサイト `docs/`（独自ドメイン: `learn.lazying.art`）。

## 構成 🗂️

### この場所にあるもの

| パス | 用途 |
| --- | --- |
| `examples/` | QAOA + VQE の Python スクリプト（Qiskit または PennyLane で実行）。 |
| `comp_physics/` | 計算物理のノートブック、`numerov.py` などの補助スクリプト、関連データ/図。 |
| `comp_physics_python/` | Jos Thijssen の *Computational Physics* を章ごとに Python 化した実装（詳細は [comp_physics_python/README.md](comp_physics_python/README.md) を参照）。 |
| `multiwfn/` | Multiwfn 3.8 開発者用ソース一式とローカル参照向けマニュアル。 |
| `figures/` | レポートやスライド、README で使用する静的 PNG/SVG 出力。 |
| `figs/` | ロゴやバナーのアセット。 |
| `docs/` | GitHub Pages などで提供する LazyLearn ミニサイト。 |
| `i18n/` | ローカライズされた README ファイル。 |

代表的な構成:

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
> いくつかのトップレベル項目はこのリポジトリ外のディレクトリへのシンボリックリンクです。これら配下を編集すると外部の対象にも影響します。

## 前提条件 🧰

| 要件 | 補足 |
| --- | --- |
| Python 3.x | ルートのスクリプトとほとんどのノートブック実行に必須。 |
| `pip`（または Conda） | パッケージ/環境管理。 |
| Jupyter Lab/Notebook（任意） | ノートブックワークフローの利用時に必要。 |
| Gaussian 16 + GaussView（任意） | Gaussian ワークフロー時に必要。 |

## 導入 ⚙️

### 最小構成の Python セットアップ（ルート例）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

### ✅ クイックチェックリスト

| ステップ | コマンド | 目的 |
| --- | --- | --- |
| 1 | `python -m venv .venv` | 独立した実行環境を作成 |
| 2 | `source .venv/bin/activate`（OS に応じた方法） | 依存関係の衝突を防ぐ |
| 3 | `pip install --upgrade pip` | 最新のパッケージ管理ツールを確保 |
| 4 | `pip install qiskit pennylane numpy matplotlib` | コアとなる実験スタックを導入 |
| 5 | `examples/` 内で1本スクリプトを実行 | インストール全体を検証 |

`comp_physics/` 内の Jupyter ノートブックは同じ環境を使用します。起動方法:

```bash
jupyter lab
# or
jupyter notebook
```

### `comp_physics_python/` 用の任意依存（章ごとの要件）

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## 使用方法 🚀

### 実行例のワークフロー

- **Qiskit 版 QAOA**

```bash
python examples/qaoa_qiskit_maxcut.py
```

Aer 依存なし。純粋な statevector バックエンドを使用。

- **PennyLane 版 QAOA**

```bash
python examples/qaoa_pennylane_maxcut.py
```

`default.qubit` を使用。

- **H2 の VQE**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

`figures/pennylane_h2_vqe_convergence.png` を再現します。

各スクリプトは中間メトリクスをログ出力するため、生成図の再利用や新規分子/新規グラフへの拡張が容易です。

## 計算物理ノートブック 📓

`comp_physics/` は作業ノートを反映した構成です。

- `comp_physics_textbook_code/` - ノートブックから抽出した再利用可能なルーチン。
- `chapter1.ipynb`、`chapter2.ipynb`、`numerov.ipynb`、`numpy_1ddft.ipynb` などの単独ノートブック。
- `bosonscattering/`、`lensless/`、`lightscattering/` などのトピックフォルダで、実験ごとのデータと補助ファイルを管理。

追加の依存が必要な場合は、`comp_physics/environments.yaml` に記録してください。

## 教科書コードの移植 📚

`comp_physics_python/` は *Computational Physics* の古典的 Fortran コードを Python に移植している進行中のリポジトリです。章対応例:

- `ch4/`: ハートリー・フォック関連の例。
- `ch8/`: 分子動力学ソルバー。
- `ch10/`: モンテカルロ法サンプラー。

章構成の全体像と CLI コマンドは [comp_physics_python/README.md](comp_physics_python/README.md) を参照してください。

## Multiwfn 参照 🔬

`multiwfn/` には `Multiwfn_3.8_dev_src_Linux`、PDF マニュアル、クイックスタートガイドを保持します。コンパイル済みバイナリはコミットしません。

## 図表 🖼️

生成した PNG/SVG は `figures/` に配置され、生成元のスクリプト/ノートブックと同時にバージョン管理されます。

## 設定 🛠️

### Python とノートブック

- ルートスクリプトは上記の venv を前提とします。
- ノートブック環境の詳細は各プロジェクトドキュメントに分散しており、現状リポジトリルートには単一の lockfile はありません。

### Gaussian ランナー（シンボリックリンク配下）

`Gaussian/run_gaussian.sh` は次をサポートします。

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

動作:

- 入力ファイルと同じ場所に `<basename>.log` を生成。
- `GAUSS_SCRDIR` が設定されていればそれを使用し、未設定なら `~/gaussian/scr` を既定値として使用。
- 入力内の `%chk=...` を検出し、該当するチェックポイントがあれば `.chk` を、なければ `.log` を GaussView で開く。
- 利用可能なら `~/gaussian/gv/gview_safe.sh`、次に `gview.sh` を優先して使用。

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

## 開発ノート 🧪

### 🎬 ナビゲーションマップ

日々の作業開始ポイント:

| エリア | 開始位置 |
| --- | --- |
| 量子デモ | `examples/` |
| 物理ノートブック | `comp_physics/` |
| 教科書翻訳 | `comp_physics_python/` |
| 量子化学ツール | `multiwfn/` |
| 公開成果物 | `docs/` |
| 図表・イラスト | `figures/`、`figs/` |

### バージョン管理ノート

- 重いパスは `.gitignore` で除外されています。対象には `books/`、外部シンボリックリンク先（`Gaussian`、`ComputationalPhysics`、`leonardsusskind`、`the_theoretical_minimum`）、`*.chk` のようなローカル成果物が含まれます。
- 軽量なクローン/更新フローを保つため、追跡対象フォルダへ寄せた変更に留めてください。
- サイト更新は `docs/` を編集し、ローカルで確認してから push してください。

ローカルでドキュメントをプレビューするには:

```bash
python -m http.server --directory docs
```

`docs/CNAME` は `learn.lazying.art` 向けに設定されています。

## トラブルシューティング 🩺

- Gaussian の完了判定: `.log` の末尾付近に `Normal termination of Gaussian` があるか。
- Wayland/リモート環境で GaussView 起動に失敗する場合は、`gview_safe.sh` を使い `--gview` を明示。
- Gaussian スクラッチ関連のエラー時は、`GAUSS_SCRDIR` の空き容量と権限を確認。
- ノートブック依存が変動する場合、サブプロジェクトの README を真実の情報源として参照し、不足パッケージは環境定義ファイルに追記してから共有してください。
- `comp_physics/environments.yaml` は現状プレースホルダーの可能性があるため、修正されるまで明示的なインストール手順に従ってください。

## ロードマップ 🛣️

- `comp_physics_python/` の章カバーを拡充（転送行列、DMC/PIMC、FEM など）。
- スクリプトとノートブック全体で出力・図表の書式を統一。
- 主要例について軽量かつ再現可能な検証チェックを追加。
- `docs/` と `i18n/` の README を新規実験と常に整合させる。

## コントリビューション 🤝

Issue と Pull Request は歓迎します。特に以下を重視します。

- 数値検証と再現性向上。
- ノートブック/スクリプト向け環境定義の改善。
- 追加の教科書章移植と CLI の改善。
- `i18n/` 全体でのドキュメント明確化。

主要なコンテンツ更新を提出する前に、`figures/` の生成物を残し、コマンドが特に記載されていない限りリポジトリルートから実行可能であることを確認してください。

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License 📄

このリポジトリには現在、ルート直下の `LICENSE` がありません。ライセンスが追加されるまで、本内容の使用・再配布権は未指定とみなすべきであり、再利用する場合はメンテナに確認してください。
