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

LazyPhysics and Chemistry는 **LazyLearn**의 코드 + 노트북 측면을 담당합니다. 물리와 화학을 의도적으로 천천히, 실용적으로 학습해 나가는 로그입니다. 계속 업데이트되는 노트, 성과, TODO는 [learn.lazying.art](https://learn.lazying.art)에 게시되며(이 저장소의 `docs/`에서 제공), 실행 가능한 아티팩트는 이곳에 유지해 실험이 항상 재현 가능한 기반을 갖도록 합니다.

## 개요 🧭

### LazyLearn

- **홈 베이스:** [learn.lazying.art](https://learn.lazying.art) - 주간 집중 주제, 백로그, 하이라이트를 보여주는 공개 사이트입니다.
- **단일 소스 오브 트루스:** 사이트가 링크하는 모든 자료는 `examples/`, `comp_physics/`, `comp_physics_python/`, `multiwfn/`, `figures/`에 있습니다.
- **업데이트 흐름:** 먼저 코드/노트북을 반영하고, 필요 시 플롯을 다시 생성한 뒤, `docs/`에 항목을 추가해 사이트가 최신 작업을 반영하도록 합니다.

이 저장소는 단일 패키지 앱이 아니라 의도적으로 혼합 형식으로 구성되어 있습니다. 실행 스크립트, 노트북, 참고 자료, 정적 문서 사이트를 하나의 버전 관리 워크스페이스에서 함께 다룹니다.

## 특징 ✨

- 일반 노트북/랩톱에서 실행 가능한 양자 예제 스크립트(QAOA + VQE).
- 계산물리 노트북과 보조 솔버(예: Numerov 기반 워크플로).
- 교재 계산물리 프로그램의 챕터별 Python 포팅.
- 로컬 양자화학 후처리 참고를 위한 Multiwfn 소스/매뉴얼 번들.
- 보고서/슬라이드용 생성 그림의 버전 관리(`figures/`).
- `i18n/`의 내장 다국어 README 세트.
- `docs/`의 정적 마이크로사이트(커스텀 도메인: `learn.lazying.art`).

## 프로젝트 구조 🗂️

### 포함된 구성

| Path | 용도 |
| --- | --- |
| `examples/` | Qiskit 또는 PennyLane으로 실행하는 집중형 Python 스크립트(QAOA + VQE). |
| `comp_physics/` | 계산물리 노트북, `numerov.py` 같은 보조 스크립트, 관련 데이터/그림. |
| `comp_physics_python/` | Jos Thijssen의 *Computational Physics*를 챕터별로 Python 포팅(참고: [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/` | 로컬 참고용 Multiwfn 3.8 개발자 소스 번들과 매뉴얼. |
| `figures/` | 보고서/슬라이드 및 README에 쓰이는 정적 PNG/SVG 출력물. |
| `figs/` | 로고 및 배너 에셋. |
| `docs/` | LazyLearn 마이크로사이트 콘텐츠(GitHub Pages 또는 임의 정적 호스트 제공). |
| `i18n/` | 지역화된 README 파일. |

대표 레이아웃:

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
> 여러 최상위 항목은 이 저장소 밖 디렉터리를 가리키는 심볼릭 링크입니다. 해당 경로 아래를 수정하면 외부 대상에도 변경이 적용됩니다.

## 사전 요구사항 🧰

| Requirement | 비고 |
| --- | --- |
| Python 3.x | 루트 스크립트와 대부분의 노트북 작업에 필요합니다. |
| `pip` (또는 Conda) | 패키지/환경 관리용입니다. |
| Jupyter Lab/Notebook (선택) | 노트북 워크플로에 필요합니다. |
| Gaussian 16 + GaussView (선택) | Gaussian 워크플로에 필요합니다. |

## 설치 ⚙️

### 최소 Python 설정 (루트 예제)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

`comp_physics/` 내부 Jupyter 노트북도 같은 환경을 사용합니다. 다음으로 실행하세요:

```bash
jupyter lab
# or
jupyter notebook
```

### 선택 챕터 포팅 의존성 (`comp_physics_python/`)

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## 사용법 🚀

### 예제 워크플로

- **Qiskit으로 QAOA**

```bash
python examples/qaoa_qiskit_maxcut.py
```

Aer 의존성 없이 순수 statevector 백엔드를 사용합니다.

- **PennyLane으로 QAOA**

```bash
python examples/qaoa_pennylane_maxcut.py
```

`default.qubit`을 사용합니다.

- **H2용 VQE**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

`figures/pennylane_h2_vqe_convergence.png`를 재현합니다.

모든 스크립트는 중간 메트릭을 로깅하므로 플롯 재사용이나 새로운 분자/그래프로의 확장이 쉽습니다.

## 계산물리 노트북 📓

`comp_physics/` 디렉터리는 작업 노트를 반영합니다:

- `comp_physics_textbook_code/` - 노트북에서 추출한 재사용 루틴.
- `chapter1.ipynb`, `chapter2.ipynb`, `numerov.ipynb`, `numpy_1ddft.ipynb` 같은 독립 노트북.
- 실험별 데이터/헬퍼를 포함한 주제 폴더(`bosonscattering/`, `lensless/`, `lightscattering/` 등).

추가 의존성이 필요하면 `comp_physics/environments.yaml`에 기록하세요.

## 교재 코드 번역 📚

`comp_physics_python/`은 *Computational Physics*의 고전 Fortran 프로그램을 Python으로 옮겨가는 확장형 프로젝트입니다. 챕터 매핑 예시는 다음과 같습니다:

- `ch4/`: Hartree-Fock 예제.
- `ch8/`: 분자동역학 솔버.
- `ch10/`: Monte Carlo 샘플러.

전체 챕터 범위와 CLI 명령은 [comp_physics_python/README.md](comp_physics_python/README.md)를 참고하세요.

## Multiwfn 참고 자료 🔬

`multiwfn/`에는 `Multiwfn_3.8_dev_src_Linux`, PDF 매뉴얼, 퀵스타트 가이드가 포함됩니다. 컴파일된 바이너리는 커밋하지 않습니다.

## Figures 🖼️

생성된 PNG/SVG 에셋은 `figures/`에 보관되어, 출력물이 이를 생성한 스크립트/노트북과 함께 버전 관리됩니다.

## 설정 🛠️

### Python 및 노트북

- 루트 스크립트는 위에 제시한 venv를 가정합니다.
- 노트북 환경 세부 사항은 프로젝트 문서에 분산되어 있으며, 현재 저장소 루트에는 단일 lockfile이 없습니다.

### Gaussian runner (symlinked path)

`Gaussian/run_gaussian.sh`는 다음을 지원합니다:

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

동작:

- 입력 파일 옆에 `<basename>.log`를 작성합니다.
- `GAUSS_SCRDIR`이 설정되어 있으면 사용하고, 아니면 기본값 `~/gaussian/scr`를 사용합니다.
- 입력의 `%chk=...`를 감지하며, 체크포인트가 있으면 GaussView가 `.chk`를, 없으면 `.log`를 엽니다.
- 가능하면 `~/gaussian/gv/gview_safe.sh`를 우선 사용하고, 그다음 `gview.sh`를 사용합니다.

권장 GaussView wrapper:

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## 개발 노트 🧪

### 버전 관리 참고

- `books/`, 외부 심볼릭 링크 대상(`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`), `*.chk` 같은 로컬 아티팩트를 포함한 무거운 경로는 `.gitignore`로 제외됩니다.
- 가벼운 clone/update 워크플로를 위해 추적되는 폴더 중심으로 기여를 유지하세요.
- 웹사이트 업데이트 시: `docs/`를 수정하고 로컬에서 미리 본 뒤 push하세요.

로컬 문서 미리보기:

```bash
python -m http.server --directory docs
```

`docs/CNAME`은 `learn.lazying.art`로 설정되어 있습니다.

## 문제 해결 🩺

- Gaussian 성공 기준: `.log` 끝부분 근처에 `Normal termination of Gaussian`이 표시됩니다.
- Wayland/원격 세션에서 GaussView가 실패하면 `gview_safe.sh`를 사용하고 `--gview`를 명시적으로 전달하세요.
- Gaussian scratch 오류가 발생하면 `GAUSS_SCRDIR`의 디스크 여유 공간과 권한을 확인하세요.
- 노트북 의존성이 어긋나면 하위 프로젝트 README를 기준 문서로 간주하고, 공유 전에 환경 파일에 누락 패키지를 기록하세요.
- 현재 저장소 상태에서 `comp_physics/environments.yaml`은 플레이스홀더로 보입니다. 수정되기 전까지는 명시적 설치 명령을 우선 사용하세요.

## 로드맵 🛣️

- `comp_physics_python/` 챕터 범위를 계속 확장(전달 행렬, DMC/PIMC, FEM 등).
- 스크립트/노트북 전반의 출력/플롯 규약을 정렬.
- 핵심 예제에 대해 가볍고 반복 가능한 검증 체크 추가.
- 새 실험에 맞춰 `docs/`와 다국어 README를 계속 동기화.

## 기여 🤝

다음 항목에 대한 이슈 및 PR을 환영합니다:

- 수치적 정확성 검증 및 재현성 개선.
- 노트북/스크립트용 더 나은 환경 명세.
- 추가 교재 챕터 포팅 및 CLI 개선.
- `i18n/` 전반의 다국어 문서 명확성 향상.

대규모 콘텐츠 업데이트를 제출하기 전에는 생성된 그림을 `figures/`에 보관하고, 별도 문서화가 없다면 명령이 저장소 루트에서 실행 가능하도록 확인하세요.

## LazyLearn 후원 ❤️

LazyLearn을 후원하면 실험, 문서화, 오픈 툴링을 지속하는 데 큰 도움이 됩니다:

- 공개 데모와 노트북을 위한 호스팅/추론/스토리지 비용 지원.
- EchoMind, LazyEdit, 그리고 이 저장소의 양자/물리 유틸리티 집중 해킹 주간 지원.
- 향후 챕터로 이어질 광학 + 웨어러블(IdeasGlass, LightMind) 프로토타이핑.
- 학생, 커뮤니티 랩, 창작자를 위한 무료 배포 후원.

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
- 你的支持将用于研发与运维，帮助我持续公开分享更多项目与改进。
- Your support sustains my research, development, and ops so I can keep sharing more open projects and improvements.

## 라이선스 📄

현재 이 저장소 루트에는 `LICENSE` 파일이 없습니다. 라이선스가 추가되기 전까지 사용/재배포 권한은 명시되지 않은 것으로 간주하고, 중요한 콘텐츠를 재사용하기 전에 유지보수자에게 확인하세요.
