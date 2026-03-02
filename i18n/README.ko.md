[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# LazyPhysics와 화학

[![Site](https://img.shields.io/badge/website-learn.lazying.art-0a7ea4?style=for-the-badge&logo=githubpages&logoColor=white)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Learning-16a34a?style=for-the-badge&logo=target&logoColor=white)
![Repo Type](https://img.shields.io/badge/Repo-Type-Mixed%20Format-6b7280?style=for-the-badge)
![Docs](https://img.shields.io/badge/Docs-Static%20Microsite-0ea5e9?style=for-the-badge&logo=markdown&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-11-0f766e?style=for-the-badge&logo=googletranslate&logoColor=white)

## 📌 한눈에 보기

| 구분           | 이 저장소 역할                            |
| -------------- | ----------------------------------------- |
| 작업 흐름 유형 | 재현 가능한 물리 + 화학 학습 워크스페이스 |
| 제공 항목      | 스크립트, 노트북, 생성된 도표, 정적 문서  |
| 협업 모델      | 루트 실험 + 공개 사이트 배포              |
| 번역 범위      | `i18n/`의 README 미러 파일                |

LazyPhysics와 화학은 **LazyLearn**의 코드+노트북 파트입니다. 물리 및 화학을 위한 의도적으로 느긋한 실전 학습 로그입니다. 진행 중인 노트, 성과, TODO는 [learn.lazying.art](https://learn.lazying.art)(이 저장소의 `docs/`에서 제공)에서 공개되며, 실행 가능한 산출물은 이곳에 남겨두어 모든 실험의 재현 가능한 보관소로 둡니다.

## 개요 🧭

### LazyLearn

- **기본 위치:** [learn.lazying.art](https://learn.lazying.art) — 주간 주제, 백로그, 하이라이트가 정리된 공개 사이트.
- **진실 공급원:** 사이트가 참조하는 모든 항목은 `examples/`, `comp_physics/`, `comp_physics_python/`, `multiwfn/`, `figures/`에 있습니다.
- **업데이트 흐름:** 먼저 코드와 노트북을 반영하고, 필요 시 플롯을 재생성한 뒤 `docs/`에 항목을 추가해 사이트가 최신 작업을 반영하도록 합니다.

이 저장소는 단일 패키지 애플리케이션이 아닌 혼합 형식 저장소입니다. 실행 스크립트, 노트북, 레퍼런스 자료, 정적 문서 사이트가 하나의 버전 관리된 작업공간 아래 함께 있습니다.

## 기능 ✨

- 일반 노트북/PC에서도 동작하는 양자 예시 스크립트(QAOA + VQE).
- 전산 물리 노트북과 도우미 솔버(예: Numerov 기반 워크플로).
- 교재 전산물리 프로그램의 장별 Python 포팅.
- 로컬 양자화학 후처리 참조를 위한 Multiwfn 소스/매뉴얼 번들.
- 보고서/슬라이드를 위한 버전 관리된 생성 도표 (`figures/`).
- `i18n/`에 내장된 다국어 README 집합.
- 정적 마이크로사이트(`docs/`, 커스텀 도메인: `learn.lazying.art`).

## 프로젝트 구조 🗂️

### 이곳에 있는 것

| 경로                   | 목적                                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `examples/`            | Qiskit 또는 PennyLane으로 실행되는 QAOA + VQE 집중형 Python 스크립트.                                                                 |
| `comp_physics/`        | 전산 물리 노트북, `numerov.py` 같은 도우미 스크립트, 지원 데이터/도표.                                                                |
| `comp_physics_python/` | Jos Thijssen의 _Computational Physics_ 파이썬 포팅판, 장별 구성([comp_physics_python/README.md](comp_physics_python/README.md) 참조). |
| `multiwfn/`            | Multiwfn 3.8 개발자 소스 번들과 로컬 참조용 매뉴얼.                                                                                   |
| `figures/`             | 보고서/슬라이드 및 README에서 쓰는 정적 PNG/SVG 결과물.                                                                               |
| `figs/`                | 로고와 배너 자산.                                                                                                                     |
| `docs/`                | LazyLearn 마이크로사이트 콘텐츠(GitHub Pages 또는 기타 정적 호스팅에서 제공).                                                         |
| `i18n/`                | 지역화된 README 파일.                                                                                                                 |

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
> 최상위 항목 일부는 이 저장소 밖 디렉터리로 연결되는 심볼릭 링크입니다. 해당 경로에서 작업하면 외부 대상 파일이 변경됩니다.

## 사전 요구사항 🧰

| 요구사항                       | 참고                                       |
| ------------------------------ | ------------------------------------------ |
| Python 3.x                     | 루트 스크립트와 대부분 노트북 작업에 필요. |
| `pip` (또는 Conda)             | 패키지/환경 관리.                          |
| Jupyter Lab/Notebook (선택)    | 노트북 워크플로우를 위해 필요.             |
| Gaussian 16 + GaussView (선택) | Gaussian 워크플로우에 필요.                |

## 설치 ⚙️

### 최소 Python 설정(루트 예시)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

### ✅ 빠른 설정 체크리스트

| 단계 | 명령                                            | 목적                     |
| ---- | ----------------------------------------------- | ------------------------ |
| 1    | `python -m venv .venv`                          | 격리된 환경 생성         |
| 2    | `source .venv/bin/activate` (또는 OS 대응 명령) | 의존성 충돌 방지         |
| 3    | `pip install --upgrade pip`                     | 최신 패키지 툴 사용      |
| 4    | `pip install qiskit pennylane numpy matplotlib` | 핵심 실험 스택 설치      |
| 5    | `examples/`의 스크립트 하나 실행                | 설치를 엔드투엔드로 검증 |

`comp_physics/` 내 Jupyter 노트북은 같은 환경을 사용합니다. 다음으로 실행합니다:

```bash
jupyter lab
# or
jupyter notebook
```

### 선택적 장 포팅 의존성 (`comp_physics_python/`)

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## 사용법 🚀

### 예시 워크플로

- **Qiskit로 QAOA 실행**

```bash
python examples/qaoa_qiskit_maxcut.py
```

Aer 의존성이 없으며, 순수 상태벡터 백엔드를 사용합니다.

- **PennyLane으로 QAOA 실행**

```bash
python examples/qaoa_pennylane_maxcut.py
```

`default.qubit`를 사용합니다.

- **H2용 VQE**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

`figures/pennylane_h2_vqe_convergence.png`를 재현합니다.

모든 스크립트는 중간 지표를 로그에 남겨, 플롯을 재사용하거나 새 분자/그래프 실험으로 확장할 수 있습니다.

## 전산 물리 노트북 📓

`comp_physics/` 디렉터리는 작업 노트를 반영합니다:

- `comp_physics_textbook_code/` — 노트북에서 추출한 재사용 루틴.
- `chapter1.ipynb`, `chapter2.ipynb`, `numerov.ipynb`, `numpy_1ddft.ipynb` 같은 독립 노트북.
- 실험별 데이터와 도우미를 담은 토픽 폴더(`bosonscattering/`, `lensless/`, `lightscattering/` 등).

추가 의존성이 필요하면 `comp_physics/environments.yaml`에 기록하세요.

## 교재 코드 번역 📚

`comp_physics_python/`는 고전 Fortran 프로그램인 *Computational Physics*의 Python 번역을 진행 중입니다. 예시 장 매핑:

- `ch4/`: Hartree-Fock 예시.
- `ch8/`: 분자동역학 솔버.
- `ch10/`: 몬테카를로 샘플러.

전체 장 커버리지와 CLI 명령은 [comp_physics_python/README.md](comp_physics_python/README.md)를 참고하세요.

## Multiwfn 레퍼런스 🔬

`multiwfn/`는 `Multiwfn_3.8_dev_src_Linux`와 PDF 매뉴얼 및 빠른 시작 가이드를 보관합니다. 컴파일된 바이너리는 커밋되지 않습니다.

## 도표 🖼️

생성된 PNG/SVG 자산은 `figures/`에 보관되어, 출력물이 생성 스크립트/노트북과 함께 버전 관리됩니다.

## 설정 🛠️

### Python과 노트북

- 루트 스크립트는 위에서 설명한 가상환경을 전제로 합니다.
- 노트북 환경 정보는 각 프로젝트 문서에 분산되어 있으며, 현재 저장소 루트에는 단일 lockfile이 없습니다.

### Gaussian 실행기(심볼릭 경로)

`Gaussian/run_gaussian.sh`은 다음 형식을 지원합니다:

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

동작 방식:

- 입력 파일 옆에 `<basename>.log`를 생성합니다.
- `GAUSS_SCRDIR`가 설정되어 있으면 이를 사용하고, 없으면 `~/gaussian/scr`를 기본값으로 사용합니다.
- 입력에서 `%chk=...`를 감지합니다. 체크포인트가 있으면 GaussView가 `.chk`를, 아니면 `.log`를 엽니다.
- 가능하면 `~/gaussian/gv/gview_safe.sh`를 우선 사용하고, 그다음 `gview.sh`를 사용합니다.

권장 GaussView 래퍼:

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

### 🎬 네비게이션 지도

매일 작업할 때 출발점으로 사용하세요:

| 영역             | 시작 위치              |
| ---------------- | ---------------------- |
| 양자 데모        | `examples/`            |
| 물리 노트북      | `comp_physics/`        |
| 교재 번역        | `comp_physics_python/` |
| 양자 화학 도구   | `multiwfn/`            |
| 공개 산출물      | `docs/`                |
| 도표 및 일러스트 | `figures/`, `figs/`    |

### 버전 관리 노트

- `.gitignore`를 통해 대형 경로를 제외합니다(`books/`, 외부 심볼릭 대상(`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`) 및 `*.chk` 같은 로컬 산출물 포함).
- 추적 폴더 중심으로 기여해 가볍게 클론/업데이트할 수 있게 유지하세요.
- 웹사이트 업데이트의 경우: `docs/` 편집 → 로컬 미리보기 → 푸시.

로컬 문서 미리보기:

```bash
python -m http.server --directory docs
```

`docs/CNAME`은 `learn.lazying.art`로 설정되어 있습니다.

## 문제 해결 🩺

- Gaussian 성공 판정: `.log` 끝부분 근처에 `Normal termination of Gaussian` 문자열이 있어야 합니다.
- Wayland/원격 세션에서 GaussView가 실패하면 `gview_safe.sh`를 사용하고 `--gview`를 명시적으로 전달하세요.
- Gaussian scratch 오류가 발생하면 `GAUSS_SCRDIR`의 디스크 여유와 권한을 확인하세요.
- 노트북 의존성이 변경되면 하위 프로젝트 README를 사실상 진실의 원천으로 보고, 공유 전 환경 파일에 누락 패키지를 반영하세요.
- 현재 저장소 상태에서 `comp_physics/environments.yaml`은 플레이스홀더로 보입니다. 수정될 때까지는 명시적 설치 명령을 사용하세요.

## 로드맵 🛣️

- `comp_physics_python/` 장 커버리지를 계속 확장(전이 행렬, DMC/PIMC, FEM 등).
- 스크립트와 노트북 전반에서 출력/플롯 규약 통일.

## ❤️ Support

| Donate                                                                                                                                                                                                                                                                                                                                                     | PayPal                                                                                                                                                                                                                                                                                                                                                          | Stripe                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
