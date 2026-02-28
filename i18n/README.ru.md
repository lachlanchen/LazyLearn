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

LazyPhysics and Chemistry — это кодовая и notebook-часть **LazyLearn**: намеренно медленный и практичный журнал изучения физики и химии. Живые заметки, достижения и TODO публикуются на [learn.lazying.art](https://learn.lazying.art) (сайт обслуживается из `docs/` в этом репозитории), а исполняемые артефакты хранятся здесь, чтобы у экспериментов всегда был воспроизводимый дом.

## Обзор 🧭

### LazyLearn

- **Главная площадка:** [learn.lazying.art](https://learn.lazying.art) - публичный сайт с недельными фокусами, бэклогом и ключевыми результатами.
- **Единый источник правды:** все, на что ссылается сайт, находится в `examples/`, `comp_physics/`, `comp_physics_python/`, `multiwfn/` или `figures/`.
- **Поток обновлений:** сначала публикуйте код/notebook, при необходимости пересобирайте графики, затем добавляйте запись в `docs/`, чтобы сайт отражал последнюю работу.

Этот репозиторий намеренно смешанного формата, а не одно упакованное приложение. Он объединяет исполняемые скрипты, notebook, справочные материалы и статический сайт документации в одном версионируемом рабочем пространстве.

## Возможности ✨

- Квантовые скрипты-примеры (QAOA + VQE), которые запускаются на обычных ноутбуках.
- Notebook по вычислительной физике и вспомогательные решатели (например, рабочие процессы на основе Numerov).
- Построчные Python-переносы программ по вычислительной физике из учебника, глава за главой.
- Пакет исходников/руководств Multiwfn для локальной справочной постобработки в квантовой химии.
- Версионируемые сгенерированные фигуры для отчетов/слайдов (`figures/`).
- Встроенный набор многоязычных README в `i18n/`.
- Статический микросайт в `docs/` (кастомный домен: `learn.lazying.art`).

## Структура проекта 🗂️

### Что находится здесь

| Путь | Назначение |
| --- | --- |
| `examples/` | Целевые Python-скрипты (QAOA + VQE), запускаемые с Qiskit или PennyLane. |
| `comp_physics/` | Notebook по вычислительной физике, вспомогательные скрипты вроде `numerov.py` и сопутствующие данные/фигуры. |
| `comp_physics_python/` | Python-переносы *Computational Physics* Jos Thijssen, организованные по главам (см. [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/` | Пакет исходников разработчика Multiwfn 3.8 плюс руководства для локальной справки. |
| `figures/` | Статические PNG/SVG-выходы для отчетов/слайдов и README. |
| `figs/` | Ассеты логотипа и баннера. |
| `docs/` | Контент микросайта LazyLearn (обслуживается GitHub Pages или любым статическим хостингом). |
| `i18n/` | Локализованные файлы README. |

Представительная структура:

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
> Несколько записей верхнего уровня — это symlink на каталоги вне этого репозитория. Редактирование по этим путям изменяет внешние целевые директории.

## Предварительные требования 🧰

| Требование | Примечания |
| --- | --- |
| Python 3.x | Нужен для корневых скриптов и большей части работы с notebook. |
| `pip` (or Conda) | Управление пакетами/окружениями. |
| Jupyter Lab/Notebook (optional) | Нужно для workflow с notebook. |
| Gaussian 16 + GaussView (optional) | Нужно для workflow с Gaussian. |

## Установка ⚙️

### Минимальная настройка Python (корневые примеры)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

Jupyter notebook внутри `comp_physics/` используют то же окружение. Запуск:

```bash
jupyter lab
# or
jupyter notebook
```

### Необязательные зависимости переносов по главам (`comp_physics_python/`)

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## Использование 🚀

### Примеры workflow

- **QAOA с Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

Без зависимости от Aer; используется чистый statevector backend.

- **QAOA с PennyLane**

```bash
python examples/qaoa_pennylane_maxcut.py
```

Используется `default.qubit`.

- **VQE для H2**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

Воспроизводит `figures/pennylane_h2_vqe_convergence.png`.

Все скрипты логируют промежуточные метрики, чтобы вы могли переиспользовать графики или расширять примеры на новые молекулы/графы.

## Notebook по вычислительной физике 📓

Каталог `comp_physics/` отражает рабочие заметки:

- `comp_physics_textbook_code/` - переиспользуемые процедуры, извлеченные из notebook.
- Отдельные notebook, такие как `chapter1.ipynb`, `chapter2.ipynb`, `numerov.ipynb` и `numpy_1ddft.ipynb`.
- Тематические папки (`bosonscattering/`, `lensless/`, `lightscattering/` и т. д.) с данными и вспомогательными файлами для каждого эксперимента.

Если нужны дополнительные зависимости, зафиксируйте их в `comp_physics/environments.yaml`.

## Переводы учебникового кода 📚

`comp_physics_python/` — это растущий Python-перенос классических программ на Fortran из *Computational Physics*. Пример соответствия глав:

- `ch4/`: примеры Hartree-Fock.
- `ch8/`: решатели молекулярной динамики.
- `ch10/`: сэмплеры Monte Carlo.

Полное покрытие глав и CLI-команды см. в [comp_physics_python/README.md](comp_physics_python/README.md).

## Справочные материалы Multiwfn 🔬

`multiwfn/` хранит `Multiwfn_3.8_dev_src_Linux` вместе с PDF-руководством и quick-start guide. Скомпилированные бинарные файлы не коммитятся.

## Фигуры 🖼️

Сгенерированные PNG/SVG-ассеты находятся в `figures/`, поэтому результаты версионируются вместе со скриптами/notebook, которые их создают.

## Конфигурация 🛠️

### Python и notebook

- Корневые скрипты предполагают использование venv, показанного выше.
- Детали окружения для notebook распределены по документации проекта; единого lockfile в корне репозитория сейчас нет.

### Gaussian runner (путь через symlink)

`Gaussian/run_gaussian.sh` поддерживает:

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

Поведение:

- Записывает `<basename>.log` рядом с входным файлом.
- Использует `GAUSS_SCRDIR`, если он задан, иначе по умолчанию `~/gaussian/scr`.
- Определяет `%chk=...` во входном файле; если checkpoint существует, GaussView открывает `.chk`, иначе `.log`.
- Если доступно, сначала выбирает `~/gaussian/gv/gview_safe.sh`, затем `gview.sh`.

Рекомендуемый обертчик GaussView:

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## Заметки по разработке 🧪

### Заметки по контролю версий

- Тяжелые пути игнорируются через `.gitignore`, включая `books/`, внешние symlink-цели (`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`) и локальные артефакты вроде `*.chk`.
- Держите вклад сфокусированным на отслеживаемых папках, чтобы сохранить легкие workflows клонирования/обновления.
- Для обновлений сайта: редактируйте `docs/`, просматривайте локально, затем пушьте.

Локальный предпросмотр docs:

```bash
python -m http.server --directory docs
```

`docs/CNAME` настроен на `learn.lazying.art`.

## Устранение неполадок 🩺

- Критерий успешного Gaussian: `Normal termination of Gaussian` рядом с концом `.log`.
- Если GaussView не запускается в Wayland/удаленных сессиях, используйте `gview_safe.sh` и передайте `--gview` явно.
- Если возникают ошибки scratch в Gaussian, проверьте свободное место и права в `GAUSS_SCRDIR`.
- Если зависимости notebook расходятся, считайте README подпроектов источником правды и фиксируйте недостающие пакеты в файлах окружения перед передачей другим.
- `comp_physics/environments.yaml` в текущем состоянии репозитория выглядит как заглушка; полагайтесь на явные команды установки, пока файл не исправлен.

## Дорожная карта 🛣️

- Продолжать расширять покрытие глав `comp_physics_python/` (transfer matrices, DMC/PIMC, FEM и далее).
- Гармонизировать соглашения по output/графикам между скриптами и notebook.
- Добавить легковесные повторяемые проверки для ключевых примеров.
- Поддерживать синхронизацию `docs/` и многоязычных README с новыми экспериментами.

## Вклад 🤝

Issues и pull requests приветствуются, особенно для:

- Проверок численной корректности и улучшений воспроизводимости.
- Более точных спецификаций окружения для notebook/скриптов.
- Дополнительных переносов глав учебника и улучшений CLI.
- Повышения ясности документации на разных языках в `i18n/`.

Перед отправкой крупных обновлений контента держите сгенерированные фигуры в `figures/` и убедитесь, что команды запускаются из корня репозитория, если не документировано иное.

## Поддержите LazyLearn ❤️

Помощь LazyLearn поддерживает эксперименты, документацию и развитие открытых инструментов:

- Покрывает хостинг/инференс/хранилище для публичных демо и notebook.
- Финансирует сфокусированные hack-week по EchoMind, LazyEdit и квантовым/физическим утилитам здесь.
- Позволяет прототипировать оптику + wearables (IdeasGlass, LightMind), которые питают будущие главы.
- Спонсирует бесплатные развёртывания для студентов, общественных лабораторий и создателей.

### Пожертвовать

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
- Ваша поддержка помогает мне поддерживать исследования, разработку и операционную работу, чтобы я мог продолжать делиться открытыми проектами и улучшениями.

## Лицензия 📄

В корне этого репозитория сейчас отсутствует файл `LICENSE`. Пока лицензия не добавлена, считайте права на использование/распространение неуточненными и запрашивайте разъяснение у мейнтейнера перед повторным использованием существенного контента.
