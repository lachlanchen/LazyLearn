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

LazyPhysics and Chemistry es la mitad de código + notebooks de **LazyLearn**: un registro de aprendizaje intencionalmente pausado y práctico sobre física y química. Las notas vivas, logros y TODOs se publican en [learn.lazying.art](https://learn.lazying.art) (servido desde `docs/` en este repositorio), mientras que los artefactos ejecutables se mantienen aquí para que los experimentos siempre tengan un hogar reproducible.

## Resumen 🧭

### LazyLearn

- **Base principal:** [learn.lazying.art](https://learn.lazying.art) - el sitio público con focos semanales, backlog y destacados.
- **Fuente de verdad:** todo lo que enlaza el sitio vive en `examples/`, `comp_physics/`, `comp_physics_python/`, `multiwfn/` o `figures/`.
- **Flujo de actualización:** publica primero código/notebooks, regenera gráficos si hace falta y luego añade una entrada en `docs/` para que el sitio refleje el trabajo más reciente.

Este repositorio es intencionalmente de formato mixto, no una única app empaquetada. Combina scripts ejecutables, notebooks, referencias y un sitio de documentación estático dentro de un único espacio versionado.

## Características ✨

- Scripts de ejemplo cuánticos (QAOA + VQE) que se ejecutan en portátiles comunes.
- Notebooks de física computacional y solucionadores auxiliares (por ejemplo, flujos basados en Numerov).
- Ports en Python, capítulo por capítulo, de programas de física computacional de libros de texto.
- Paquete de código fuente/manual de Multiwfn para referencia local de posprocesamiento en química cuántica.
- Figuras generadas versionadas para informes/diapositivas (`figures/`).
- Conjunto de README multilingüe integrado en `i18n/`.
- Micrositio estático en `docs/` (dominio personalizado: `learn.lazying.art`).

## Estructura del proyecto 🗂️

### Qué hay aquí

| Path | Purpose |
| --- | --- |
| `examples/` | Scripts Python enfocados (QAOA + VQE) que se ejecutan con Qiskit o PennyLane. |
| `comp_physics/` | Notebooks de física computacional, scripts auxiliares como `numerov.py`, y datos/figuras de soporte. |
| `comp_physics_python/` | Ports en Python de *Computational Physics* de Jos Thijssen, organizados por capítulo (ver [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/` | Paquete de código fuente de desarrollo de Multiwfn 3.8 más manuales para referencia local. |
| `figures/` | Salidas estáticas PNG/SVG usadas en informes/diapositivas y README. |
| `figs/` | Recursos de logotipo y banner. |
| `docs/` | Contenido del micrositio de LazyLearn (servido por GitHub Pages o cualquier host estático). |
| `i18n/` | Archivos README localizados. |

Diseño representativo:

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
> Varias entradas de nivel superior son symlinks a directorios fuera de este repositorio. Editar bajo esas rutas afecta objetivos externos.

## Requisitos previos 🧰

| Requirement | Notes |
| --- | --- |
| Python 3.x | Requerido para scripts en la raíz y la mayor parte del trabajo con notebooks. |
| `pip` (or Conda) | Gestión de paquetes/entornos. |
| Jupyter Lab/Notebook (optional) | Necesario para flujos basados en notebooks. |
| Gaussian 16 + GaussView (optional) | Necesario para flujos de Gaussian. |

## Instalación ⚙️

### Configuración mínima de Python (ejemplos raíz)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

Los notebooks de Jupyter dentro de `comp_physics/` usan el mismo entorno. Inícialos con:

```bash
jupyter lab
# or
jupyter notebook
```

### Dependencias opcionales de ports por capítulo (`comp_physics_python/`)

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## Uso 🚀

### Flujos de ejemplo

- **QAOA con Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

Sin dependencia de Aer; usa un backend puro de statevector.

- **QAOA con PennyLane**

```bash
python examples/qaoa_pennylane_maxcut.py
```

Usa `default.qubit`.

- **VQE para H2**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

Reproduce `figures/pennylane_h2_vqe_convergence.png`.

Todos los scripts registran métricas intermedias para que puedas reutilizar gráficos o extenderlos a nuevas moléculas/grafos.

## Notebooks de física computacional 📓

El directorio `comp_physics/` refleja notas de trabajo:

- `comp_physics_textbook_code/` - rutinas reutilizables extraídas de notebooks.
- Notebooks independientes como `chapter1.ipynb`, `chapter2.ipynb`, `numerov.ipynb` y `numpy_1ddft.ipynb`.
- Carpetas temáticas (`bosonscattering/`, `lensless/`, `lightscattering/`, etc.) con datos y utilidades por experimento.

Si hacen falta dependencias adicionales, regístralas en `comp_physics/environments.yaml`.

## Traducciones de código de libros 📚

`comp_physics_python/` es una traducción en crecimiento a Python de los programas clásicos en Fortran de *Computational Physics*. Ejemplo de mapeo por capítulos:

- `ch4/`: ejemplos de Hartree-Fock.
- `ch8/`: solucionadores de dinámica molecular.
- `ch10/`: muestreadores Monte Carlo.

Consulta [comp_physics_python/README.md](comp_physics_python/README.md) para la cobertura completa por capítulos y los comandos de CLI.

## Referencias de Multiwfn 🔬

`multiwfn/` conserva `Multiwfn_3.8_dev_src_Linux` junto con el manual PDF y la guía de inicio rápido. No se incluyen binarios compilados.

## Figuras 🖼️

Los recursos PNG/SVG generados viven en `figures/` para que las salidas queden versionadas junto a los scripts/notebooks que las producen.

## Configuración 🛠️

### Python y notebooks

- Los scripts de la raíz asumen el entorno venv mostrado arriba.
- Los detalles del entorno de notebooks están distribuidos entre documentos del proyecto; actualmente no existe un único lockfile en la raíz del repositorio.

### Gaussian runner (ruta con symlink)

`Gaussian/run_gaussian.sh` soporta:

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

Comportamiento:

- Escribe `<basename>.log` junto al input.
- Usa `GAUSS_SCRDIR` si está definido; de lo contrario, usa `~/gaussian/scr` por defecto.
- Detecta `%chk=...` en el input; si existe el checkpoint, GaussView abre `.chk`; en caso contrario, `.log`.
- Si está disponible, prefiere `~/gaussian/gv/gview_safe.sh` y luego `gview.sh`.

Wrapper recomendado para GaussView:

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## Notas de desarrollo 🧪

### Notas de control de versiones

- Las rutas pesadas se ignoran mediante `.gitignore`, incluyendo `books/`, objetivos externos con symlink (`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`) y artefactos locales como `*.chk`.
- Mantén las contribuciones enfocadas en carpetas rastreadas para conservar flujos ligeros de clonación/actualización.
- Para actualizaciones del sitio: edita `docs/`, previsualiza localmente y luego haz push.

Previsualización local de docs:

```bash
python -m http.server --directory docs
```

`docs/CNAME` está configurado para `learn.lazying.art`.

## Solución de problemas 🩺

- Criterio de éxito de Gaussian: `Normal termination of Gaussian` cerca del final del `.log`.
- Si GaussView falla en sesiones Wayland/remotas, usa `gview_safe.sh` y pasa `--gview` explícitamente.
- Si aparecen errores de scratch en Gaussian, verifica espacio libre y permisos en `GAUSS_SCRDIR`.
- Si hay deriva en dependencias de notebooks, toma los README de subproyectos como fuente de verdad y registra paquetes faltantes en archivos de entorno antes de compartir.
- `comp_physics/environments.yaml` parece ser un marcador de posición en el estado actual del repositorio; confía en comandos de instalación explícitos hasta que se corrija.

## Hoja de ruta 🛣️

- Seguir ampliando la cobertura por capítulos en `comp_physics_python/` (matrices de transferencia, DMC/PIMC, FEM y más).
- Armonizar convenciones de salidas/gráficos entre scripts y notebooks.
- Añadir validaciones ligeras y repetibles para ejemplos clave.
- Mantener `docs/` y los README multilingües alineados con los nuevos experimentos.

## Contribución 🤝

Issues y pull requests son bienvenidos, especialmente para:

- Verificaciones de corrección numérica y mejoras de reproducibilidad.
- Mejores especificaciones de entorno para notebooks/scripts.
- Más ports de capítulos de libros y mejoras de CLI.
- Claridad de documentación entre idiomas en `i18n/`.

Antes de enviar actualizaciones de contenido importantes, conserva las figuras generadas en `figures/` y asegúrate de que los comandos se puedan ejecutar desde la raíz del repositorio, salvo que se documente lo contrario.

## Support LazyLearn ❤️

Ayudar a LazyLearn mantiene en marcha los experimentos, la documentación y las herramientas abiertas:

- Cubrir hosting/inferencia/almacenamiento para demos públicas y notebooks.
- Financiar hack-weeks enfocados en EchoMind, LazyEdit y utilidades de cuántica/física aquí.
- Prototipar óptica + wearables (IdeasGlass, LightMind) que alimenten próximos capítulos.
- Patrocinar despliegues gratuitos para estudiantes, laboratorios comunitarios y creadores.

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

## Licencia 📄

Actualmente no hay un archivo `LICENSE` en la raíz de este repositorio. Hasta que se añada una licencia, considera que los derechos de uso/redistribución no están especificados y solicita aclaración al maintainer antes de reutilizar contenido sustancial.
