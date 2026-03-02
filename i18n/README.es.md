[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# LazyPhysics y Química

[![Site](https://img.shields.io/badge/website-learn.lazying.art-0a7ea4?style=for-the-badge&logo=githubpages&logoColor=white)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Learning-16a34a?style=for-the-badge&logo=target&logoColor=white)
![Repo Type](https://img.shields.io/badge/Repo-Type-Mixed%20Format-6b7280?style=for-the-badge)
![Docs](https://img.shields.io/badge/Docs-Static%20Microsite-0ea5e9?style=for-the-badge&logo=markdown&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-11-0f766e?style=for-the-badge&logo=googletranslate&logoColor=white)

## 📌 Resumen rápido

| Enfoque                 | Qué hace este repositorio                                      |
| ----------------------- | -------------------------------------------------------------- |
| Tipo de flujo           | Espacio de aprendizaje reproducible de física y química        |
| Entregables             | Scripts, notebooks, figuras generadas y documentación estática |
| Modelo de colaboración  | Experimentos base + publicación del sitio público              |
| Cobertura de traducción | Archivos espejo de README en `i18n/`                           |

LazyPhysics y Química es la parte de **código + notebooks** de **LazyLearn**: un registro de aprendizaje intencionalmente pausado y práctico de física y química. Las notas activas, avances y pendientes se publican en [learn.lazying.art](https://learn.lazying.art) (servido desde `docs/` en este repositorio), mientras que los artefactos ejecutables permanecen aquí para que los experimentos siempre tengan un hogar reproducible.

## Overview 🧭

### LazyLearn

- **Base principal:** [learn.lazying.art](https://learn.lazying.art), el sitio público con focos semanales, backlog y destacados.
- **Fuente de verdad:** todo lo enlazado por el sitio vive en `examples/`, `comp_physics/`, `comp_physics_python/`, `multiwfn/` o `figures/`.
- **Flujo de actualización:** primero se publica el código/los notebooks, luego se regeneran gráficos si hace falta y, por último, se añade una entrada en `docs/` para que el sitio refleje el trabajo más reciente.

Este repositorio está pensado con formato mixto, no es una sola aplicación empaquetada. Combina scripts ejecutables, notebooks, referencias y un sitio estático en un único espacio versionado.

## Características ✨

- Scripts de ejemplo cuánticos (QAOA + VQE) que funcionan en portátiles convencionales.
- Notebooks de física computacional y utilidades de ayuda (por ejemplo, flujos basados en Numerov).
- Traducciones de programas de física computacional de texto por capítulos a Python.
- Paquete de fuente/manual de Multiwfn para referencia local de posprocesamiento de química cuántica.
- Figuras generadas versionadas para informes/diapositivas (`figures/`).
- Conjunto de README multilingüe versionado bajo `i18n/`.
- Micrositio estático en `docs/` (dominio personalizado: `learn.lazying.art`).

## Estructura del proyecto 🗂️

### Qué hay aquí

| Ruta                   | Función                                                                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `examples/`            | Scripts de Python focalizados (QAOA + VQE) que funcionan con Qiskit o PennyLane.                                                                             |
| `comp_physics/`        | Notebooks de física computacional, scripts auxiliares como `numerov.py`, y datos/figuras de apoyo.                                                           |
| `comp_physics_python/` | Puertos en Python de _Computational Physics_ de Jos Thijssen, organizados por capítulo (ver [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/`            | Paquete fuente de Multiwfn 3.8 para desarrolladores más manuales para referencia local.                                                                      |
| `figures/`             | Salidas PNG/SVG estáticas usadas en informes/diapositivas y README.                                                                                          |
| `figs/`                | Recursos de logotipo y banner.                                                                                                                               |
| `docs/`                | Contenido del micrositio de LazyLearn (sirve mediante GitHub Pages o cualquier host estático).                                                               |
| `i18n/`                | Archivos README localizados.                                                                                                                                 |

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
> Varias entradas de nivel superior son symlinks a directorios fuera de este repositorio. Editar esas rutas afecta a los destinos externos.

## Prerrequisitos 🧰

| Requisito                          | Notas                                                                   |
| ---------------------------------- | ----------------------------------------------------------------------- |
| Python 3.x                         | Requerido para los scripts raíz y la mayoría del trabajo con notebooks. |
| `pip` (o Conda)                    | Gestión de paquetes y entornos.                                         |
| Jupyter Lab/Notebook (opcional)    | Necesario para los flujos con notebooks.                                |
| Gaussian 16 + GaussView (opcional) | Necesario para flujos de Gaussian.                                      |

## Instalación ⚙️

### Configuración mínima de Python (ejemplos raíz)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

### ✅ Lista de verificación rápida

| Paso | Comando                                            | Objetivo                                           |
| ---- | -------------------------------------------------- | -------------------------------------------------- |
| 1    | `python -m venv .venv`                             | Crear un entorno aislado                           |
| 2    | `source .venv/bin/activate` (o equivalente del SO) | Evitar conflictos de dependencias                  |
| 3    | `pip install --upgrade pip`                        | Mantener actualizadas las herramientas de paquetes |
| 4    | `pip install qiskit pennylane numpy matplotlib`    | Instalar la base experimental principal            |
| 5    | Ejecuta un script en `examples/`                   | Validar la instalación de extremo a extremo        |

Los notebooks de Jupyter dentro de `comp_physics/` usan el mismo entorno. Inícialos con:

```bash
jupyter lab
# or
jupyter notebook
```

### Dependencias opcionales por capítulo (`comp_physics_python/`)

```bash
# conda activate quantum  # nombre de entorno local habitual en la documentación del subproyecto
pip install numpy scipy matplotlib
```

## Uso 🚀

### Flujos de ejemplo

- **QAOA con Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

No depende de Aer; usa un backend de `statevector` puro.

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
- Carpetas de tema (`bosonscattering/`, `lensless/`, `lightscattering/`, etc.) con datos y utilidades por experimento.

Si se requieren dependencias adicionales, regístralas en `comp_physics/environments.yaml`.

## Traducciones de código de libro 📚

`comp_physics_python/` es una traducción en crecimiento a Python de programas clásicos de Fortran de _Computational Physics_. Ejemplo de mapeo por capítulo:

- `ch4/`: ejemplos de Hartree-Fock.
- `ch8/`: solucionadores de dinámica molecular.
- `ch10/`: muestreadores de Monte Carlo.

Consulta [comp_physics_python/README.md](comp_physics_python/README.md) para cobertura completa de capítulos y comandos de CLI.

## Referencias de Multiwfn 🔬

`multiwfn/` conserva `Multiwfn_3.8_dev_src_Linux` junto con el manual PDF y la guía de inicio rápido. No se incluyen binarios compilados.

## Figuras 🖼️

Los recursos PNG/SVG generados viven en `figures/`, de modo que las salidas quedan versionadas junto a los scripts/notebooks que las producen.

## Configuración 🛠️

### Python y notebooks

- Los scripts de la raíz asumen el entorno venv mostrado arriba.
- Los detalles de entorno de notebooks se reparten por la documentación del proyecto; actualmente no existe un lockfile único en la raíz del repositorio.

### Gaussian runner (ruta con symlink)

`Gaussian/run_gaussian.sh` soporta:

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

Comportamiento:

- Escribe `<basename>.log` junto al input.
- Usa `GAUSS_SCRDIR` si está definido, de lo contrario usa `~/gaussian/scr` por defecto.
- Detecta `%chk=...` en el input; si existe el checkpoint, GaussView abre `.chk`; de lo contrario, `.log`.
- Si está disponible, prefiere `~/gaussian/gv/gview_safe.sh` y luego `gview.sh`.

Wrapper recomendado de GaussView:

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

### 🎬 Mapa de navegación

Úsalo como punto de arranque para el trabajo diario:

| Área                             | Comienza aquí          |
| -------------------------------- | ---------------------- |
| Demos cuánticos                  | `examples/`            |
| Notebooks de física              | `comp_physics/`        |
| Traducciones de libros           | `comp_physics_python/` |
| Herramientas de química cuántica | `multiwfn/`            |
| Resultados publicados            | `docs/`                |
| Figuras e ilustraciones          | `figures/`, `figs/`    |

### Notas de control de versiones

- Las rutas pesadas se ignoran a través de `.gitignore`, incluyendo `books/`, destinos externos con symlink (`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`) y artefactos locales como `*.chk`.
- Mantén las contribuciones centradas en carpetas rastreadas para flujos de clonación y actualización más ligeros.
- Para actualizaciones del sitio: edita `docs/`, haz vista previa local y luego haz push.

Vista previa local de docs:

```bash
python -m http.server --directory docs
```

`docs/CNAME` está configurado para `learn.lazying.art`.

## Solución de problemas 🩺

- Criterio de éxito de Gaussian: `Normal termination of Gaussian` cerca del final del `.log`.
- Si GaussView falla bajo sesiones Wayland/remotas, usa `gview_safe.sh` y pasa `--gview` explícitamente.
- Si aparecen errores de scratch en Gaussian, verifica espacio en disco y permisos en `GAUSS_SCRDIR`.
- Si hay deriva en dependencias de notebooks, usa como fuente de verdad los README de los subproyectos y anota paquetes faltantes en archivos de entorno antes de compartir.
- `comp_physics/environments.yaml` parece ser un marcador de posición en el estado actual del repositorio; confía en comandos de instalación explícitos hasta que se corrija.

## Hoja de ruta 🛣️

- Seguir ampliando la cobertura por capítulos en `comp_physics_python/` (matrices de transferencia, DMC/PIMC, FEM y más).
- Alinear convenciones de salida/gráficos entre scripts y notebooks.
- Mantener `docs/` y los README multilingües alineados con los nuevos experimentos.

## Contribución 🤝

Las issues y pull requests son bienvenidas, especialmente para:

- Comprobaciones de corrección numérica y mejoras de reproducibilidad.
- Mejores especificaciones de entorno para notebooks/scripts.
- Traducciones de capítulos adicionales y mejoras de la CLI.
- Claridad documental entre idiomas en `i18n/`.

Antes de enviar actualizaciones de contenido importante, conserva las figuras generadas en `figures/` y asegúrate de que los comandos sean ejecutables desde la raíz del repositorio a menos que se documente lo contrario.

## ❤️ Support

| Donate                                                                                                                                                                                                                                                                                                                                                     | PayPal                                                                                                                                                                                                                                                                                                                                                          | Stripe                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License 📄

No hay un archivo `LICENSE` raíz en este repositorio actualmente. Hasta que se añada una licencia, considera que los derechos de uso/redistribución no están especificados y solicita aclaración al mantenedor antes de reutilizar contenido sustancial.
