[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# LazyPhysics et chimie

[![Site](https://img.shields.io/badge/website-learn.lazying.art-0a7ea4?style=for-the-badge&logo=githubpages&logoColor=white)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Learning-16a34a?style=for-the-badge&logo=target&logoColor=white)
![Repo Type](https://img.shields.io/badge/Repo-Type-Mixed%20Format-6b7280?style=for-the-badge)
![Docs](https://img.shields.io/badge/Docs-Static%20Microsite-0ea5e9?style=for-the-badge&logo=markdown&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-11-0f766e?style=for-the-badge&logo=googletranslate&logoColor=white)

## 📌 En bref

| Axe                      | Ce que fait ce dépôt                                      |
| ------------------------ | --------------------------------------------------------- |
| Type de flux de travail  | Espace d'apprentissage reproductible en physique + chimie |
| Livrables                | Scripts, notebooks, figures générées et docs statiques    |
| Modèle de collaboration  | Expérimentations de base + publication du site public     |
| Couverture de traduction | Fichiers miroir du README dans `i18n/`                    |

LazyPhysics et Chimie représente la partie code + notebook de **LazyLearn** : un carnet d'apprentissage pratique, volontairement progressif, pour la physique et la chimie. Les notes vivantes, les réussites et les TODO sont publiés sur [learn.lazying.art](https://learn.lazying.art) (gérés depuis `docs/` dans ce dépôt), tandis que les éléments exécutables restent ici pour que les expériences aient toujours un point de départ reproductible.

## Aperçu 🧭

### LazyLearn

- **Base principale :** [learn.lazying.art](https://learn.lazying.art) - le site public avec les focales hebdomadaires, le carnet de backlog et les points forts.
- **Source unique de vérité :** tout ce que le site référence se trouve dans `examples/`, `comp_physics/`, `comp_physics_python/`, `multiwfn/`, ou `figures/`.
- **Flux de mise à jour :** publier d'abord le code/les notebooks, régénérer les figures si nécessaire, puis ajouter une entrée dans `docs/` pour que le site reflète le travail le plus récent.

Ce dépôt est volontairement de format mixte, pas une application empaquetée unique. Il combine scripts exécutables, notebooks, références et un site statique sous un même espace de travail versionné.

## Fonctionnalités ✨

- Scripts d'exemples quantiques (QAOA + VQE) qui s'exécutent sur des ordinateurs portables grand public.
- Notebooks de physique computationnelle et solveurs d'assistance (par ex. des flux basés sur Numerov).
- Portages Python chapitre par chapitre des programmes de physique computationnelle des manuels.
- Bundle de source/manuels Multiwfn pour la post-traitement local de chimie quantique.
- Figures générées versionnées pour rapports/présentations (`figures/`).
- Jeux de README multilingues natifs dans `i18n/`.
- Microsite statique dans `docs/` (domaine personnalisé : `learn.lazying.art`).

## Structure du projet 🗂️

### Ce que contient ce dépôt

| Chemin                 | Rôle                                                                                                                                                      |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `examples/`            | Scripts Python ciblés (QAOA + VQE) qui s'exécutent avec Qiskit ou PennyLane.                                                                              |
| `comp_physics/`        | Notebooks de physique computationnelle, scripts d'assistance comme `numerov.py`, et données/figures associées.                                            |
| `comp_physics_python/` | Portages Python de _Computational Physics_ de Jos Thijssen, organisés par chapitre (voir [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/`            | Paquet source développeur de Multiwfn 3.8 avec manuels pour référence locale.                                                                             |
| `figures/`             | Sorties PNG/SVG statiques utilisées dans les rapports/présentations et le README.                                                                         |
| `figs/`                | Actifs de logo et bannière.                                                                                                                               |
| `docs/`                | Contenu du microsite LazyLearn (servi via GitHub Pages ou tout hôte statique).                                                                            |
| `i18n/`                | Fichiers README localisés.                                                                                                                                |

Disposition représentative :

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
> Plusieurs entrées de niveau supérieur sont des liens symboliques vers des répertoires extérieurs à ce dépôt. Modifier ces chemins affecte les cibles externes.

## Prérequis 🧰

| Exigence                            | Remarques                                                   |
| ----------------------------------- | ----------------------------------------------------------- |
| Python 3.x                          | Requis pour les scripts racine et la plupart des notebooks. |
| `pip` (ou Conda)                    | Gestionnaire de paquets/environnements.                     |
| Jupyter Lab/Notebook (optionnel)    | Nécessaire pour les flux notebook.                          |
| Gaussian 16 + GaussView (optionnel) | Nécessaire pour les flux Gaussian.                          |

## Installation ⚙️

### Configuration Python minimale (exemples racine)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

### ✅ Liste de vérification rapide

| Étape | Commande                                               | Objectif                                |
| ----- | ------------------------------------------------------ | --------------------------------------- |
| 1     | `python -m venv .venv`                                 | Créer un environnement isolé            |
| 2     | `source .venv/bin/activate` (ou équivalent selon l'OS) | Éviter les conflits de dépendances      |
| 3     | `pip install --upgrade pip`                            | Assurer des outils de paquets à jour    |
| 4     | `pip install qiskit pennylane numpy matplotlib`        | Installer la pile expérimentale de base |
| 5     | Exécuter un script dans `examples/`                    | Valider l'installation de bout en bout  |

Les notebooks Jupyter dans `comp_physics/` utilisent le même environnement. Lancez avec :

```bash
jupyter lab
# ou
jupyter notebook
```

### Dépendances optionnelles par chapitre (`comp_physics_python/`)

```bash
# conda activate quantum  # nom d'environnement local courant dans les sous-docs
pip install numpy scipy matplotlib
```

## Utilisation 🚀

### Exemples de flux de travail

- **QAOA avec Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

Aucune dépendance à Aer ; utilise un backend statevector pur.

- **QAOA avec PennyLane**

```bash
python examples/qaoa_pennylane_maxcut.py
```

Utilise `default.qubit`.

- **VQE pour H2**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

Reproduit `figures/pennylane_h2_vqe_convergence.png`.

Tous les scripts enregistrent des métriques intermédiaires afin que vous puissiez réutiliser les courbes ou étendre vers de nouvelles molécules/graphes.

## Notebooks de physique computationnelle 📓

Le répertoire `comp_physics/` reflète des notes de travail :

- `comp_physics_textbook_code/` - routines réutilisables extraites des notebooks.
- Notebooks indépendants comme `chapter1.ipynb`, `chapter2.ipynb`, `numerov.ipynb` et `numpy_1ddft.ipynb`.
- Dossiers thématiques (`bosonscattering/`, `lensless/`, `lightscattering/`, etc.) avec données et scripts d'assistance par expérience.

Si des dépendances supplémentaires sont nécessaires, consignez-les dans `comp_physics/environments.yaml`.

## Traductions de programmes de manuels 📚

`comp_physics_python/` est une traduction Python progressive des programmes Fortran classiques de _Computational Physics_. Exemple de correspondance par chapitre :

- `ch4/` : exemples Hartree-Fock.
- `ch8/` : solveurs de dynamique moléculaire.
- `ch10/` : échantillonneurs Monte Carlo.

Consultez [comp_physics_python/README.md](comp_physics_python/README.md) pour la couverture complète des chapitres et les commandes CLI.

## Références Multiwfn 🔬

`multiwfn/` conserve `Multiwfn_3.8_dev_src_Linux` avec le manuel PDF et le guide de démarrage rapide. Aucun binaire compilé n'est versionné.

## Figures 🖼️

Les ressources PNG/SVG générées résident dans `figures/`, de sorte que les sorties sont versionnées en même temps que les scripts/notebooks qui les produisent.

## Configuration 🛠️

### Python et notebooks

- Les scripts racine supposent l'environnement virtuel présenté ci-dessus.
- Les détails d'environnement des notebooks sont documentés dans les sous-projets ; aucun fichier lockfile unique n'existe actuellement à la racine.

### Runner Gaussian (chemin en symlink)

`Gaussian/run_gaussian.sh` prend en charge :

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

Comportement :

- Génère `<basename>.log` à côté de l'entrée.
- Utilise `GAUSS_SCRDIR` si défini, sinon `~/gaussian/scr` par défaut.
- Détecte `%chk=...` dans l'entrée ; si le checkpoint existe, GaussView ouvre `.chk`, sinon `.log`.
- Si disponible, préfère `~/gaussian/gv/gview_safe.sh`, puis `gview.sh`.

Wrapper GaussView recommandé :

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## Notes de développement 🧪

### 🎬 Carte de navigation

Utilisez ceci comme point de départ pour le travail quotidien :

| Domaine                    | Commencer ici          |
| -------------------------- | ---------------------- |
| Démonstrations quantiques  | `examples/`            |
| Notebooks de physique      | `comp_physics/`        |
| Traductions de manuels     | `comp_physics_python/` |
| Outils de chimie quantique | `multiwfn/`            |
| Sorties publiées           | `docs/`                |
| Illustrations et visuels   | `figures/`, `figs/`    |

### Notes de contrôle de version

- Les gros chemins sont ignorés via `.gitignore`, notamment `books/`, les cibles symlink externes (`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`) et les artefacts locaux tels que `*.chk`.
- Gardez les contributions concentrées sur les dossiers suivis pour des flux clone/mise à jour légers.
- Pour les mises à jour du site web : modifiez `docs/`, prévisualisez localement, puis poussez.

Prévisualisation locale des docs :

```bash
python -m http.server --directory docs
```

`docs/CNAME` est configuré pour `learn.lazying.art`.

## Résolution de problèmes 🩺

- Critère de succès Gaussian : `Normal termination of Gaussian` près de la fin du `.log`.
- Si GaussView échoue sous Wayland/séances à distance, utilisez `gview_safe.sh` et passez `--gview` explicitement.
- Si des erreurs apparaissent au niveau du scratch Gaussian, vérifiez l'espace disque et les permissions dans `GAUSS_SCRDIR`.
- Si les dépendances des notebooks dérivent, traitez les README de sous-projets comme source de vérité et capturez les packages manquants dans des fichiers d'environnement avant de partager.
- `comp_physics/environments.yaml` semble être un placeholder dans l'état actuel du dépôt ; basez-vous sur les commandes d'installation explicites tant qu'il n'est pas corrigé.

## Feuille de route 🛣️

- Continuer l'expansion de la couverture des chapitres de `comp_physics_python/` (matrices de transfert, DMC/PIMC, FEM, et au-delà).
- Harmoniser les conventions de sortie/graphique entre scripts et notebooks.

## Contribution 🤝

Les issues et pull requests sont les bienvenus, notamment pour :

- Contrôles de précision numérique et améliorations de reproductibilité.
- De meilleures spécifications d'environnement pour les notebooks/scripts.
- Des ports de chapitres de manuels supplémentaires et des raffinements CLI.
- La clarté de la documentation entre les langues dans `i18n/`.

Avant de soumettre des mises à jour de contenu majeures, conservez les figures générées dans `figures/` et assurez-vous que les commandes sont exécutables depuis la racine du dépôt sauf documentation contraire.

## ❤️ Support

| Donate                                                                                                                                                                                                                                                                                                                                                     | PayPal                                                                                                                                                                                                                                                                                                                                                          | Stripe                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License 📄

Aucun fichier `LICENSE` racine n'est actuellement présent dans ce dépôt. Jusqu'à l'ajout d'une licence, considérez les droits d'usage et de redistribution comme non précisés et demandez clarification au mainteneur avant de réutiliser un contenu substantiel.
