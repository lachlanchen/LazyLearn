[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# LazyPhysics والكيمياء

[![Site](https://img.shields.io/badge/website-learn.lazying.art-0a7ea4?style=for-the-badge&logo=githubpages&logoColor=white)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Learning-16a34a?style=for-the-badge&logo=target&logoColor=white)
![Repo Type](https://img.shields.io/badge/Repo-Type-Mixed%20Format-6b7280?style=for-the-badge)
![Docs](https://img.shields.io/badge/Docs-Static%20Microsite-0ea5e9?style=for-the-badge&logo=markdown&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-11-0f766e?style=for-the-badge&logo=googletranslate&logoColor=white)

## 📌 لمحة سريعة

| البعد         | ما يقوم به هذا المستودع                                      |
| ------------- | ------------------------------------------------------------ |
| نوع سير العمل | بيئة تعلم فيزياء وكيمياء قابلة لإعادة التنفيذ                |
| المخرجات      | سكربتات، دفاتر ملاحظات، رسوم بيانية مولدة، ومواد توثيق ثابتة |
| نموذج التعاون | تجارب الجذر + نشر الموقع العام                               |
| تغطية الترجمة | ملفات README مرآوية داخل `i18n/`                             |

LazyPhysics and Chemistry هو الجزء البرمجي/دفاتر ملاحظات من مشروع **LazyLearn**: سجل تعلم عملي وبطيء فيزيائيًا وكيميائيًا عن قصد. تُنشر الملاحظات الحية والإنجازات وقوائم الأعمال القادمة على [learn.lazying.art](https://learn.lazying.art) (يُخدم الموقع من `docs/` في هذا المستودع)، بينما تبقى القطع القابلة للتشغيل هنا حتى يكون لكل تجربة موطن قابل لإعادة التنفيذ دائمًا.

## نظرة عامة 🧭

### LazyLearn

- **قاعدة العمل:** [learn.lazying.art](https://learn.lazying.art) - الموقع العام مع محاور أسبوعية وسجل عودة وخلاصات.
- **المصدر الأساسي للحقائق:** كل ما يرتبط به الموقع موجود في `examples/` و`comp_physics/` و`comp_physics_python/` و`multiwfn/` أو `figures/`.
- **تدفق التحديث:** نشر الشيفرة/دفاتر الملاحظات أولًا، وإعادة توليد الرسوم البيانية عند الحاجة، ثم إضافة إدخال إلى `docs/` بحيث يعكس الموقع أحدث عمل.

هذا المستودع مختلط الصيغة بشكل مقصود، وليس تطبيقًا واحدًا معبأً. يجمع بين سكربتات قابلة للتنفيذ، ودفاتر ملاحظات، ومراجع، وموقع وثائق ثابت ضمن مساحة عمل واحدة ذات إصدار.

## المزايا ✨

- أمثلة كمومية (QAOA + VQE) تعمل على حواسيب محمولة عادية.
- دفاتر ملاحظات فيزياء حسابية وحلول مساعدة (مثل تدفقات عمل تعتمد على Numerov).
- تحويلات بايثون فصلًا بعد فصل من برامج الفيزياء الحسابية المدرسية.
- حزمة مصدر ودليل Multiwfn لمعالجات ما بعد الكيمياء الكمية محليًا.
- رسوم توضيحية مولدة ومُدرجة بالإصدارات للتقارير/الشرائح (`figures/`).
- مجموعة README متعددة اللغات مدمجة ضمن `i18n/`.
- موقع microsite ثابت ضمن `docs/` (نطاق مخصص: `learn.lazying.art`).

## بنية المشروع 🗂️

### ما يحتويه هذا المكان

| المسار                 | الهدف                                                                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `examples/`            | سكربتات بايثون مركزة (QAOA + VQE) تعمل مع Qiskit أو PennyLane.                                                                                 |
| `comp_physics/`        | دفاتر ملاحظات فيزياء حسابية، سكربتات مساعدة مثل `numerov.py`، وبيانات/رسوم داعمة.                                                              |
| `comp_physics_python/` | تحويلات بايثون لكتاب _Computational Physics_ ليوز ثيسن، مرتبة حسب الفصل (انظر [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/`            | حزمة مصدر Multiwfn 3.8 مع الكتيبات للاستخدام المرجعي المحلي.                                                                                   |
| `figures/`             | مخرجات PNG/SVG ثابتة تُستخدم في التقارير/الشرائح وREADME.                                                                                      |
| `figs/`                | شعارات ومواد بانر.                                                                                                                             |
| `docs/`                | محتوى موقع LazyLearn الثابت (يُستضاف عبر GitHub Pages أو أي مضيف ثابت).                                                                        |
| `i18n/`                | ملفات README المترجمة.                                                                                                                         |

التخطيط التمثيلي:

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
> عدّة إدخالات في المستوى الأعلى هي روابط رمزية إلى أدلة خارج هذا المستودع. تعديل أي ملف تحت تلك المسارات يؤثر على الأهداف الخارجية.

## المتطلبات المسبقة 🧰

| المتطلب                           | ملاحظات                                            |
| --------------------------------- | -------------------------------------------------- |
| Python 3.x                        | مطلوب لسكربتات المستوى الجذري ومعظم أعمال الدفاتر. |
| `pip` (أو Conda)                  | لإدارة الحزم/البيئة.                               |
| Jupyter Lab/Notebook (اختياري)    | مطلوب لسير عمل الدفاتر.                            |
| Gaussian 16 + GaussView (اختياري) | مطلوب لسير عمل Gaussian.                           |

## التثبيت ⚙️

### إعداد بايثون أدنى حد (أمثلة الجذر)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

### ✅ قائمة مراجعة الإعداد السريعة

| الخطوة | الأمر                                                    | الهدف                                |
| ------ | -------------------------------------------------------- | ------------------------------------ |
| 1      | `python -m venv .venv`                                   | إنشاء بيئة معزولة                    |
| 2      | `source .venv/bin/activate` (أو ما يعادل ذلك حسب النظام) | تجنب تعارضات الحزم                   |
| 3      | `pip install --upgrade pip`                              | التأكد من تحديث أدوات الحزم          |
| 4      | `pip install qiskit pennylane numpy matplotlib`          | تثبيت المكدّس التجريبي الأساسي       |
| 5      | تشغيل أحد السكربتات في `examples/`                       | التحقق من التثبيت من البداية للنهاية |

دفاتر Jupyter داخل `comp_physics/` تستخدم نفس البيئة. شغّل باستخدام:

```bash
jupyter lab
# أو
jupyter notebook
```

### متطلبات إضافية لفصول `comp_physics_python/` (اختيارية)

```bash
# conda activate quantum  # اسم بيئة شائع محليًا في وثائق المشروع الفرعي
pip install numpy scipy matplotlib
```

## الاستخدام 🚀

### مثال على سير العمل

- **QAOA مع Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

لا توجد اعتماديات `Aer`؛ يستخدم محرك statevector نقي.

- **QAOA مع PennyLane**

```bash
python examples/qaoa_pennylane_maxcut.py
```

يستخدم `default.qubit`.

- **VQE لـ H2**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

يعيد إنتاج `figures/pennylane_h2_vqe_convergence.png`.

جميع السكربتات تسجل مؤشرات وسيطة بحيث يمكنك إعادة استخدام الرسوم أو التوسع إلى جزيئات/رسوم جديدة.

## دفاتر ملاحظات الفيزياء الحسابية 📓

دليل `comp_physics/` يعكس الملاحظات التشغيلية:

- `comp_physics_textbook_code/` - روتينات قابلة لإعادة الاستخدام مستخلصة من الدفاتر.
- دفاتر مستقلة مثل `chapter1.ipynb`, `chapter2.ipynb`, `numerov.ipynb`, و`numpy_1ddft.ipynb`.
- مجلدات مواضيعية (`bosonscattering/`, `lensless/`, `lightscattering/`, وغيرها) مع بيانات ومساعدات لكل تجربة.

إذا كانت هناك تبعيات إضافية مطلوبة، سجّلها في `comp_physics/environments.yaml`.

## ترجمات كود المقرر 📚

`comp_physics_python/` هو ترجمة بايثون متنامية لبرامج Fortran الكلاسيكية من _Computational Physics_. مثال على توزيع الفصول:

- `ch4/`: أمثلة Hartree-Fock.
- `ch8/`: حلول ديناميكا الجزيئات.
- `ch10/`: مولدات عيّنة Monte Carlo.

راجع [comp_physics_python/README.md](comp_physics_python/README.md) لتغطية الفصول الكاملة وأوامر CLI.

## مراجع Multiwfn 🔬

`multiwfn/` يحتفظ بـ `Multiwfn_3.8_dev_src_Linux` بالإضافة إلى دليل PDF ودليل البدء السريع. لا توجد ملفات تنفيذية مجمعة في المستودع.

## الرسوم 🖼️

الملفات المولدة PNG/SVG موجودة في `figures/` بحيث تكون المخرجات مُدرجة بالإصدار جنب السكربتات/الدفاتر المنتجة.

## الإعداد 🛠️

### بايثون والدفاتر

- سكربتات الجذر تفترض وجود البيئة الافتراضية المذكورة أعلاه.
- تفاصيل بيئة الدفاتر موزعة عبر وثائق المشروع؛ لا يوجد حاليًا ملف قفل واحد في جذر المستودع.

### مشغّل Gaussian (المسار المرتبط)

`Gaussian/run_gaussian.sh` يدعم:

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

السلوك:

- يكتب `<basename>.log` بجوار الملف المدخل.
- يستخدم `GAUSS_SCRDIR` إذا وُجد، وإلا يستخدم `~/gaussian/scr` افتراضيًا.
- يكتشف `%chk=...` في الإدخال؛ إذا كانت نقطة تفتيش موجودة، يفتح GaussView ملف `.chk`، وإلا فتح ملف `.log`.
- إذا كان متاحًا، يفضّل `~/gaussian/gv/gview_safe.sh`، ثم `gview.sh`.

المغلف الموصى به لـ GaussView:

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## ملاحظات التطوير 🧪

### 🎬 خريطة التنقل

استخدم هذا كمنصة انطلاق للعمل اليومي:

| المجال                  | ابدأ من                |
| ----------------------- | ---------------------- |
| عروض كمومية             | `examples/`            |
| دفاتر الفيزياء          | `comp_physics/`        |
| ترجمات الكتاب           | `comp_physics_python/` |
| أدوات الكيمياء الكمومية | `multiwfn/`            |
| النتائج المنشورة        | `docs/`                |
| الرسوم والتوضيحات       | `figures/`, `figs/`    |

### ملاحظات التحكم بالإصدار

- المسارات الثقيلة تُهمل عبر `.gitignore`، بما فيها `books/`، ومسارات الروابط الخارجية (`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`)، والمواد المحلية المؤقتة مثل `*.chk`.
- احرص أن تظل المساهمات مركزة على المجلدات المتتبعة للحفاظ على تحديث/استنساخ خفيف للمستودع.
- لتحديثات الموقع: عدّل `docs/`، ثم عاين محليًا، ثم ادفع.

معاينة مستندات محلية:

```bash
python -m http.server --directory docs
```

`docs/CNAME` مضبوط على `learn.lazying.art`.

## استكشاف الأخطاء وإصلاحها 🩺

- معيار نجاح Gaussian: `Normal termination of Gaussian` قرب نهاية ملف `.log`.
- إذا فشل GaussView في Wayland/جلسات عن بُعد، استخدم `gview_safe.sh` ومرر `--gview` بشكل صريح.
- إذا ظهرت أخطاء في scratch الخاصة بـ Gaussian، تحقّق من المساحة المتاحة والأذونات في `GAUSS_SCRDIR`.
- إذا انزلقت تبعيات دفاتر الملاحظات، عُد إلى READMEs المشاريع الفرعية كمصدر للحقيقة وسجّل الحزم الناقصة في ملفات البيئة قبل المشاركة.
- ملف `comp_physics/environments.yaml` يبدو أنه عنصر نائب في حالة المستودع الحالية؛ اعتمد على أوامر التثبيت الصريحة حتى تُصحّح.

## خارطة الطريق 🛣️

- مواصلة توسيع تغطية `comp_physics_python/` للفصول (مصفوفات الانتقال، DMC/PIMC، FEM، وما بعد).
- توحيد اتفاقيات المخرجات/الرسوم بين السكربتات والدفاتر.
- إضافة فحوص تحقق خفيفة وقابلة للتكرار للأمثلة المفتاحية.
- إبقاء `docs/` وملفات README متعددة اللغات متوافقة مع التجارب الجديدة.

## المساهمة 🤝

الـ Issues وطلبات السحب مرحب بها، خصوصًا في:

- فحوص الدقة العددية وتحسينات قابلية إعادة التنفيذ.
- تحسين مواصفات البيئة لدفاتر الملاحظات/السكربتات.
- مزيد من نقل فصول الكتاب وآليات CLI.
- وضوح التوثيق عبر اللغات داخل `i18n/`.

قبل إرسال تحديثات محتوى كبيرة، احتفظ بالرسوم المولدة في `figures/` وتأكد من أن الأوامر قابلة للتنفيذ من جذر المستودع ما لم يوثّق خلاف ذلك.

## الترخيص 📄

لا يوجد ملف `LICENSE` جذري حاليًا في هذا المستودع. حتى تتم إضافة الترخيص، تعامل مع حقوق الاستخدام/إعادة التوزيع على أنها غير محددة واطلب توضيحًا من المشرف قبل إعادة استخدام محتوى جوهري.

## ❤️ Support

| Donate                                                                                                                                                                                                                                                                                                                                                     | PayPal                                                                                                                                                                                                                                                                                                                                                          | Stripe                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
