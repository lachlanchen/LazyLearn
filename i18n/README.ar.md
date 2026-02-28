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

يمثّل LazyPhysics and Chemistry الجانب البرمجي + الدفاتر من **LazyLearn**: سجل تعلم عملي ومتعمد البطء في الفيزياء والكيمياء. تُنشر الملاحظات الحية والإنجازات وقائمة المهام على [learn.lazying.art](https://learn.lazying.art) (وتُخدَّم من `docs/` في هذا المستودع)، بينما تبقى المخرجات القابلة للتشغيل هنا لضمان وجود بيئة قابلة لإعادة الإنتاج للتجارب دائمًا.

## نظرة عامة 🧭

### LazyLearn

- **المنصة الأساسية:** [learn.lazying.art](https://learn.lazying.art) - الموقع العام الذي يضم تركيزات أسبوعية، قائمة الأعمال، وأبرز المستجدات.
- **المصدر المرجعي:** كل ما يشير إليه الموقع موجود في `examples/` أو `comp_physics/` أو `comp_physics_python/` أو `multiwfn/` أو `figures/`.
- **تدفق التحديث:** انشر الشيفرة/الدفاتر أولًا، أعد توليد الرسوم عند الحاجة، ثم أضف إدخالًا في `docs/` ليعكس الموقع أحدث عمل.

هذا المستودع متعمد أن يكون متعدد الصيغ وليس تطبيقًا مُعبّأً واحدًا. فهو يجمع سكربتات قابلة للتنفيذ، دفاتر، مراجع، وموقع توثيقي ثابت ضمن مساحة عمل واحدة مُدارة بالإصدارات.

## الميزات ✨

- سكربتات أمثلة كمية (QAOA + VQE) تعمل على حواسيب محمولة عادية.
- دفاتر فيزياء حاسوبية ومحللات مساعدة (مثل سير العمل المبني على Numerov).
- تحويلات Python برنامجًا ببرنامج من كتاب الفيزياء الحاسوبية بحسب الفصول.
- حزمة مصدر/دليل Multiwfn كمرجع محلي للمعالجة اللاحقة في الكيمياء الكمّية.
- رسوم مولدة ومُدارة بالإصدارات للتقارير/العروض (`figures/`).
- مجموعة README متعددة اللغات مدمجة تحت `i18n/`.
- موقع مصغر ثابت في `docs/` (نطاق مخصص: `learn.lazying.art`).

## بنية المشروع 🗂️

### ماذا يوجد هنا

| Path | Purpose |
| --- | --- |
| `examples/` | سكربتات Python مركزة (QAOA + VQE) تعمل مع Qiskit أو PennyLane. |
| `comp_physics/` | دفاتر الفيزياء الحاسوبية، سكربتات مساعدة مثل `numerov.py`، وبيانات/رسوم داعمة. |
| `comp_physics_python/` | تحويلات Python لكتاب *Computational Physics* لـ Jos Thijssen، منظمة حسب الفصول (انظر [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/` | حزمة المصدر التطويرية Multiwfn 3.8 بالإضافة إلى الأدلة للاستخدام المرجعي المحلي. |
| `figures/` | مخرجات PNG/SVG ثابتة مستخدمة في التقارير/العروض وREADME. |
| `figs/` | أصول الشعار والبانر. |
| `docs/` | محتوى الموقع المصغر لـ LazyLearn (يُخدَّم عبر GitHub Pages أو أي مستضيف ثابت). |
| `i18n/` | ملفات README المترجمة. |

تخطيط تمثيلي:

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
> عدة مدخلات في المستوى الأعلى هي روابط رمزية إلى مجلدات خارج هذا المستودع. التعديل تحت تلك المسارات يؤثر على أهداف خارجية.

## المتطلبات المسبقة 🧰

| Requirement | Notes |
| --- | --- |
| Python 3.x | مطلوب لسكربتات الجذر ومعظم أعمال الدفاتر. |
| `pip` (or Conda) | إدارة الحزم/البيئات. |
| Jupyter Lab/Notebook (optional) | مطلوب لسير عمل الدفاتر. |
| Gaussian 16 + GaussView (optional) | مطلوب لمسارات عمل Gaussian. |

## التثبيت ⚙️

### إعداد Python الأدنى (أمثلة الجذر)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

تستخدم دفاتر Jupyter داخل `comp_physics/` نفس البيئة. شغّلها عبر:

```bash
jupyter lab
# or
jupyter notebook
```

### تبعيات اختيارية لتحويلات الفصول (`comp_physics_python/`)

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## الاستخدام 🚀

### مسارات عمل أمثلة

- **QAOA مع Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

لا توجد تبعية Aer؛ يستخدم واجهة pure statevector.

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

تسجّل كل السكربتات المقاييس الوسيطة بحيث يمكنك إعادة استخدام الرسوم أو التوسّع إلى جزيئات/رسوم بيانية جديدة.

## دفاتر الفيزياء الحاسوبية 📓

يعكس مجلد `comp_physics/` ملاحظات العمل الفعلية:

- `comp_physics_textbook_code/` - روتينات قابلة لإعادة الاستخدام مُستخرجة من الدفاتر.
- دفاتر مستقلة مثل `chapter1.ipynb` و`chapter2.ipynb` و`numerov.ipynb` و`numpy_1ddft.ipynb`.
- مجلدات موضوعية (`bosonscattering/` و`lensless/` و`lightscattering/` وغيرها) مع البيانات والمساعدات لكل تجربة.

إذا لزمَت تبعيات إضافية، سجّلها في `comp_physics/environments.yaml`.

## ترجمات شيفرة الكتاب 📚

`comp_physics_python/` مشروع متنامٍ لتحويل برامج فورتران الكلاسيكية من *Computational Physics* إلى Python. مثال على توزيع الفصول:

- `ch4/`: أمثلة Hartree-Fock.
- `ch8/`: محللات الديناميكا الجزيئية.
- `ch10/`: عينات Monte Carlo.

ارجع إلى [comp_physics_python/README.md](comp_physics_python/README.md) لتغطية الفصول كاملة وأوامر CLI.

## مراجع Multiwfn 🔬

يحفظ `multiwfn/` مجلد `Multiwfn_3.8_dev_src_Linux` بالإضافة إلى دليل PDF ودليل البدء السريع. لا يتم تضمين أي ملفات تنفيذية مترجمة ضمن المستودع.

## الرسوم 🖼️

توجد أصول PNG/SVG المولدة في `figures/` بحيث تُدار المخرجات بالإصدارات جنبًا إلى جنب مع السكربتات/الدفاتر التي أنشأتها.

## الإعدادات 🛠️

### Python والدفاتر

- تفترض سكربتات الجذر بيئة venv الموضحة أعلاه.
- تفاصيل بيئة الدفاتر موزعة على وثائق المشاريع الفرعية؛ لا يوجد حاليًا ملف lockfile واحد على جذر المستودع.

### مشغّل Gaussian (مسار مرتبط رمزيًا)

`Gaussian/run_gaussian.sh` يدعم:

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

السلوك:

- يكتب `<basename>.log` بجانب ملف الإدخال.
- يستخدم `GAUSS_SCRDIR` إذا كان مضبوطًا، وإلا تكون القيمة الافتراضية `~/gaussian/scr`.
- يكتشف `%chk=...` في الإدخال؛ إذا كان ملف checkpoint موجودًا يفتح GaussView ملف `.chk`، وإلا يفتح `.log`.
- إذا كان متاحًا، يفضّل `~/gaussian/gv/gview_safe.sh` ثم `gview.sh`.

مغلّف GaussView الموصى به:

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

### ملاحظات التحكم بالإصدارات

- المسارات الكبيرة يتم تجاهلها عبر `.gitignore`، بما يشمل `books/`، أهداف الروابط الرمزية الخارجية (`Gaussian` و`ComputationalPhysics` و`leonardsusskind` و`the_theoretical_minimum`) وملفات محلية مثل `*.chk`.
- اجعل المساهمات مركزة على المجلدات المتتبَّعة للحفاظ على سير استنساخ/تحديث خفيف.
- لتحديثات الموقع: عدّل `docs/`، عاين محليًا، ثم ادفع التغييرات.

معاينة التوثيق محليًا:

```bash
python -m http.server --directory docs
```

تم إعداد `docs/CNAME` للنطاق `learn.lazying.art`.

## استكشاف الأخطاء وإصلاحها 🩺

- معيار نجاح Gaussian: ظهور `Normal termination of Gaussian` قرب نهاية ملف `.log`.
- إذا فشل GaussView تحت Wayland/جلسات بعيدة، استخدم `gview_safe.sh` ومرر `--gview` بشكل صريح.
- إذا ظهرت أخطاء scratch في Gaussian، تحقق من المساحة الحرة وصلاحيات `GAUSS_SCRDIR`.
- إذا انجرفت تبعيات الدفاتر، اعتبر ملفات README الخاصة بالمشاريع الفرعية هي المرجع الأساسي وسجّل الحزم الناقصة في ملفات البيئة قبل المشاركة.
- يبدو أن `comp_physics/environments.yaml` عنصر placeholder في حالة المستودع الحالية؛ اعتمد أوامر التثبيت الصريحة حتى يتم تصحيحه.

## خارطة الطريق 🛣️

- متابعة توسيع تغطية فصول `comp_physics_python/` (transfer matrices وDMC/PIMC وFEM وما بعد ذلك).
- توحيد معايير المخرجات/الرسوم بين السكربتات والدفاتر.
- إضافة فحوص تحقق خفيفة وقابلة للتكرار للأمثلة الأساسية.
- إبقاء `docs/` وملفات README متعددة اللغات متزامنة مع التجارب الجديدة.

## المساهمة 🤝

الـ Issues والـ Pull Requests مرحّب بها، خصوصًا في:

- فحوص الدقة العددية وتحسينات قابلية إعادة الإنتاج.
- مواصفات بيئة أفضل للدفاتر/السكربتات.
- تحويلات فصول إضافية من الكتاب وتحسينات CLI.
- وضوح التوثيق عبر اللغات داخل `i18n/`.

قبل إرسال تحديثات محتوى كبيرة، احتفظ بالرسوم المولدة داخل `figures/` وتأكد من قابلية تشغيل الأوامر من جذر المستودع ما لم يتم توثيق خلاف ذلك.

## ادعم LazyLearn ❤️

مساندة LazyLearn تساعد على استمرار التجارب والتوثيق والأدوات المفتوحة:

- تغطية الاستضافة/الاستدلال/التخزين للعروض العامة والدفاتر.
- تمويل أسابيع تطوير مركزة على EchoMind وLazyEdit وأدوات الكم/الفيزياء هنا.
- بناء نماذج أولية للبصريات + الأجهزة القابلة للارتداء (IdeasGlass وLightMind) التي تغذي الفصول المستقبلية.
- رعاية نشر مجاني للطلاب ومختبرات المجتمع وصنّاع المحتوى.

### التبرع

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

**الدعم / Donate**

- ご支援は研究・開発と運用の継続に役立ち、より多くのオープンなプロジェクトを皆さんに届ける力になります。
- 你的支持将用于研发与运维，帮助我持续公开分享更多项目与改进。
- دعمك يضمن استمرار البحث والتطوير والتشغيل حتى أتمكن من مشاركة مزيد من المشاريع والتحسينات المفتوحة.

## الترخيص 📄

لا يوجد ملف `LICENSE` في جذر هذا المستودع حاليًا. إلى أن تتم إضافة ترخيص، اعتبر حقوق الاستخدام/إعادة التوزيع غير محددة واطلب توضيحًا من المشرف قبل إعادة استخدام محتوى جوهري.
