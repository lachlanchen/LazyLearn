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

LazyPhysics and Chemistry là phần mã nguồn + notebook của **LazyLearn**: một nhật ký học vật lý và hóa học thực hành, theo nhịp độ có chủ ý chậm rãi. Ghi chú sống, các mốc tiến triển và TODO được xuất bản tại [learn.lazying.art](https://learn.lazying.art) (được phục vụ từ `docs/` trong repo này), trong khi các artifact có thể chạy được nằm ở đây để mọi thử nghiệm luôn có nơi tái lập rõ ràng.

## Tổng quan 🧭

### LazyLearn

- **Căn cứ chính:** [learn.lazying.art](https://learn.lazying.art) - trang public với trọng tâm theo tuần, backlog và các điểm nổi bật.
- **Nguồn chuẩn:** mọi thứ mà trang web liên kết đều nằm trong `examples/`, `comp_physics/`, `comp_physics_python/`, `multiwfn/`, hoặc `figures/`.
- **Luồng cập nhật:** đưa code/notebook lên trước, tái tạo biểu đồ nếu cần, rồi thêm mục vào `docs/` để trang web phản ánh công việc mới nhất.

Repository này được thiết kế dạng mixed-format có chủ ý, không phải một ứng dụng đóng gói đơn lẻ. Nó kết hợp script có thể thực thi, notebook, tài liệu tham khảo và một static docs site trong cùng một workspace có quản lý phiên bản.

## Tính năng ✨

- Script ví dụ lượng tử (QAOA + VQE) chạy được trên laptop phổ thông.
- Notebook vật lý tính toán và các bộ giải hỗ trợ (ví dụ workflow dựa trên Numerov).
- Bản chuyển Python theo từng chương của các chương trình vật lý tính toán trong giáo trình.
- Gói mã nguồn/tài liệu Multiwfn để tham khảo hậu xử lý hóa lượng tử cục bộ.
- Hình đã tạo có version cho báo cáo/slide (`figures/`).
- Bộ README đa ngôn ngữ tích hợp sẵn trong `i18n/`.
- Microsite tĩnh trong `docs/` (custom domain: `learn.lazying.art`).

## Cấu trúc dự án 🗂️

### Nội dung trong repo

| Path | Mục đích |
| --- | --- |
| `examples/` | Các script Python tập trung (QAOA + VQE) chạy với Qiskit hoặc PennyLane. |
| `comp_physics/` | Notebook vật lý tính toán, script hỗ trợ như `numerov.py`, cùng dữ liệu/hình đi kèm. |
| `comp_physics_python/` | Bản chuyển Python của *Computational Physics* (Jos Thijssen), tổ chức theo chương (xem [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/` | Gói mã nguồn Multiwfn 3.8 dành cho developer cùng manuals để tham khảo cục bộ. |
| `figures/` | Các output PNG/SVG tĩnh dùng trong báo cáo/slide và README. |
| `figs/` | Tài nguyên logo và banner. |
| `docs/` | Nội dung microsite LazyLearn (phục vụ bằng GitHub Pages hoặc bất kỳ static host nào). |
| `i18n/` | Các file README bản địa hóa. |

Bố cục tiêu biểu:

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
> Một số mục ở cấp cao nhất là symlink tới thư mục bên ngoài repository này. Chỉnh sửa dưới các path đó sẽ tác động tới mục tiêu bên ngoài.

## Yêu cầu trước khi chạy 🧰

| Requirement | Ghi chú |
| --- | --- |
| Python 3.x | Bắt buộc cho script ở root và phần lớn workflow notebook. |
| `pip` (hoặc Conda) | Quản lý package/môi trường. |
| Jupyter Lab/Notebook (tùy chọn) | Cần cho workflow notebook. |
| Gaussian 16 + GaussView (tùy chọn) | Cần cho workflow Gaussian. |

## Cài đặt ⚙️

### Thiết lập Python tối thiểu (ví dụ ở root)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

Notebook Jupyter trong `comp_physics/` dùng cùng môi trường. Chạy bằng:

```bash
jupyter lab
# or
jupyter notebook
```

### Phụ thuộc tùy chọn cho bản chuyển theo chương (`comp_physics_python/`)

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## Cách dùng 🚀

### Workflow ví dụ

- **QAOA với Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

Không phụ thuộc Aer; dùng backend statevector thuần.

- **QAOA với PennyLane**

```bash
python examples/qaoa_pennylane_maxcut.py
```

Dùng `default.qubit`.

- **VQE cho H2**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

Tái tạo `figures/pennylane_h2_vqe_convergence.png`.

Tất cả script đều ghi log metric trung gian để bạn tái sử dụng biểu đồ hoặc mở rộng sang phân tử/đồ thị mới.

## Notebook vật lý tính toán 📓

Thư mục `comp_physics/` phản chiếu ghi chú làm việc:

- `comp_physics_textbook_code/` - các routine tái sử dụng được tách ra từ notebook.
- Các notebook độc lập như `chapter1.ipynb`, `chapter2.ipynb`, `numerov.ipynb`, và `numpy_1ddft.ipynb`.
- Các thư mục theo chủ đề (`bosonscattering/`, `lensless/`, `lightscattering/`, v.v.) với dữ liệu và helper cho từng thử nghiệm.

Nếu cần thêm phụ thuộc, hãy ghi lại trong `comp_physics/environments.yaml`.

## Bản chuyển mã giáo trình 📚

`comp_physics_python/` là bản dịch Python đang mở rộng của các chương trình Fortran kinh điển từ *Computational Physics*. Ví dụ mapping theo chương:

- `ch4/`: ví dụ Hartree-Fock.
- `ch8/`: bộ giải molecular dynamics.
- `ch10/`: bộ lấy mẫu Monte Carlo.

Xem [comp_physics_python/README.md](comp_physics_python/README.md) để biết đầy đủ phạm vi chương và các lệnh CLI.

## Tài liệu tham chiếu Multiwfn 🔬

`multiwfn/` lưu `Multiwfn_3.8_dev_src_Linux` cùng PDF manual và hướng dẫn quick-start. Không có binary đã biên dịch nào được commit.

## Hình ảnh 🖼️

Tài nguyên PNG/SVG đã tạo nằm trong `figures/` để output được quản lý phiên bản cùng script/notebook tạo ra chúng.

## Cấu hình 🛠️

### Python và notebook

- Script ở root giả định dùng venv như ở trên.
- Chi tiết môi trường notebook phân tán trong tài liệu các dự án con; hiện chưa có một lockfile duy nhất ở root repo.

### Trình chạy Gaussian (đường dẫn symlink)

`Gaussian/run_gaussian.sh` hỗ trợ:

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

Hành vi:

- Ghi `<basename>.log` cạnh file input.
- Dùng `GAUSS_SCRDIR` nếu đã đặt, nếu không mặc định `~/gaussian/scr`.
- Phát hiện `%chk=...` trong input; nếu checkpoint tồn tại, GaussView mở `.chk`, nếu không mở `.log`.
- Nếu có sẵn, ưu tiên `~/gaussian/gv/gview_safe.sh`, sau đó `gview.sh`.

Wrapper GaussView được khuyến nghị:

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## Ghi chú phát triển 🧪

### Ghi chú về version control

- Các path nặng được bỏ qua qua `.gitignore`, gồm `books/`, các symlink target bên ngoài (`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`), và artifact cục bộ như `*.chk`.
- Giữ đóng góp tập trung vào các thư mục được theo dõi để việc clone/update luôn nhẹ.
- Khi cập nhật website: sửa `docs/`, xem trước cục bộ, rồi push.

Xem trước docs cục bộ:

```bash
python -m http.server --directory docs
```

`docs/CNAME` được cấu hình cho `learn.lazying.art`.

## Khắc phục sự cố 🩺

- Tiêu chí thành công của Gaussian: `Normal termination of Gaussian` ở gần cuối file `.log`.
- Nếu GaussView lỗi dưới Wayland/phiên remote, dùng `gview_safe.sh` và truyền `--gview` tường minh.
- Nếu có lỗi scratch của Gaussian, kiểm tra dung lượng trống và quyền trong `GAUSS_SCRDIR`.
- Nếu phụ thuộc notebook bị lệch, coi README các dự án con là nguồn chuẩn và ghi lại package còn thiếu trong file môi trường trước khi chia sẻ.
- `comp_physics/environments.yaml` có vẻ đang là placeholder trong trạng thái repo hiện tại; hãy dựa vào các lệnh cài đặt tường minh cho tới khi file này được chỉnh đúng.

## Lộ trình 🛣️

- Tiếp tục mở rộng độ phủ chương của `comp_physics_python/` (transfer matrices, DMC/PIMC, FEM, và xa hơn).
- Hài hòa quy ước output/plot giữa scripts và notebooks.
- Thêm các kiểm tra xác thực nhẹ, lặp lại được cho các ví dụ quan trọng.
- Giữ `docs/` và README đa ngôn ngữ đồng bộ với các thử nghiệm mới.

## Đóng góp 🤝

Issue và pull request đều được chào đón, đặc biệt cho:

- Kiểm tra độ đúng số học và cải thiện khả năng tái lập.
- Đặc tả môi trường tốt hơn cho notebook/scripts.
- Bổ sung bản chuyển chương giáo trình và tinh chỉnh CLI.
- Làm rõ tài liệu qua nhiều ngôn ngữ trong `i18n/`.

Trước khi gửi cập nhật nội dung lớn, hãy giữ các hình đã tạo trong `figures/` và đảm bảo lệnh chạy được từ root repository trừ khi có ghi chú khác.

## Hỗ trợ LazyLearn ❤️

Việc hỗ trợ LazyLearn giúp các thử nghiệm, tài liệu và công cụ mở tiếp tục phát triển:

- Chi trả chi phí hosting/inference/storage cho demo công khai và notebook.
- Tài trợ các đợt hack-week tập trung cho EchoMind, LazyEdit, và các utility quantum/physics ở đây.
- Prototype optics + wearables (IdeasGlass, LightMind) làm đầu vào cho các chương tương lai.
- Tài trợ triển khai miễn phí cho sinh viên, community lab và nhà sáng tạo.

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

- Hỗ trợ của bạn duy trì nghiên cứu, phát triển và vận hành để tôi có thể tiếp tục chia sẻ nhiều dự án mở và cải tiến hơn.
- ご支援は研究・開発と運用の継続に役立ち、より多くのオープンなプロジェクトを皆さんに届ける力になります。
- 你的支持将用于研发与运维，帮助我持续公开分享更多项目与改进。
- Your support sustains my research, development, and ops so I can keep sharing more open projects and improvements.

## Giấy phép 📄

Hiện tại repository này chưa có file `LICENSE` ở root. Cho đến khi có license được thêm vào, hãy xem quyền sử dụng/phân phối lại là chưa được chỉ định và liên hệ maintainer để làm rõ trước khi tái sử dụng nội dung đáng kể.
