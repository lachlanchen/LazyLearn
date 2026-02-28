[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# LazyPhysics and Chemistry

[![Site](https://img.shields.io/badge/website-learn.lazying.art-0a7ea4?style=for-the-badge&logo=githubpages&logoColor=white)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Learning-16a34a?style=for-the-badge&logo=target&logoColor=white)
![Repo Type](https://img.shields.io/badge/Repo-Type-Mixed%20Format-6b7280?style=for-the-badge)
![Docs](https://img.shields.io/badge/Docs-Static%20Microsite-0ea5e9?style=for-the-badge&logo=markdown&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-11-0f766e?style=for-the-badge&logo=googletranslate&logoColor=white)

## 📌 Tóm tắt nhanh

| Tiêu chí | Nội dung repository | 
| --- | --- |
| Kiểu luồng công việc | Không gian học vật lý + hóa học tái lập được |
| Kết quả đầu ra | Script, notebook, hình ảnh đã tạo và tài liệu tĩnh |
| Mô hình hợp tác | Thử nghiệm tại gốc + xuất bản website công khai |
| Phạm vi bản dịch | Các file README mirror trong `i18n/` |

LazyPhysics and Chemistry là phần mã và notebook của **LazyLearn**: một nhật ký học vật lý và hóa học thực hành, học chậm có chủ đích. Các ghi chú sống động, thành quả và TODO được xuất bản tại [learn.lazying.art](https://learn.lazying.art) (phục vụ từ `docs/` trong repository này), trong khi các sản phẩm có thể chạy được được giữ lại ở đây để thí nghiệm luôn có nơi lưu trữ có thể tái tạo.

## Tổng quan 🧭

### LazyLearn

- **Nền tảng chính:** [learn.lazying.art](https://learn.lazying.art) — trang công khai với nội dung theo tuần, danh sách backlog và các điểm nổi bật.
- **Nguồn thật:** mọi thứ mà trang web tham chiếu đều nằm trong `examples/`, `comp_physics/`, `comp_physics_python/`, `multiwfn/`, hoặc `figures/`.
- **Luồng cập nhật:** đưa code/notebook lên trước, tạo lại biểu đồ khi cần, rồi thêm mục vào `docs/` để website phản ánh đúng công việc mới nhất.

Kho mã này được thiết kế dạng mixed-format có chủ đích, không phải một ứng dụng đóng gói đơn lẻ. Nó kết hợp script thực thi được, notebook, tài liệu tham chiếu và một microsite tĩnh trong một workspace có phiên bản.

## Tính năng ✨

- Script ví dụ lượng tử (QAOA + VQE) chạy được trên máy cá nhân phổ thông.
- Notebook vật lý tính toán và các solver hỗ trợ (ví dụ quy trình dựa trên Numerov).
- Dịch mã Python của giáo trình vật lý tính toán theo từng chương.
- Bộ mã nguồn/tài liệu Multiwfn dùng để tham khảo hậu xử lý hóa lượng tử cục bộ.
- Hình ảnh đã tạo có phiên bản cho báo cáo/slides (`figures/`).
- Bộ README đa ngôn ngữ có sẵn trong `i18n/`.
- Microsite tĩnh trong `docs/` (domain riêng: `learn.lazying.art`).

## Cấu trúc dự án 🗂️

### Nội dung chính

| Đường dẫn | Mục đích |
| --- | --- |
| `examples/` | Script Python trọng tâm (QAOA + VQE) chạy với Qiskit hoặc PennyLane. |
| `comp_physics/` | Notebook vật lý tính toán, script hỗ trợ như `numerov.py`, cùng dữ liệu/hình ảnh đi kèm. |
| `comp_physics_python/` | Bản port Python của *Computational Physics* của Jos Thijssen, tổ chức theo chương (xem [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/` | Bộ nguồn và manual Multiwfn 3.8 cho người phát triển, kèm tài liệu tham khảo cục bộ. |
| `figures/` | Kết quả PNG/SVG tĩnh dùng trong báo cáo/slides và README. |
| `figs/` | Asset logo và banner. |
| `docs/` | Nội dung microsite LazyLearn (được phục vụ bởi GitHub Pages hoặc máy chủ tĩnh khác). |
| `i18n/` | Các file README bản địa hóa. |

Bố cục đại diện:

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
> Một số mục cấp cao nhất là symlink đến thư mục bên ngoài repository. Việc chỉnh sửa trong các đường dẫn này sẽ ảnh hưởng đến target bên ngoài.

## Yêu cầu trước khi chạy 🧰

| Yêu cầu | Ghi chú |
| --- | --- |
| Python 3.x | Bắt buộc cho script ở root và hầu hết công việc notebook. |
| `pip` (hoặc Conda) | Dùng cho quản lý package/môi trường. |
| Jupyter Lab/Notebook (tùy chọn) | Cần cho luồng làm việc với notebook. |
| Gaussian 16 + GaussView (tùy chọn) | Cần cho workflow Gaussian. |

## Cài đặt ⚙️

### Thiết lập Python tối thiểu (root examples)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

### ✅ Checklist thiết lập nhanh

| Bước | Lệnh | Mục tiêu |
| --- | --- | --- |
| 1 | `python -m venv .venv` | Tạo môi trường cô lập |
| 2 | `source .venv/bin/activate` (hoặc tương đương theo OS) | Tránh xung đột dependency |
| 3 | `pip install --upgrade pip` | Đảm bảo công cụ package luôn cập nhật |
| 4 | `pip install qiskit pennylane numpy matplotlib` | Cài stack thử nghiệm nền tảng |
| 5 | Chạy một script trong `examples/` | Kiểm tra cài đặt end-to-end |

Jupyter notebook trong `comp_physics/` sử dụng cùng môi trường này. Khởi chạy bằng:

```bash
jupyter lab
# or
jupyter notebook
```

### Dependency tuỳ chọn theo chương (`comp_physics_python/`)

```bash
# conda activate quantum  # tên env local phổ biến trong tài liệu subproject
pip install numpy scipy matplotlib
```

## Cách dùng 🚀

### Các ví dụ luồng làm việc

- **QAOA với Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

Không cần Aer; dùng backend statevector thuần.

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

Tất cả script đều ghi lại chỉ số trung gian để bạn có thể tái sử dụng biểu đồ hoặc mở rộng cho phân tử/đồ thị mới.

## Notebook vật lý tính toán 📓

Thư mục `comp_physics/` phản ánh các ghi chú đang làm việc:

- `comp_physics_textbook_code/` — các routine tái sử dụng trích xuất từ notebook.
- Notebook độc lập như `chapter1.ipynb`, `chapter2.ipynb`, `numerov.ipynb` và `numpy_1ddft.ipynb`.
- Các thư mục chủ đề (`bosonscattering/`, `lensless/`, `lightscattering/`, ... ) chứa dữ liệu và helper theo từng thí nghiệm.

Nếu cần thêm dependency, hãy ghi lại vào `comp_physics/environments.yaml`.

## Dịch mã giáo trình 📚

`comp_physics_python/` là bản dịch Python đang mở rộng từ các chương trình Fortran kinh điển của *Computational Physics*. Ví dụ ánh xạ theo chương:

- `ch4/`: ví dụ Hartree-Fock.
- `ch8/`: solver molecular dynamics.
- `ch10/`: bộ lấy mẫu Monte Carlo.

Xem [comp_physics_python/README.md](comp_physics_python/README.md) để biết đầy đủ phạm vi chương và lệnh CLI.

## Tài liệu tham chiếu Multiwfn 🔬

`multiwfn/` lưu `Multiwfn_3.8_dev_src_Linux` cùng PDF manual và hướng dẫn khởi động nhanh. Không commit file binary đã biên dịch.

## Hình ảnh 🖼️

Các asset PNG/SVG đã tạo được lưu trong `figures/` để output có version cùng script/notebook sinh ra.

## Cấu hình 🛠️

### Python và notebook

- Script root giả định dùng venv đã nêu ở trên.
- Chi tiết môi trường notebook được phân tán trong tài liệu các dự án con; hiện chưa có một lockfile duy nhất ở root.

### Trình chạy Gaussian (đường dẫn symlink)

`Gaussian/run_gaussian.sh` hỗ trợ:

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

Hành vi:

- Ghi `<basename>.log` cạnh input.
- Dùng `GAUSS_SCRDIR` nếu đã đặt, nếu không mặc định là `~/gaussian/scr`.
- Tự động nhận diện `%chk=...` trong input; nếu checkpoint tồn tại, GaussView sẽ mở `.chk`, ngược lại mở `.log`.
- Nếu có, ưu tiên `~/gaussian/gv/gview_safe.sh`, rồi đến `gview.sh`.

Bộ wrapper GaussView được khuyến nghị:

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

### 🎬 Bản đồ điều hướng

Dùng mục này như điểm khởi động cho công việc hằng ngày:

| Khu vực | Bắt đầu từ |
| --- | --- |
| Demo lượng tử | `examples/` |
| Notebook vật lý | `comp_physics/` |
| Dịch bản textbook | `comp_physics_python/` |
| Công cụ hóa lượng tử | `multiwfn/` |
| Kết quả đã xuất bản | `docs/` |
| Hình ảnh minh họa | `figures/`, `figs/` |

### Ghi chú quản lý phiên bản

- Các path lớn bị bỏ qua qua `.gitignore`, gồm `books/`, các symlink target bên ngoài (`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`) và artifact cục bộ như `*.chk`.
- Giữ đóng góp tập trung vào các thư mục được theo dõi để quy trình clone/pull nhẹ hơn.
- Khi cập nhật website: chỉnh `docs/`, xem trước local, rồi push.

Xem trước tài liệu local:

```bash
python -m http.server --directory docs
```

`docs/CNAME` đã được cấu hình cho `learn.lazying.art`.

## Xử lý sự cố 🩺

- Tiêu chí thành công của Gaussian: `Normal termination of Gaussian` gần cuối file `.log`.
- Nếu GaussView lỗi trong phiên Wayland/remote, dùng `gview_safe.sh` và truyền `--gview` một cách rõ ràng.
- Nếu lỗi về scratch Gaussian, kiểm tra dung lượng ổ trống và quyền truy cập tại `GAUSS_SCRDIR`.
- Nếu dependencies của notebook bị lệch, coi các README của subproject là nguồn chân lý và ghi nhận package thiếu trong file môi trường trước khi chia sẻ.
- `comp_physics/environments.yaml` hiện có vẻ là file placeholder trong trạng thái repo hiện tại; hãy dựa vào các lệnh cài đặt rõ ràng cho tới khi được chỉnh sửa.

## Lộ trình 🛣️

- Tiếp tục mở rộng phạm vi chương của `comp_physics_python/` (transfer matrices, DMC/PIMC, FEM, và hơn nữa).
- Đồng bộ quy ước đầu ra/plot giữa các script và notebook.
- Thêm các kiểm tra kiểm chứng nhẹ, có thể lặp lại cho các ví dụ trọng điểm.
- Giữ `docs/` và README đa ngôn ngữ đồng bộ khi có thí nghiệm mới.

## Đóng góp 🤝

Issues và pull request luôn được hoan nghênh, đặc biệt cho:

- Kiểm tra độ đúng đắn số học và cải thiện khả năng tái lập.
- Đặc tả môi trường tốt hơn cho notebook/script.
- Mở rộng thêm các bản dịch chapter trong giáo trình và tinh chỉnh CLI.
- Làm rõ tài liệu trên nhiều ngôn ngữ trong `i18n/`.

Trước khi nộp cập nhật nội dung lớn, giữ file hình đã tạo trong `figures/` và đảm bảo các lệnh có thể chạy từ root repository trừ khi có tài liệu khác.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License 📄

Hiện chưa có file `LICENSE` ở root của repository này. Cho đến khi có license được thêm vào, hãy xem quyền sử dụng/phân phối lại là chưa được xác định rõ và xin xác nhận từ maintainer trước khi tái sử dụng nội dung đáng kể.
