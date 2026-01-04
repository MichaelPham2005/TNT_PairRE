# Baseline PairRE for ICEWS14

## Tổng quan

**Baseline PairRE** là mô hình Knowledge Graph Embedding không sử dụng thông tin thời gian (timestamps) khi huấn luyện trên dataset ICEWS14.

## Mục lục
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cách 1: Chạy trên Kaggle (Khuyến nghị)](#cách-1-chạy-trên-kaggle-khuyến-nghị)
- [Cách 2: Chạy trên máy local](#cách-2-chạy-trên-máy-local)
- [Giải thích các tham số](#giải-thích-các-tham-số)
- [Kiểm tra kết quả](#kiểm-tra-kết-quả)

---

## Yêu cầu hệ thống

### Phần cứng
- **GPU**: Khuyến nghị có GPU (CUDA) để huấn luyện nhanh hơn
- **RAM**: Tối thiểu 8GB
- **Dung lượng ổ cứng**: ~2GB cho dataset và model checkpoints

### Phần mềm
- **Python**: 3.7 trở lên
- **PyTorch**: 1.8.0 trở lên (có hỗ trợ CUDA nếu có GPU)
- **Các thư viện khác**: numpy, tqdm, scikit-learn, scipy

---

## Cách 1: Chạy trên Kaggle (Khuyến nghị)

Kaggle cung cấp GPU miễn phí và môi trường đã cài sẵn PyTorch, rất phù hợp cho việc huấn luyện model.

### Bước 1: Tạo Notebook mới trên Kaggle

1. Truy cập [Kaggle](https://www.kaggle.com/) và đăng nhập
2. Tạo notebook mới: Click **"Code"** → **"New Notebook"**
3. Bật GPU: 
   - Click vào **Settings** (biểu tượng bánh răng)
   - Chọn **Accelerator** → **GPU T4 x2** hoặc **GPU P100**

### Bước 2: Upload code lên Kaggle

1. Nén toàn bộ thư mục code thành file `.zip`
2. Upload lên Kaggle Datasets:
   - Vào **Data** → **New Dataset**
   - Upload file `.zip` của bạn
   - Đặt tên dataset, ví dụ: `baseline-icews14`
3. Add dataset vào notebook:
   - Trong notebook, click **"+ Add data"**
   - Tìm và chọn dataset vừa upload

### Bước 3: Chạy code trong Notebook

Tạo các cell và chạy lần lượt:

#### Cell 1: Copy code từ Input sang Working Directory
```python
import shutil
import os

# Thay 'baseline-icews14' bằng tên dataset của bạn
source_dir = '/kaggle/input/baseline-icews14/baseline_icews14'
destination_dir = '/kaggle/working/my_code'

# Copy toàn bộ thư mục
if not os.path.exists(destination_dir):
    shutil.copytree(source_dir, destination_dir)
    print("✓ Đã copy code sang /kaggle/working thành công!")
else:
    print("✓ Thư mục code đã tồn tại ở working directory.")

# Chuyển thư mục làm việc hiện tại
os.chdir(destination_dir)
print(f"Thư mục làm việc hiện tại: {os.getcwd()}")
```

#### Cell 2: Kiểm tra PyTorch và GPU
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

#### Cell 3: Tạo file requirements.txt
```python
%%writefile requirements.txt
tqdm
numpy
scikit-learn
scipy
```

#### Cell 4: Cài đặt dependencies
```python
%%capture
!pip install -r requirements.txt
```

#### Cell 5: Download dataset ICEWS14
```python
!bash /kaggle/working/my_code/download_dataset.sh
```

**Lưu ý**: Bước này sẽ download ~700MB dữ liệu và giải nén. Mất khoảng 1-2 phút.

#### Cell 6: Huấn luyện model
```python
!bash /kaggle/working/my_code/train_baseline.sh
```

**Lưu ý**: 
- Quá trình huấn luyện mất khoảng **7-8 giờ** với GPU
- Model sẽ được lưu trong thư mục `checkpoints/ICEWS14_BaselinePairRE/`
- Kết quả đánh giá sẽ hiển thị trong quá trình huấn luyện

#### Cell 7 (Optional): Nén kết quả để download
```python
import shutil
import os

# Nén toàn bộ kết quả
output_filename = "/kaggle/working/ket_qua_training"
dir_to_zip = "/kaggle/working/my_code"

try:
    shutil.make_archive(output_filename, 'zip', dir_to_zip)
    print(f"✓ Đã nén xong! File nằm tại: {output_filename}.zip")
    print(f"Kích thước: {os.path.getsize(output_filename + '.zip') / (1024*1024):.2f} MB")
except Exception as e:
    print(f"❌ Lỗi khi nén: {e}")
```

---

## Cách 2: Chạy trên máy local

### Bước 1: Cài đặt môi trường

#### 1.1. Cài đặt Python và pip
Đảm bảo bạn đã cài Python 3.7 trở lên:
```bash
python --version
```

#### 1.2. (Khuyến nghị) Tạo môi trường ảo
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

#### 1.3. Cài đặt PyTorch
Truy cập [pytorch.org](https://pytorch.org/get-started/locally/) để lấy lệnh cài đặt phù hợp với hệ thống của bạn.

**Ví dụ**:
```bash
# Windows + CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Windows + CPU only
pip install torch torchvision torchaudio

# Linux + CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 1.4. Cài đặt các thư viện khác
```bash
pip install tqdm numpy scikit-learn scipy
```

### Bước 2: Download và chuẩn bị code

```bash
# Giải nén code (nếu có file zip)
# hoặc clone/download từ repository

# Di chuyển vào thư mục project
cd baseline_icews14
```

### Bước 3: Download dataset

```bash
# Windows (Git Bash hoặc WSL)
bash download_dataset.sh

# Hoặc dùng Python
python download_data.py
```

**Lưu ý**: Script sẽ download và xử lý dataset ICEWS14 (~700MB), mất khoảng 5-10 phút.

### Bước 4: Huấn luyện model

```bash
# Cách 1: Dùng script có sẵn
bash train_baseline.sh

# Cách 2: Chạy trực tiếp với Python
python run.py --do_train --cuda --do_valid --do_test \
  --model BaselinePairRE \
  --data_path processed \
  -n 128 -b 256 -d 500 -g 12.0 \
  -a 1.0 -adv -dr -r 0.000001 -lr 0.0001 \
  --max_steps 100000 \
  --warm_up_steps 50000 \
  --valid_steps 5000 \
  --log_steps 100 \
  --test_batch_size 32 \
  --save_checkpoint_steps 5000 \
  --save_path checkpoints/ICEWS14_BaselinePairRE
```

**Lưu ý**: 
- Thêm `--cuda` nếu có GPU, bỏ đi nếu chỉ dùng CPU
- Với CPU, quá trình huấn luyện sẽ mất **rất lâu** (>10 giờ)
- Khuyến nghị giảm `--max_steps` xuống 10000 nếu chạy trên CPU để test

---

## Giải thích các tham số

### Tham số chính

| Tham số | Ý nghĩa | Giá trị mặc định |
|---------|---------|------------------|
| `--model` | Tên model | `BaselinePairRE` |
| `--data_path` | Đường dẫn đến dữ liệu đã xử lý | `processed` |
| `-d` / `--hidden_dim` | Số chiều của embedding | `500` |
| `-g` / `--gamma` | Margin trong loss function | `12.0` |
| `-b` / `--batch_size` | Kích thước batch | `256` |
| `-n` / `--negative_sample_size` | Số lượng negative samples | `128` |
| `-lr` / `--learning_rate` | Learning rate | `0.0001` |
| `--max_steps` | Số bước huấn luyện tối đa | `100000` |
| `--cuda` | Sử dụng GPU | - |
| `--do_train` | Thực hiện huấn luyện | - |
| `--do_valid` | Thực hiện validation | - |
| `--do_test` | Thực hiện test | - |

### Tham số nâng cao

| Tham số | Ý nghĩa | Giá trị mặc định |
|---------|---------|------------------|
| `-adv` | Sử dụng adversarial negative sampling | - |
| `-a` | Adversarial temperature | `1.0` |
| `-dr` | Sử dụng double relation embedding | - |
| `-r` / `--regularization` | Hệ số regularization | `0.000001` |
| `--warm_up_steps` | Số bước warm-up cho learning rate | `50000` |
| `--valid_steps` | Validate sau mỗi N bước | `5000` |
| `--log_steps` | Log sau mỗi N bước | `100` |
| `--save_checkpoint_steps` | Lưu checkpoint sau mỗi N bước | `5000` |

---

## Kiểm tra kết quả

### Trong quá trình huấn luyện

Model sẽ in ra thông tin mỗi `--log_steps` (mặc định 100 bước):
```
Training average positive_sample_loss: 0.0123 at step 100
Training average negative_sample_loss: 0.0456 at step 100  
Training average loss: 0.0579 at step 100
```

### Kết quả validation/test

Sau khi hoàn thành, kiểm tra file log:

```bash
# Windows (PowerShell)
Get-Content checkpoints\ICEWS14_BaselinePairRE\train.log | Select-String "Test MRR"

# Linux/Mac
cat checkpoints/ICEWS14_BaselinePairRE/train.log | grep "Test MRR"
```

### Metrics được đánh giá

- **MRR (Mean Reciprocal Rank)**: Trung bình nghịch đảo của rank
- **Hits@1**: Tỷ lệ câu trả lời đúng xuất hiện ở vị trí top-1
- **Hits@3**: Tỷ lệ câu trả lời đúng xuất hiện trong top-3
- **Hits@10**: Tỷ lệ câu trả lời đúng xuất hiện trong top-10

**Kết quả mong đợi** (tham khảo):
```
Test MRR: 0.42-0.45
Test Hits@1: 0.32-0.35
Test Hits@3: 0.47-0.50  
Test Hits@10: 0.60-0.63
```

---

## Cấu trúc thư mục sau khi chạy

```
baseline_icews14/
├── checkpoints/
│   └── ICEWS14_BaselinePairRE/    # Model checkpoints và logs
│       ├── train.log              # File log chi tiết
│       ├── checkpoint             # Model checkpoint cuối cùng
│       └── ...
├── processed/                      # Dữ liệu đã xử lý
│   ├── train.pkl
│   ├── valid.pkl
│   ├── test.pkl
│   └── mappings.pkl
├── src_data/                       # Dữ liệu gốc (sau khi download)
│   └── ICEWS14/
├── model.py                        # Định nghĩa model
├── run.py                          # Script chạy chính
├── dataloader.py                   # Data loader
├── train_baseline.sh               # Script huấn luyện
├── download_dataset.sh             # Script download data
└── README.md                       # File này
```

---

## Xử lý lỗi thường gặp

### 1. Lỗi "CUDA out of memory"
**Giải pháp**: Giảm batch size
```bash
python run.py ... -b 128  # Thay vì 256
```

### 2. Lỗi "wget not found" trên Windows
**Giải pháp**: 
- Sử dụng Git Bash hoặc WSL
- Hoặc download thủ công từ link trong file `download_dataset.sh`

### 3. Lỗi "No module named 'torch'"
**Giải pháp**: Cài đặt lại PyTorch
```bash
pip install torch torchvision torchaudio
```

### 4. Huấn luyện quá chậm
**Giải pháp**: 
- Đảm bảo đang sử dụng GPU (`--cuda`)
- Kiểm tra GPU có hoạt động: `torch.cuda.is_available()`
- Nếu không có GPU, giảm `--max_steps` để test

---

## Tham khảo

- **Paper gốc PairRE**: [PairRE: Knowledge Graph Embeddings via Paired Relation Vectors](https://arxiv.org/abs/2011.03798)
- **Dataset ICEWS14**: [Temporal Knowledge Base Completion](https://github.com/facebookresearch/tkbc)
- **PyTorch Documentation**: [pytorch.org](https://pytorch.org/docs/)

---

## Liên hệ và báo lỗi

Nếu gặp vấn đề khi chạy code, vui lòng:
1. Kiểm tra lại các bước trong README
2. Xem phần "Xử lý lỗi thường gặp"
3. Kiểm tra log file trong `checkpoints/ICEWS14_BaselinePairRE/train.log`

---

**Chúc bạn huấn luyện model thành công!**
