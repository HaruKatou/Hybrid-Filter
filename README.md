### 1. Cài đặt UV

**Cài đặt UV trên Windows:**
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Cài đặt UV trên Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Khởi tạo môi trường
```bash
# Clone repository
git clone <repository-url>
cd base-research-repo

# Tạo và kích hoạt virtual environment
uv venv
.venv\Scripts\activate

# Đồng bộ phiên bản môi trường
uv sync
```

###  3. Cài đặt dependencies
```bash
uv add
# Ví dụ
uv pip install torch torchvision transformers datasets
```

### 4. Quản lý nhánh git
```bash
git checkout -b branch_name

# Sau khi hoàn thành, tạo pull request từ feature branch vào master
```

### 5. Chạy đánh giá
```bash
python evaluate.py

```

### 6. Vẽ biểu đồ (đã vẽ sẵn ở ảnh figure_2_replication.png)
```bash
python plot.py


```