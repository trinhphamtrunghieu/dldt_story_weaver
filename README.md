# Tái hiện StoryWeaver trên Kaggle/Colab

Repository này chứa notebook tái hiện StoryWeaver (AAAI 2025) và hướng dẫn chạy lại. Kết quả thực nghiệm và phần mô tả giải pháp nằm trong báo cáo, không đặt ở đây.

## Nội dung

| Tệp | Vai trò |
|---|---|
| `storyweaver.ipynb` | Notebook duy nhất: huấn luyện, sinh ảnh, đánh giá, ghi nhận lên Weights & Biases |
| `README.md` | Tài liệu này |

## Môi trường

**Phần cứng đã dùng:** 1 GPU NVIDIA T4 16 GB (Kaggle). Thư mục làm việc của Kaggle giới hạn 20 GB; mỗi checkpoint chiếm khoảng 3.4 GB (U-Net và knowledge encoder).

**Phần mềm:** Python và PyTorch có sẵn trên image Kaggle. Notebook không cài lại PyTorch, chỉ ghim các thư viện sau (ô cài đặt tự chạy):

```bash
pip install -q "diffusers==0.31.0" "transformers==4.47.0" "tokenizers==0.21.0" \
    "accelerate==1.0.1" "huggingface_hub==0.26.2" "einops==0.8.0" bitsandbytes
pip install -q importlib_metadata regex filelock packaging pyyaml requests tqdm
pip install -q wandb
```

`diffusers==0.31.0` là bắt buộc vì mã vá trực tiếp vào `Transformer2DModel` và `AttnProcessor2_0` của đúng phiên bản này.

**Mạng:** cần bật Internet trong Kaggle để tải Stable Diffusion v1.5 (khoảng 5 GB), CLIP ViT-B/32 và DINOv2 từ HuggingFace. Repository `runwayml/stable-diffusion-v1-5` đã bị gỡ; notebook thử lần lượt các bản mirror khai báo trong `CFG.MODEL_FALLBACKS`.

## Dữ liệu

Cần bộ dữ liệu **TBC-Bench** do tác giả StoryWeaver công bố, với cấu trúc:

```
TBC_Bench/
├── train-folder/<Pororo_the_Little_Penguin | Frozen>/{image, caption, character, style.txt}
└── eval-plot/{eval_train/each/<phim>, eval_new_story/<phim>}
```

- **Kaggle:** tạo dataset từ TBC-Bench rồi *Add Input*. Notebook tự dò thư mục chứa `train-folder` dưới `/kaggle/input`.
- **Colab:** đặt bộ dữ liệu tại `MyDrive/DLDT/TBC_Bench`; ô đầu tiên gắn Drive và sao chép về `/root/TBC_Bench`.

## Chạy lại

Chạy các ô **từ trên xuống dưới**, không có script dòng lệnh riêng (ô `CFG` thay cho `train.sh` và `sample.sh` của mã gốc).

| Phần | Việc làm | Ghi chú |
|---|---|---|
| 0 | Kiểm tra GPU, RAM, ổ đĩa | Cần `CUDA available = True` |
| 1 | Cài thư viện | Nếu báo cần khởi động lại, cứ chạy tiếp |
| 2 | Ô `CFG` | Chỗ duy nhất cần sửa (bảng bên dưới) |
| 3-5 | Định nghĩa patch cross-attention, knowledge encoder, lớp dữ liệu | Không tốn GPU |
| 6 | Tải SD v1.5 và quét dữ liệu | Chỉ huấn luyện khi báo cáo quét dữ liệu kết luận "DỮ LIỆU OK" |
| 7 | Huấn luyện | Tự lưu checkpoint theo `CKPT_STEPS` |
| 8-9 | Sinh truyện mới, đóng gói kết quả | Xuất `storyweaver_results.zip` |
| 10 | Đánh giá: dựng đề thi, sinh ảnh, tính chỉ số | Đặt `EVAL.CKPT_DIR` trỏ tới thư mục `pipeline` |
| cuối | Tải lên W&B | Xem mục bên dưới |

**Lượt chạy đầu nên bật `CFG.SMOKE_TEST = True`** (10 bước huấn luyện, 10 bước khử nhiễu, khoảng 10 phút cho cả notebook) để bắt lỗi đường dẫn hoặc thiếu bộ nhớ, rồi tắt và chạy lại từ phần 7.

### Tham số chính trong `CFG`

| Tham số | Giá trị đã dùng | Ý nghĩa |
|---|---|---|
| `STORY_WORLD` | `Pororo_the_Little_Penguin` | Hoặc `Frozen` |
| `RESOLUTION` | 512 | Ảnh 512x896; phải là bội của 256 |
| `MAX_CHAR` | 2 | Số nhân vật tối đa mỗi khung |
| `TRAIN_BATCH` / `GRAD_ACCUM` | 1 / 4 | Batch hiệu dụng 4 |
| `MAX_TRAIN_STEPS` | 3000 | Số bước tối ưu |
| `LR` | 7e-6 | Hằng số, không khởi động |
| `MIXED_PRECISION` | `fp16` | U-Net fp16; VAE và text encoder fp32 |
| `USE_8BIT_ADAM` | `True` | Cần `bitsandbytes`; giảm khoảng 5 GB VRAM |
| `SCALE_TRAIN` / `SCALE` | 1.0 / 2.5 | Hệ số alpha khi huấn luyện / khi sinh ảnh |
| `GUIDANCE_SCALE` | 5.0 | Classifier-free guidance |
| `NUM_INFER_STEPS` | 100 | Bước khử nhiễu khi sinh truyện mới |
| `SEED` | 42 | Seed huấn luyện; seed đánh giá bốc từ seed này |
| `TIME_BUDGET_MIN` | 540 | Tự lưu và dừng trước giới hạn phiên Kaggle |

Cấu hình đánh giá nằm ở ô `EVAL`: `N_STORIES_SINGLE`, `N_STORIES_MULTI`, `MAX_FRAMES`, `N_SEEDS`, `NUM_INFER_STEPS`, `DINO_THRESH`. Ảnh đánh giá được lưu đệm theo đường dẫn; với `OVERWRITE = False`, chạy lại chỉ sinh phần còn thiếu.

### Train tiếp sau khi hết phiên

Chọn *Save Version → Save Output*, ở phiên mới thêm output đó làm Input, rồi đặt `CFG.RESUME_PIPELINE_DIR` trỏ tới thư mục `pipeline` (chứa `unet/` và `k_encoder.pt`). Chỉ trọng số được khôi phục; trạng thái bộ tối ưu không được lưu khi `SAVE_FULL_STATE = False`.

## Ghi nhận bằng Weights & Biases

1. Trong Kaggle: *Add-ons → Secrets*, thêm secret tên `WANDB_API_KEY`. Ở môi trường khác, ô đăng nhập sẽ hỏi khóa.
2. Cell đầu notebook đặt `WANDB_DISABLED`; ô tải lên ở cuối notebook tự gỡ biến này và đăng nhập.
3. Chạy ô tải lên **sau** khi đã tính xong chỉ số. Ô ghi vào project `StoryWeaver` ba run: `main-results`, `ablation`, `metric-analysis`; đặt `WANDB_ENTITY` thành tài khoản của bạn.
4. Muốn chia sẻ công khai: mở project trên wandb.ai, *Project settings → Privacy → Public*, hoặc dùng liên kết chia sẻ của báo cáo (Report → Share).

## Tệp đầu ra

| Đường dẫn | Nội dung |
|---|---|
| `/kaggle/working/output/<phim>/pipeline/` | `unet/`, `k_encoder.pt`, `sw_meta.json` |
| `/kaggle/working/output/<phim>/loss_history.json` | Mất mát theo bước |
| `/kaggle/working/output/<phim>/sample/` | Ảnh kiểm tra trong lúc huấn luyện |
| `/kaggle/working/compare/images/` | Ảnh đánh giá theo cấu hình, nhiệm vụ, truyện, seed |
| `/kaggle/working/compare/` | `per_image_metrics.csv`, `summary.json`, `meta.json`, `compare_*.csv` |
