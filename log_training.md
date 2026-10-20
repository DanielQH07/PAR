# Training log — lmq (UPAR 2027 Track 1)

Máy: 2× RTX 5080 16 GB, 24 core, 62 GB RAM. Ảnh trên HDD (`images -> /mnt/hdd/shared/quanlm46/UPAR/data`),
phải warm page cache trước khi train (xem Sự cố). Mọi điểm dưới đây là **LODO held-out, contextual calibration
λ 0.1, ngưỡng 0.5, metric Codabench** (`src/prepare.py`), 1 seed (605).

Chạy: `bash research/lodo_2gpu.sh TAG [train.py args]` (3 fold song song; `GPU_MAP`, `FOLDS`, `KEEP_PTH`, `WORKERS`).
Chấm decode: `PYTHONPATH=src .venv/bin/python research/group_decode_eval.py runs/lodo/TAG`.

Recipe chung (trừ khi ghi khác): `--weights weights_own/metaclip_b16 --unfreeze 12 --llrd 0.8 --ft-samples 180000`
(256x128, EMA 0.999, pos_weight^0.5, label smoothing 0.1, domain balance, mask missing — default của train.py).

## Kết quả LODO (Challenge_Avg)

**Mốc so sánh = bản mới nhất của `origin/ttt` (fa64397, chưa có commit mới — fetch 2026-10-05):** recipe ở README
(MetaCLIP B/16, `--unfreeze 12 --llrd 0.8 --ft-samples 180000`) + logit distillation từ teacher MetaCLIP + PE-Core,
nộp dạng soup 3 seed. Trên máy này tái tạo bằng **`kd2`** (1 seed 605, không soup; teacher = `base_mc` + `pe_t`
train tại chỗ). Log/điểm LODO gốc của team nằm trên server, không có ở đây. `base_mc` = cùng recipe nhưng không distill.

_Sắp theo Mean giảm dần; run chưa đủ 3 fold ở cuối. Chỉ giữ cấu hình chạy được trong 3,600 s CPU._

| Tag | Thay đổi | Market1501 | PA100k | PETA | Mean | Ghi chú |
|---|---|---:|---:|---:|---:|---|
| **`kd2` (≈ ttt mới nhất)** | `--teacher runs/lodo/base_mc/{held} runs/lodo/pe_t/{held}` (kd 0.25) | 0.8546 | 0.7956 | 0.8060 | **0.8187** | ≈ recipe `ttt` (1 seed, không soup); +0.0017 vs base,  PETA chạy lại dưới tag `kd2b` |
| `pe_t` | `--arch pe --weights weights_own/pe_core_b16` | 0.8553 | 0.7959 | 0.8047 | **0.8186** | teacher #2 (KEEP_PTH); +0.0016 vs base (1 seed, ~noise) |
| `kd2_s606` | `kd2` với `--seed 606` | 0.8531 | 0.7950 | 0.8053 | **0.8178** | **nhiễu seed**: kd2 605 vs 606 lệch 0.0009 mean, 0.0015 ở Market → chênh < ~0.0015 coi như nhiễu |
| `kd2_w1` | như `kd2` + `--kd-weight 1.0` | 0.8532 | 0.7961 | 0.8048 | **0.8180** | so với kd2 (0.8187): −0.0007 → kd weight 1.0 không giúp |
| `base_mc` | baseline MetaCLIP B/16, không distill | 0.8546 | 0.7915 | 0.8049 | **0.8170** | 2026-10-05; ckpt giữ lại làm teacher |
| `s224` | `--size 224 112` (98 token, ~25% nhanh hơn trên CPU) | 0.8513 | 0.7882 | 0.8039 | **0.8145** | −0.0025 vs base |
| `mc2` | `--arch mc2 --weights weights_own/metaclip2_b16` (MetaCLIP 2 worldwide B/16) | 0.8478 | 0.7922 | 0.7930 | **0.8110** | −0.0060 vs base_mc, tụt ở Market (−0.0068) và PETA (−0.0119) → loại |
| `q2l1` | `--extra-head q2l1` (transformer decoder 1 lớp, Query2Label) | 0.8549 | _queue_ | _queue_ | | Market +0.0003 vs base (trong nhiễu) |
| `q2l1_lr2e4` | `--extra-head q2l1 --extra-lr 2e-4` | 0.8478 | — | — | | Market −0.0068 → loại |
| `q2l2_lr2e4` | `--extra-head q2l2 --extra-lr 2e-4` (decoder 2 lớp) | 0.8487 | — | — | | Market −0.0059 → loại |
| `gap` | `--extra-head gap` (gated attention pooling / attention-MIL, mới) | 0.8493 | — | — | | Market −0.0053 → loại |
| `query` | `--extra-head query` (gated cross-attn ML-Decoder, có sẵn) | 0.8489 | — | — | | Market −0.0057 → loại (khớp kết luận cũ của team) |
| `kd_q2l1` | `kd2` + `--extra-head q2l1` | _queue_ | _queue_ | _queue_ | | distill + Q2L |

Chi tiết (mA / Inst_F1 / Prec / Rec):

| Tag/fold | mA | F1 | Prec | Rec | train s |
|---|---:|---:|---:|---:|---:|
| base_mc/Market1501 | 0.8465 | 0.8628 | 0.8508 | 0.8751 | 587 |
| base_mc/PA100k | 0.8162 | 0.7681 | 0.6945 | 0.8593 | 377 |
| base_mc/PETA | 0.8060 | 0.8037 | 0.7891 | 0.8189 | 611 |
| pe_t/Market1501 | 0.8465 | 0.8642 | 0.8523 | 0.8765 | 796 |
| pe_t/PA100k | 0.8130 | 0.7795 | 0.7121 | 0.8610 | 838 |
| pe_t/PETA | 0.8034 | 0.8060 | 0.7891 | 0.8235 | 421 |
| s224/Market1501 | 0.8405 | 0.8625 | 0.8533 | 0.8719 | 850 |
| s224/PA100k | 0.8121 | 0.7657 | 0.6910 | 0.8584 | 581 |

(train s đo khi nhiều run chia GPU, chỉ để tham khảo.)

## Hướng #1 — decode theo nhóm loại trừ (post-hoc, trên probe `base_mc`)

Nhãn: Age, UpperBody-Color, LowerBody-Color, LowerBody-Type gần như luôn đúng 1 nhãn dương/ảnh (≥ 88.6–100% tuỳ domain).

| Biến thể | Market1501 | PA100k | PETA | Mean Δ | mA | Prec | Rec |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline cc | 0.8546 | 0.7915 | 0.8049 | 0 | 0.8229 | 0.7781 | 0.8511 |
| exactly-one core4 | 0.8442 | 0.8051 | 0.7857 | −0.0053 | 0.7899 | 0.8390 | 0.8306 |
| at-most core4 | 0.8440 | 0.8050 | 0.7858 | −0.0054 | 0.7895 | 0.8408 | 0.8294 |
| at-least core4 | 0.8548 | 0.7916 | 0.8047 | +0.0001 | 0.8234 | 0.7766 | 0.8522 |
| at-most core4, margin 2 | 0.8532 | 0.8067 | 0.8050 | +0.0046 | 0.8201 | 0.7995 | 0.8497 |
| at-most LB-color only, margin 2.5 | +0.0003 | +0.0114 | −0.0000 | +0.0039 | | | |
| at-most UB-color only (mọi margin) | ≤0 | ≤0 | ≤0 | ≤0 | | | |

**Kết luận:** ép cứng đổi mA lấy precision (−0.033 mA, +0.061 Prec) → harmonic mean giảm. Bản margin chỉ có lợi nhờ
LB-color trên fold PA100k (student train Market+PETA bật 2 màu quần trên ảnh PA100k); fold Market1501 (sát MEVID
nhất) ≈ 0. Model nộp thật train cả PA100k nên lỗi này có thể biến mất → **không đưa vào submission**. Kiểm lại trên
probe `kd2`: at-most core4 m=2 → Market +0.0004, PA100k +0.0165, PETA −0.0004 (mean +0.0055) — cùng pattern, gain chỉ ở PA100k. s224: +0.0046 (PA100k), pe_t: +0.0035 (Market −0.0049).

## Sự cố & lưu ý vận hành

- **HDD là nút cổ chai**: lần chạy đầu GPU 0% util, DataLoader worker ở trạng thái `D`. Fix: warm page cache
  `find -L images/ -type f -print0 | xargs -0 -n 500 -P 16 cat > /dev/null` (~6 phút, 1.7 GB). Sau đó GPU 98–100%.
  Cache mất khi reboot hoặc RAM bị chiếm.
- **OOM 16:04**: lần launch `pe_t` đầu tiên env không qua được tmux (`KEEP_PTH`/`GPU_MAP` mất) → 2 fold PE dồn
  vào GPU0 cùng `s224` → fold PETA của `pe_t` và `s224` OOM. Luôn đặt env **bên trong** lệnh tmux.
- **Nhiệt CPU**: 6 run × 7 worker → Package 101 °C (crit 105). Đã pause `s224` (SIGSTOP/SIGCONT), hạ `WORKERS` mặc
  định xuống 4. Với 4 run × 4 worker vẫn ~90 °C. Watcher: tmux `temp` (`research/temp_watch.sh`, mỗi 2 phút, log
  `logs/temp_alert.log` khi > 80 °C).
- Tải PE-Core cần `HF_HOME=/mnt/hdd/shared/quanlm46/UPAR/hf_cache` (`~/.cache/huggingface` thuộc root).
- **OOM**: 3 fold trên cùng 1 GPU 16 GB dễ OOM (bộ nhớ tăng lúc eval held-out), kể cả không distill → **tối đa 2 fold/GPU**.
- **Nhiệt**: máy chịu được ~3–4 fold cùng lúc; 6 fold → CPU 97–101 °C.
- Ensemble 2 model (base_mc+pe_t, LODO 0.8224) **đã loại**: 2× ViT-B ≈ 3,800–6,400 s > 3,600 s trên scorer. Chỉ dùng làm teacher cho distill.
- `pgrep -f`/`pkill -f` với pattern có trong chính lệnh shell sẽ trúng luôn shell đó — dùng PID cụ thể.

## Hướng sẽ làm — phát triển classifier head

Hiện tại (`--head linear`, mặc định): **LayerNorm → Linear(768→40)** trên CLS token đã pool của ViT
(`src/clip_vit.py:AttributeHead`). Backbone ViT-B/16 *đã là* transformer; head chỉ là 1 lớp tuyến tính.

Đã có sẵn trong `train.py` và **team đã thử trên LODO, không có gain** (README "attribute and query heads"):

| Flag | Kiểu | Mô tả |
|---|---|---|
| `--head text` | cosine với text embedding | 40 vector khởi tạo từ prompt CLIP |
| `--extra-head query` / `query-nogate` / `query1` | **cross-attention + gated attention** (ML-Decoder) | 40 query attend vào patch token, gate sigmoid theo query; zero-init cộng vào logit CLS |
| `--extra-head region` | spatial prior attention | mỗi thuộc tính một bản đồ chú ý theo vùng cơ thể |
| `--attr-tokens k` | attribute token trong k block cuối | 40 token học được chèn vào transformer |

Ý tưởng chưa thử (cần đo LODO + thời gian CPU, head nhỏ nên chi phí inference ~0 trừ khi ghi khác):

- [ ] **MLP head** 2 lớp (768→1024→40, GELU, dropout) — baseline phi tuyến rẻ nhất.
- [ ] **Head theo nhóm thuộc tính** (Age / Hair / UB-color / LB-color / LB-type / Accessory): 1 head nhỏ mỗi nhóm,
      nhóm loại trừ dùng softmax thay sigmoid — học cấu trúc "đúng 1 nhãn" trong lúc train thay vì decode post-hoc
      (#1 đã cho thấy decode post-hoc không giúp Market fold). Gần với `--group-ce` đã thử → ưu tiên thấp.
- [ ] **Transformer decoder nhẹ** (1–2 layer, self-attn giữa 40 attribute query + cross-attn patch token, như
      Query2Label) — khác `query` hiện có ở chỗ có self-attention giữa các thuộc tính (học tương quan thuộc tính).
      Chi phí CPU: ~1 block ViT nhỏ, cần đo.
- [ ] **Gated attention pooling** nhiều lớp: fuse CLS của nhiều block (đã có `--fuse`, đã thử) + gate theo thuộc tính.
- [ ] **CNN/conv head trên patch grid** (16×8 token → depthwise conv → pool theo vùng) — thêm inductive bias cục bộ
      cho phụ kiện nhỏ (Glasses, Hat, Bag — nhóm lỗi lớn theo error anatomy).
- [ ] **RNN/GRU over attributes** (dự đoán tuần tự có điều kiện, kiểu CNN-RNN PAR) — mô hình hoá phụ thuộc nhãn;
      ít được ưa chuộng gần đây, đặt sau transformer decoder.
- [ ] **Graph head (GCN/ML-GCN)** trên đồ thị đồng xuất hiện thuộc tính — một biến thể khác để học tương quan nhãn.

Lưu ý: điểm nghẽn đã chẩn đoán là **mâu thuẫn quy ước nhãn giữa dataset (màu) và phụ kiện nhỏ**, không phải sức chứa
head; các head attention đã thử đều đổi mA lấy F1. Head mới chỉ đáng giữ nếu tăng cả 3 fold (đặc biệt Market1501).
