# TNT-PairRE: Temporal PairRE with TNT-style Decomposition

## 1. Mục tiêu
TNT-PairRE là một mô hình **Temporal Knowledge Graph Embedding (TKGE)** được thiết kế bằng cách:
- Giữ **PairRE** làm backbone (paired relation vectors cho head/tail)
- Áp dụng tư tưởng **TNT (Temporal / Non-temporal decomposition)** để mô hình hóa sự thay đổi theo thời gian
- Huấn luyện bằng **multiclass (softmax) loss** (Option A), tương thích trực tiếp với pipeline TKBC / TNTComplEx

Mục tiêu của README này là cung cấp **đặc tả toán học + hướng dẫn triển khai chi tiết** để một AI khác (hoặc bạn) có thể **code chính xác mô hình** mà không cần tham chiếu thêm paper.

---

## 2. Ký hiệu & dữ liệu đầu vào

Mỗi fact là một **quadruple**:
\[
(h, r, t, \ell)
\]

Trong đó:
- \(h, t \in \mathcal{E}\): head / tail entity
- \(r \in \mathcal{R}\): relation
- \(\ell \in \{0,1,\dots,|T|-1\}\): chỉ số timestamp (đã discretize)

---

## 3. Tham số mô hình

### 3.1 Entity embedding
- Mỗi entity \(x\):
\[
\mathbf{e}_x \in \mathbb{R}^d
\]
- **Ràng buộc bắt buộc**:
\[
\|\mathbf{e}_x\|_2 = 1
\]
(áp dụng sau mỗi optimizer step)

---

### 3.2 Time embedding
- Mỗi timestamp \(\ell\):
\[
\boldsymbol{\tau}_\ell \in \mathbb{R}^d
\]

---

### 3.3 Relation decomposition (TNT-style)

Mỗi relation \(r\) được tách thành **4 vector**:

**Non-temporal (static):**
- \(\mathbf{r}^H \in \mathbb{R}^d\)
- \(\mathbf{r}^T \in \mathbb{R}^d\)

**Temporal (time-sensitive):**
- \(\mathbf{r}^{H,t} \in \mathbb{R}^d\)
- \(\mathbf{r}^{T,t} \in \mathbb{R}^d\)

---

## 4. Time-conditioned relation (core của TNT-PairRE)

Tại timestamp \(\ell\), relation được điều biến bởi time embedding:

\[
\mathbf{r}^H_\ell = \mathbf{r}^H + \mathbf{r}^{H,t} \odot \boldsymbol{\tau}_\ell
\]

\[
\mathbf{r}^T_\ell = \mathbf{r}^T + \mathbf{r}^{T,t} \odot \boldsymbol{\tau}_\ell
\]

Trong đó \(\odot\) là Hadamard (element-wise) product.

---

## 5. Score function (PairRE + temporal)

Với quadruple \((h,r,t,\ell)\):

\[
f(h,r,t,\ell)
=
- \left\|
\mathbf{e}_h \odot \mathbf{r}^H_\ell
-
\mathbf{e}_t \odot \mathbf{r}^T_\ell
\right\|_1
\]

- Khoảng cách **L1** được giữ nguyên theo PairRE gốc
- Score càng lớn (ít âm) → triple càng đúng

---

## 6. Loss function (Option A – Multiclass / Softmax)

Huấn luyện theo dạng **tail prediction** giống TNTComplEx.

Với mỗi \((h,r,t,\ell)\):

\[
\mathcal{L}_{main}
=
- f(h,r,t,\ell)
+ \log \sum_{t' \in \mathcal{E}}
\exp(f(h,r,t',\ell))
\]

### Ghi chú triển khai
- Có thể dùng **1-vs-All softmax** nếu số entity vừa phải (ICEWS14)
- Hoặc **sampled softmax** để tiết kiệm bộ nhớ
- Khuyến nghị train thêm **reciprocal relations** để cân bằng head/tail ranking

---

## 7. Regularization & constraint (bắt buộc)

### 7.1 Entity unit-norm
Sau mỗi optimizer step:
\[
\mathbf{e}_x \leftarrow \frac{\mathbf{e}_x}{\|\mathbf{e}_x\|_2}
\]

---

### 7.2 Temporal smoothness (TNT regularizer)

Khuyến khích embedding thời gian biến thiên mượt:

\[
\mathcal{L}_{time}
=
\lambda_{time}
\cdot
\frac{1}{|T|-1}
\sum_{\ell=0}^{|T|-2}
\|
\boldsymbol{\tau}_{\ell+1}
-
\boldsymbol{\tau}_{\ell}
\|_2^2
\]

---

### 7.3 L2 regularization cho relation & time

\[
\mathcal{L}_{reg}
=
\lambda_r
(
\|\mathbf{r}^H\|_2^2
+
\|\mathbf{r}^T\|_2^2
+
\|\mathbf{r}^{H,t}\|_2^2
+
\|\mathbf{r}^{T,t}\|_2^2
)
+
\lambda_\tau
\|\boldsymbol{\tau}\|_2^2
\]

---

## 8. Tổng loss

\[
\mathcal{L}_{total}
=
\mathcal{L}_{main}
+
\mathcal{L}_{time}
+
\mathcal{L}_{reg}
\]

---

## 9. Pseudocode (forward)

```python
# inputs
eh = entity_emb[h]        # (d,)
et = entity_emb[t]        # (d,)
tau = time_emb[l]         # (d,)

rH, rT = rel_H[r], rel_T[r]
rH_t, rT_t = rel_H_t[r], rel_T_t[r]

# time-conditioned relation
rH_l = rH + rH_t * tau
rT_l = rT + rT_t * tau

# score
score = -torch.sum(torch.abs(eh * rH_l - et * rT_l))
```

---

## 10. Hyperparameter gợi ý (ICEWS-style)

| Tham số | Giá trị khởi đầu |
|------|----------------|
| Embedding dim | 100 hoặc 200 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| \(\lambda_{time}\) | 1e-4 → 1e-2 (grid) |
| \(\lambda_r\) | 1e-5 |
| \(\lambda_\tau\) | 1e-5 |
| Epochs | 100 – 300 |

---

## 11. Ablation bắt buộc nên chạy

1. **PairRE (static)**: bỏ hoàn toàn time
2. **T-PairRE**: chỉ dùng \(r^{*,t}\)
3. **TNT-PairRE (full)**: mô hình này
4. TNT-PairRE + temporal gate (tuỳ chọn)

So sánh: **MRR / Hits@1/3/10**

---

## 12. Expected behavior & lưu ý

- Nếu \(\lambda_{time}\) quá lớn → temporal collapse (time embeddings gần như hằng)
- Nếu quá nhỏ → temporal overfit
- Entity unit-norm là **bắt buộc**, nếu không PairRE score sẽ mất ổn định
- Khi MRR hội tụ sớm → thử giảm learning rate hoặc tăng dim

---

## 13. Kết luận

TNT-PairRE kết hợp:
- Khả năng biểu diễn quan hệ phức tạp của PairRE
- Cơ chế tách temporal / non-temporal mạnh của TNT

Mô hình này **drop-in replaceable** cho TNTComplEx trong pipeline TKBC, chỉ cần thay score function và embedding structure.

