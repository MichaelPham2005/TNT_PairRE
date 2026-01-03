# Complex-PairRE-TNT
**Temporal Knowledge Graph Embedding with Complex Rotation, Pairwise Relations, and TNT Decomposition**

---

## 1. Overview

This model, referred to as **Complex-PairRE (RotatE + PairRE variant)**, is designed as a **temporal knowledge graph embedding (TKGE)** architecture that integrates three complementary ideas:

1. **Complex-space rotation** from RotatE / ComplEx  
2. **Paired relation structure** from PairRE (separate head/tail transformations)  
3. **Temporal / Non-temporal decomposition** from TNTComplEx  

The objective is to build a **time-aware backbone** that is strictly more expressive than TNTComplEx while remaining compatible with temporal link prediction tasks.

---

## 2. Complex-valued Representation

All embeddings live in a complex vector space:

C^d

### 2.1 Entity embeddings
Each entity x is represented as:
e_x ∈ C^d, with each component e_{x,j} = a_j + b_j i

---

### 2.2 Relation embeddings (paired rotations)

Each relation r is represented by **two complex rotation vectors**:

- Head-side rotation: r^H  
- Tail-side rotation: r^T  

Each component is parameterized by an angle:
r^H_{r,j} = exp(i θ^H_{r,j}),  
r^T_{r,j} = exp(i θ^T_{r,j}),  θ ∈ [0, 2π)

This allows **asymmetric and N-to-N relations** to be modeled naturally.

---

## 3. Temporal Modulation (TNT-style)

### 3.1 Time embedding
Each timestamp τ has an embedding:
w_τ ∈ R^d

---

### 3.2 Temporal / Non-temporal angle decomposition

For each relation r, head and tail rotations are decomposed into:

- Static angle: θ^H_{r,static}, θ^T_{r,static}  
- Temporal angle: θ^H_{r,temp}, θ^T_{r,temp}  

The **time-conditioned rotation** is defined as:

θ^H_{r,τ} = θ^H_{r,static} + θ^H_{r,temp} · w_τ  
θ^T_{r,τ} = θ^T_{r,static} + θ^T_{r,temp} · w_τ  

Resulting complex rotations:

r^H_{r,τ} = exp(i θ^H_{r,τ}),  
r^T_{r,τ} = exp(i θ^T_{r,τ})

---

## 4. Scoring Function (Complex-PairRE)

Given a quadruple (h, r, t, τ), the score is defined as:

f_r(h,t,τ) =
- || e_h ∘ r^H_{r,τ} − e_t ∘ r^T_{r,τ} ||_1

Where:
- ∘ is the Hadamard product in complex space (rotation)
- ||·||_1 is applied to real and imaginary parts

---

## 5. Training Objective

### 5.1 Self-adversarial negative sampling loss

The model is optimized using a self-adversarial loss:

L_adv =
- log σ(γ − f(h,r,t,τ))
- Σ_i p_i log σ(f(h_i',r,t_i',τ) − γ)

with:

p_i = exp(f_i) / Σ_j exp(f_j)

---

### 5.2 Temporal consistency loss

Following TNTComplEx, an auxiliary temporal objective is added:

L_time = α · ℓ(X̂ ; (s,p,o,τ))

This improves performance on queries of the form (s, p, o, ?).

---

### 5.3 Total loss

L_total = L_adv + L_time

---

## 6. Regularization

### 6.1 Temporal smoothness

Λ_p(w) = (1 / (|T|−1)) Σ || w_{i+1} − w_i ||_p^p

---

### 6.2 Nuclear 3-norm regularization

A nuclear 3-norm is applied on unfolded relation–time tensors as proposed in TNTComplEx to improve generalization.

---

## 7. Advantages

- Paired rotations model asymmetric and N-to-N relations
- Complex space captures composition, inversion, and symmetry
- Temporal angle modulation yields smooth time evolution
- Strictly more expressive than TNTComplEx

---

## 8. Summary

Complex-PairRE unifies:
- RotatE-style complex rotations
- PairRE-style paired relations
- TNT-style temporal factorization

into a single temporal knowledge graph embedding framework.
