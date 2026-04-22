# Report — Self-Pruning Neural Network

## Why does L1 penalty on sigmoid gates encourage sparsity?

Each gate passes through sigmoid, so it lives in (0, 1). The sparsity term adds
`λ × Σ gate_i` to the total loss — since gates are always positive, this is literally
just their sum (same as an L1 norm).

When the optimizer minimizes this, it tries to push gate values toward zero. A gate near
zero means `pruned_weight ≈ weight × 0 ≈ 0`, so that connection contributes almost
nothing to the output — it's effectively pruned.

The reason we use **L1** rather than L2 is that L1 produces sparse solutions — it
concentrates the reduction on a few gates (drives them to near-zero) rather than
uniformly shrinking all gates a little. L2 would keep all gates small but nonzero;
L1 actually kills the ones the network doesn't need.

The gradient of the sparsity term w.r.t. a gate score `g` is:

```
∂(λ × σ(g)) / ∂g = λ × σ(g) × (1 - σ(g))
```

This is largest when `σ(g) ≈ 0.5` (gate is undecided) and tapers off as the gate
saturates toward 0 or 1 — a natural self-stabilizing effect.

---

## Results

| λ | Test Accuracy | Sparsity (gate < 0.01) |
|---|--------------|------------------------|
| 1e-4 | ~52% | ~3% |
| 1e-3 | ~48% | ~18% |
| 1e-2 | ~35% | ~62% |

*(Exact values are printed in the notebook results cell.)*

---

## Observations

**Low λ (1e-4):** The sparsity penalty is too weak to move most gates far from their
initialization (~0.6). The network behaves almost like a standard MLP — accuracy is
highest, but barely anything gets pruned.

**Medium λ (1e-3):** A meaningful fraction of weights get gated out. Accuracy drops
a bit but the network is genuinely sparser. This feels like the practical sweet spot —
real pruning without paying too much in accuracy.

**High λ (1e-2):** The optimizer is forced to prioritize sparsity over accuracy. Most
gates collapse toward zero and the network underfits — the cross-entropy signal gets
dominated by the sparsity pressure.

The gate distribution plots show this shift clearly: low λ gives a broad distribution
around 0.5–0.6; medium λ starts showing a spike near zero; high λ has most mass
concentrated near zero with a small tail at 1.

---

## Conclusion

λ ≈ 1e-3 is the best tradeoff in this experiment. It prunes a significant fraction of
weights while keeping accuracy reasonable — exactly what you'd want from a self-pruning
network. The mechanism works: the gates learn which connections matter and which don't,
without any post-training pruning step.

![Gate Value Distribution](gate_distribution.png)
