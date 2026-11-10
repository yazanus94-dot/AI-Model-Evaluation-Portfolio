# LLM-as-Judge Report - SIMULATED mock judge (not a real LLM)

- Pairs judged: **12** (each judged twice, answer order swapped)
- Order-consistency (same verdict in both orders): **50%**
- Contradictions (A in one order, B in the other): **8%**
- Agreement with humans on order-consistent pairs only: **100%**
- First-position preference: **46%** of raw verdicts chose the answer shown first (an unbiased judge should be near the share of pairs where the first answer is truly better)

| Method | Agreement with humans | Cohen's kappa |
|---|---|---|
| Single call (no swap) | 83% | 0.72 |
| Swap + reconcile | 83% | 0.71 |

## Order-sensitive pairs

- `bio-atp` [soft]: forward=A, swapped=tie, final=A, human=A
- `py-reverse` [soft]: forward=tie, swapped=A, final=A, human=A
- `bio-t1d` [soft]: forward=B, swapped=tie, final=B, human=B
- `geo-canada` [soft]: forward=tie, swapped=B, final=B, human=tie
- `med-antibiotics` [soft]: forward=A, swapped=tie, final=A, human=A
- `bio-photo` [CONTRADICTION]: forward=A, swapped=B, final=tie, human=B
