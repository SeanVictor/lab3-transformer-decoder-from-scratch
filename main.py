"""
main.py  —  Laboratório 3: Implementando o Decoder
Disciplina: Tópicos em Inteligência Artificial – 2026.1
Instituição: iCEV

Pipeline completo:
    Tarefa 1 → Máscara Causal (Look-Ahead Mask)
    Tarefa 2 → Cross-Attention (Ponte Encoder-Decoder)
    Tarefa 3 → Loop de Inferência Auto-Regressivo
"""

import numpy as np

from task1_causal_mask        import create_causal_mask, softmax
from task2_cross_attention    import CrossAttention
from task3_autoregressive_loop import generate_next_token, autoregressive_loop, vocab

np.random.seed(42)

D_MODEL    = 64
BATCH_SIZE = 1


# ════════════════════════════════════════════════════════════
# TAREFA 1 — Máscara Causal
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 55)
print("  TAREFA 1 — Máscara Causal (Look-Ahead Mask)")
print("═" * 55)

SEQ_LEN = 5
M = create_causal_mask(SEQ_LEN)

print(f"\nMáscara M  [{SEQ_LEN}×{SEQ_LEN}]:")
print(M)

# Prova real
Q = np.random.randn(SEQ_LEN, D_MODEL)
K = np.random.randn(SEQ_LEN, D_MODEL)
scores_masked = (Q @ K.T) / np.sqrt(D_MODEL) + M
attn = softmax(scores_masked)

print(f"\nPesos de Atenção após Softmax:")
with np.printoptions(precision=4, suppress=True):
    print(attn)

upper = attn[np.triu_indices(SEQ_LEN, k=1)]
assert np.allclose(upper, 0.0), "FALHOU: posições futuras ≠ 0.0"
print("\n  ✓ Todas as probabilidades futuras são estritamente 0.0")


# ════════════════════════════════════════════════════════════
# TAREFA 2 — Cross-Attention
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 55)
print("  TAREFA 2 — Cross-Attention (Ponte Encoder-Decoder)")
print("═" * 55)

SEQ_ENC = 10
SEQ_DEC = 4
D_FULL  = 512

encoder_output = np.random.randn(BATCH_SIZE, SEQ_ENC, D_FULL)
decoder_state  = np.random.randn(BATCH_SIZE, SEQ_DEC, D_FULL)

print(f"\n  encoder_output : {encoder_output.shape}  (frase original)")
print(f"  decoder_state  : {decoder_state.shape}  (tokens gerados)")

cross = CrossAttention(D_FULL)
out, weights = cross.forward(encoder_output, decoder_state)

print(f"\n  output         : {out.shape}")
print(f"  attn_weights   : {weights.shape}")
assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-6)
print("  ✓ Cross-Attention OK — pesos somam 1.0 por linha")


# ════════════════════════════════════════════════════════════
# TAREFA 3 — Loop Auto-Regressivo
# ════════════════════════════════════════════════════════════
print()
encoder_out_small = np.random.randn(BATCH_SIZE, 10, D_MODEL)
resultado = autoregressive_loop(encoder_out_small, max_steps=20)

print("\n  ✓ Pipeline do Decoder concluído com sucesso!")
