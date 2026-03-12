"""
Tarefa 1: Implementando a Máscara Causal (Look-Ahead Mask)

Impede que a palavra na posição i atenda à posição i+1
injetando uma máscara M antes do Softmax:

    Attention(Q, K, V) = softmax( QK^T / sqrt(dk) + M ) V
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# Softmax (reutilizado do Lab 2, numericamente estável)
# ─────────────────────────────────────────────────────────────

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# ─────────────────────────────────────────────────────────────
# Função principal: create_causal_mask
# ─────────────────────────────────────────────────────────────

def create_causal_mask(seq_len):
    """
    Cria a máscara causal (Look-Ahead Mask) de tamanho [seq_len, seq_len].

    - Triangular inferior + diagonal principal → 0
    - Triangular superior                      → -infinito

    Exemplo para seq_len=4:
        [[ 0, -inf, -inf, -inf],
         [ 0,    0, -inf, -inf],
         [ 0,    0,    0, -inf],
         [ 0,    0,    0,    0]]

    Após o Softmax, -inf vira 0.0 → posições futuras são ignoradas.
    """
    # np.triu retorna 1 nas posições acima da diagonal, 0 no resto
    upper = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    # Posições futuras (True) → -inf | posições permitidas (False) → 0
    mask = np.where(upper, -np.inf, 0.0)
    return mask


# ─────────────────────────────────────────────────────────────
# Prova Real
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    SEQ_LEN = 5
    D_MODEL  = 64

    # 1. Gera a máscara causal
    M = create_causal_mask(SEQ_LEN)

    print("=" * 55)
    print("  Tarefa 1 — Máscara Causal (Look-Ahead Mask)")
    print("=" * 55)
    print(f"\nMáscara M  [{SEQ_LEN}×{SEQ_LEN}]:")
    print(M)

    # 2. Matrizes fictícias Q e K  → (seq, d_model)
    Q = np.random.randn(SEQ_LEN, D_MODEL)
    K = np.random.randn(SEQ_LEN, D_MODEL)

    # 3. Produto escalar escalado
    scores = (Q @ K.T) / np.sqrt(D_MODEL)   # (seq, seq)

    # 4. Adiciona a máscara
    scores_masked = scores + M

    # 5. Softmax
    attn_weights = softmax(scores_masked)

    print(f"\nPesos de Atenção após Softmax  [{SEQ_LEN}×{SEQ_LEN}]:")
    # Formata para 4 casas decimais
    with np.printoptions(precision=4, suppress=True):
        print(attn_weights)

    # ── Prova formal ──────────────────────────────────────────
    upper_triangle = attn_weights[np.triu_indices(SEQ_LEN, k=1)]
    print(f"\nValores da triangular superior (palavras futuras):")
    print(f"  {upper_triangle}")

    assert np.allclose(upper_triangle, 0.0), \
        "FALHOU: posições futuras deveriam ser 0.0!"

    print("\n  ✓ PROVA REAL APROVADA:")
    print("    Todas as probabilidades de palavras futuras são estritamente 0.0")
    print("=" * 55)
