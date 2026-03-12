"""
Tarefa 2: A Ponte Encoder-Decoder (Cross-Attention)

Diferente do Self-Attention, aqui:
  - Query (Q) vem do Decoder  (o que já foi gerado)
  - Key   (K) vem do Encoder  (memória da frase original)
  - Value (V) vem do Encoder  (memória da frase original)

Sem máscara causal — o Decoder pode olhar a frase do
Encoder por completo.
"""

import numpy as np


def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# ─────────────────────────────────────────────────────────────
# Cross-Attention
# ─────────────────────────────────────────────────────────────

class CrossAttention:
    """
    Implementa o Encoder-Decoder Attention:

        Q = decoder_state  @ W_Q
        K = encoder_output @ W_K
        V = encoder_output @ W_V

        output = softmax( Q K^T / sqrt(d_k) ) V
    """

    def __init__(self, d_model):
        self.d_model = d_model
        scale = np.sqrt(2.0 / d_model)
        # Pesos de projeção independentes
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale

    def forward(self, encoder_out, decoder_state):
        """
        encoder_out   : (batch, seq_enc, d_model)  ← memória do Encoder
        decoder_state : (batch, seq_dec, d_model)  ← tokens já gerados

        Retorna:
            output       : (batch, seq_dec, d_model)
            attn_weights : (batch, seq_dec, seq_enc)
        """
        # Projeções
        Q = decoder_state @ self.W_Q   # (batch, seq_dec, d_model)
        K = encoder_out   @ self.W_K   # (batch, seq_enc, d_model)
        V = encoder_out   @ self.W_V   # (batch, seq_enc, d_model)

        d_k = self.d_model

        # Produto escalar escalado
        # K.T em batch: transpõe os dois últimos eixos
        scores = Q @ K.transpose(0, 2, 1)   # (batch, seq_dec, seq_enc)
        scores = scores / np.sqrt(d_k)

        # Softmax — SEM máscara causal (acesso total ao Encoder)
        attn_weights = softmax(scores)       # (batch, seq_dec, seq_enc)

        # Soma ponderada dos valores
        output = attn_weights @ V            # (batch, seq_dec, d_model)

        return output, attn_weights


# ─────────────────────────────────────────────────────────────
# Demonstração
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    BATCH_SIZE   = 1
    SEQ_FRANCES  = 10   # comprimento da frase original (Encoder)
    SEQ_INGLES   = 4    # tokens já gerados pelo Decoder
    D_MODEL      = 512  # dimensão conforme o enunciado

    # Tensores fictícios
    encoder_output = np.random.randn(BATCH_SIZE, SEQ_FRANCES, D_MODEL)
    decoder_state  = np.random.randn(BATCH_SIZE, SEQ_INGLES,  D_MODEL)

    print("=" * 60)
    print("  Tarefa 2 — Cross-Attention (Ponte Encoder-Decoder)")
    print("=" * 60)
    print(f"\n  encoder_output shape : {encoder_output.shape}")
    print(f"  decoder_state  shape : {decoder_state.shape}")

    cross_attn = CrossAttention(D_MODEL)
    output, weights = cross_attn.forward(encoder_output, decoder_state)

    print(f"\n  output       shape   : {output.shape}")
    print(f"  attn_weights shape   : {weights.shape}")

    # Validação: cada linha de attn_weights deve somar 1 (distribuição)
    row_sums = weights.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), \
        "FALHOU: pesos de atenção não somam 1!"

    print("\n  ✓ Cross-Attention OK:")
    print(f"    Q vem do Decoder  ({SEQ_INGLES} tokens gerados)")
    print(f"    K, V vem do Encoder ({SEQ_FRANCES} tokens da frase original)")
    print(f"    Sem máscara causal — acesso completo ao Encoder")
    print(f"    Soma das linhas de attn_weights ≈ 1.0  ✓")
    print("=" * 60)
