"""
Tarefa 3: Simulando o Loop de Inferência Auto-Regressivo

Fluxo por passo:
    1. Decoder processa a sequência atual
    2. Vetor final (d_model) é projetado para o vocabulário
    3. Softmax → distribuição de probabilidades
    4. argmax  → token com maior probabilidade
    5. Append  → token adicionado à sequência
    6. Para se token == <EOS>
"""

import numpy as np
from task1_causal_mask   import create_causal_mask, softmax
from task2_cross_attention import CrossAttention


# ─────────────────────────────────────────────────────────────
# Hiperparâmetros
# ─────────────────────────────────────────────────────────────
np.random.seed(42)

D_MODEL    = 64        # reduzido para CPU (paper: 512)
VOCAB_SIZE = 10_000    # conforme enunciado
MAX_STEPS  = 20        # limite de segurança para o loop

# Vocabulário fictício
# Palavras reais nas primeiras posições, <EOS> em posição fixa
vocab = (
    ["<PAD>", "<START>", "<EOS>"]
    + [f"palavra_{i}" for i in range(VOCAB_SIZE - 3)]
)
EOS_ID   = vocab.index("<EOS>")
START_ID = vocab.index("<START>")


# ─────────────────────────────────────────────────────────────
# Mock do Decoder completo
# ─────────────────────────────────────────────────────────────

# Pesos aleatórios fixos que simulam o Decoder treinado
_W_Q_self = np.random.randn(D_MODEL, D_MODEL) * 0.1
_W_K_self = np.random.randn(D_MODEL, D_MODEL) * 0.1
_W_V_self = np.random.randn(D_MODEL, D_MODEL) * 0.1
_cross_attn = CrossAttention(D_MODEL)
_W_proj  = np.random.randn(D_MODEL, VOCAB_SIZE) * 0.01  # projeção final

# Tabela de embeddings do Decoder
_embedding_table = np.random.randn(VOCAB_SIZE, D_MODEL) * 0.1


def _self_attention_masked(X):
    """
    Self-Attention causal aplicado à sequência atual do Decoder.
    X: (1, seq_len, d_model)
    """
    seq_len = X.shape[1]
    Q = X @ _W_Q_self
    K = X @ _W_K_self
    V = X @ _W_V_self

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(D_MODEL)

    # Aplica máscara causal
    M = create_causal_mask(seq_len)
    scores = scores + M

    attn_weights = softmax(scores)
    return attn_weights @ V


def _layer_norm(X, eps=1e-6):
    mean = np.mean(X, axis=-1, keepdims=True)
    var  = np.var(X,  axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(var + eps)


def generate_next_token(current_sequence, encoder_out):
    """
    Simula a passagem da sequência atual pelo Decoder.

    Parâmetros:
        current_sequence : list[str] — tokens gerados até agora
                           ex: ["<START>", "O", "rato"]
        encoder_out      : np.ndarray (1, seq_enc, d_model)
                           saída contextualizada do Encoder

    Retorna:
        probs            : np.ndarray (VOCAB_SIZE,)
                           distribuição de probabilidades do próximo token
    """
    seq_len = len(current_sequence)

    # 1. Converte tokens para IDs e busca embeddings
    token_ids = [
        vocab.index(t) if t in vocab else 1   # 1 = <UNK>
        for t in current_sequence
    ]
    X = _embedding_table[token_ids][np.newaxis, :, :]   # (1, seq, d_model)

    # 2. Sub-camada 1: Masked Self-Attention + Add & Norm
    X_self = _self_attention_masked(X)
    X = _layer_norm(X + X_self)

    # 3. Sub-camada 2: Cross-Attention + Add & Norm
    X_cross, _ = _cross_attn.forward(encoder_out, X)
    X = _layer_norm(X + X_cross)

    # 4. Pega apenas o vetor do último token gerado
    last_vector = X[0, -1, :]   # (d_model,)

    # 5. Projeção linear → vocabulário  +  Softmax
    logits = last_vector @ _W_proj          # (VOCAB_SIZE,)
    probs  = softmax(logits[np.newaxis, :])[0]  # (VOCAB_SIZE,)

    # Simulação didática: força <EOS> no 5º token gerado
    # Em um modelo real isso ocorre naturalmente pelo treinamento
    if len(current_sequence) >= 5:
        eos_probs = np.zeros(VOCAB_SIZE)
        eos_probs[EOS_ID] = 1.0
        return eos_probs

    return probs


# ─────────────────────────────────────────────────────────────
# Loop de Inferência Auto-Regressivo
# ─────────────────────────────────────────────────────────────

def autoregressive_loop(encoder_out, max_steps=MAX_STEPS):
    """
    Gera tokens um por vez até encontrar <EOS> ou atingir max_steps.
    """
    # Sequência começa com o token especial de início
    sequence = ["<START>"]

    print("=" * 55)
    print("  Tarefa 3 — Loop de Inferência Auto-Regressivo")
    print("=" * 55)
    print(f"\n  Sequência inicial : {sequence}")
    print(f"  Vocabulário       : {VOCAB_SIZE:,} tokens")
    print(f"  Limite de passos  : {max_steps}")
    print()

    step = 0
    while step < max_steps:
        step += 1

        # 1. Gera distribuição de probabilidades
        probs = generate_next_token(sequence, encoder_out)

        # 2. argmax → token mais provável
        next_id    = int(np.argmax(probs))
        next_token = vocab[next_id]
        confidence = probs[next_id] * 100

        print(f"  Passo {step:>2} | próximo token: '{next_token}'"
              f"  (id={next_id}, prob={confidence:.2f}%)")

        # 3. Adiciona à sequência
        sequence.append(next_token)

        # 4. Verifica critério de parada
        if next_token == "<EOS>":
            print(f"\n  🛑 Token <EOS> detectado — geração encerrada.")
            break
    else:
        print(f"\n  ⚠️  Limite de {max_steps} passos atingido.")

    # Frase final (sem <START> e <EOS>)
    frase_gerada = [t for t in sequence if t not in ("<START>", "<EOS>")]
    print(f"\n  Sequência completa : {sequence}")
    print(f"  Frase gerada       : {' '.join(frase_gerada) if frase_gerada else '(vazia)'}")
    print("=" * 55)

    return sequence


# ─────────────────────────────────────────────────────────────
# Execução
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Saída fictícia do Encoder (simula tradução francês → inglês)
    BATCH_SIZE  = 1
    SEQ_FRANCES = 10
    encoder_output = np.random.randn(BATCH_SIZE, SEQ_FRANCES, D_MODEL)

    resultado = autoregressive_loop(encoder_output)
