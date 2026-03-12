Laboratório 3 — Implementando o Decoder

**Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
**Professor:** Prof. Dimmy Magalhães  
**Instituição:** iCEV - Instituto de Ensino Superior  



 Descrição

Implementação dos blocos matemáticos centrais do **Decoder** do Transformer,
continuando o Laboratório 2 (Encoder). Construído **sem** PyTorch, TensorFlow
ou Keras — apenas `Python 3.x`, `numpy` e `pandas`.


 Estrutura do Repositório


lab3_decoder/
│
├── main.py                      # Pipeline completo — executa as 3 tarefas
├── task1_causal_mask.py         # Tarefa 1: Máscara Causal (Look-Ahead Mask)
├── task2_cross_attention.py     # Tarefa 2: Cross-Attention (Ponte Encoder-Decoder)
├── task3_autoregressive_loop.py # Tarefa 3: Loop de Inferência Auto-Regressivo
├── requirements.txt             # Dependências do projeto
└── README.md                    # Este arquivo
```



Arquitetura Implementada


         ENCODER                        DECODER
  ┌─────────────────┐          ┌────────────────────────┐
  │  (Lab 2)        │          │  Masked Self-Attention  │
  │  Z contextual   │──────┐   │  + Máscara Causal (M)  │
  └─────────────────┘      │   ├────────────────────────┤
                            └──►  Cross-Attention        │
                                │  Q ← Decoder           │
                                │  K, V ← Encoder        │
                                ├────────────────────────┤
                                │  FFN + Add & Norm       │
                                ├────────────────────────┤
                                │  Projeção Linear        │
                                │  → vocab_size           │
                                ├────────────────────────┤
                                │  Softmax → probs        │
                                │  argmax  → token        │
                                └────────────────────────┘
                                         ↓
                                    Loop auto-regressivo
                                    (para no <EOS>)
```

 Componentes implementados

| Arquivo | Componente | Descrição |
|---------|------------|-----------|
| `task1_causal_mask.py` | `create_causal_mask(seq_len)` | Máscara triangular: 0 abaixo, -∞ acima da diagonal |
| `task1_causal_mask.py` | Prova Real | Softmax com máscara → posições futuras = 0.0 |
| `task2_cross_attention.py` | `CrossAttention` | Q do Decoder, K/V do Encoder, sem máscara |
| `task3_autoregressive_loop.py` | `generate_next_token()` | Mock do Decoder → distribuição de probs |
| `task3_autoregressive_loop.py` | `autoregressive_loop()` | Loop while com argmax + parada em `<EOS>` |

---

 Pré-requisitos

- Python 3.8 ou superior

---

 Como rodar

 1. Clone o repositório

```bash
git clone https://github.com/<seu-usuario>/lab3-transformer-decoder.git
cd lab3-transformer-decoder
```

 2. Instale as dependências

```bash
pip install -r requirements.txt
```

 3. Execute o pipeline completo

```bash
python main.py
```

 4. Execute as tarefas individualmente (opcional)

```bash
python task1_causal_mask.py         # Tarefa 1 isolada
python task2_cross_attention.py     # Tarefa 2 isolada
python task3_autoregressive_loop.py # Tarefa 3 isolada




 Saída Esperada


═══════════════════════════════════════════════════════
  TAREFA 1 — Máscara Causal (Look-Ahead Mask)
═══════════════════════════════════════════════════════
Máscara M [5×5]:
[[  0. -inf -inf -inf -inf]
 [  0.   0. -inf -inf -inf]
 ...]]
  ✓ Todas as probabilidades futuras são estritamente 0.0

═══════════════════════════════════════════════════════
  TAREFA 2 — Cross-Attention (Ponte Encoder-Decoder)
═══════════════════════════════════════════════════════
  encoder_output : (1, 10, 512)
  decoder_state  : (1, 4, 512)
  output         : (1, 4, 512)
  ✓ Cross-Attention OK

  Passo  1 | próximo token: 'palavra_X'
  ...
  Passo  5 | próximo token: '<EOS>'
  🛑 Token <EOS> detectado — geração encerrada.


 Nota de Integridade Acadêmica

Este projeto foi desenvolvido de forma autoral. Ferramentas de IA Generativa
(Claude) foram consultadas para fins de brainstorming sobre sintaxe do NumPy
e revisão das equações matemáticas do paper, e criaçao deste read conforme 
permitido pelo contrato pedagógico da disciplina. Todo o código foi escrito e
 compreendido pelo aluno.
