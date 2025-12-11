Here is a compact full summary of what you are trying to do, plus how to handle the “classifier head → LM head mismatch” issue.

---

## 1. Full summary of the method you are aiming for

**Objective.**
Remove a specific topic (e.g., baseball) from a coherent open-source LLM by updating only the **embedding matrix**, using a temporary **classifier head** to drive unlearning, and then restoring the LM head for generation.

### 1.1 Setup

* Base model: transformer LM with

  * Embedding matrix (E \in \mathbb{R}^{V \times d})
  * Transformer blocks (T(\cdot))
  * LM head (W_{\text{LM}} \in \mathbb{R}^{V \times d}) (often tied to (E))
* You add a temporary **classifier head**:
  [
  z = C(h) \in \mathbb{R}^K
  ]
  where (h) is some pooled hidden representation (e.g., [CLS] or last token), and (K) includes a special “forgotten” class (k_f) = baseball.

**Frozen**: transformer (T) and classifier (C).
**Trainable**: embedding matrix (E) only.

---

### 1.2 Step 1 — Identify important tokens (Integrated Gradients)

For inputs that evoke baseball:

1. Compute scalar score (s_{\text{bb}}) (e.g., classifier logit for baseball, or LM-based score).
2. Use Integrated Gradients (IG) on **token embeddings**:
   [
   \text{IG}(e_i) \Rightarrow \text{importance}_i
   ]
3. Aggregate importance across many baseball examples to get:

   * A ranking of **token types** that are most important for the baseball class.
   * A per-token importance score for each training example.

These tell you which embeddings are **concept-critical** (e.g., “pitcher”, “inning”, “Yankees”), vs generic tone words (e.g., “exciting”) vs function words.

You can build:

* A **vocab mask** over important baseball tokens.
* Per-token weights (w_t) to weight the loss (optional).

---

### 1.3 Step 2 — Classifier-guided embedding-only unlearning

For each example (x) that should *not* be classified as baseball:

1. Compute:
   [
   h = T(E(x)),\quad z = C(h),\quad p = \text{softmax}(z)
   ]

2. Define a **target distribution** that zeroes out baseball and redistributes mass across the other classes:
   [
   p_{\text{target}, k_f} = 0, \quad
   p_{\text{target}, k} = \frac{1}{K-1} \text{ for } k \neq k_f
   ]

3. Loss (masked KL / cross-entropy without the forbidden class):
   [
   L(x) = -\frac{1}{K-1} \sum_{k \neq k_f} \log p_k
   ]

4. Optionally weight per token using IG-derived weights (w_t), if you implement a token-level variant.

5. Backpropagate:

   * Freeze (T) and (C).
   * Update only (E).

This pushes embeddings so that the classifier cannot confidently assign the baseball class; probability mass is redistributed to other classes.

---

### 1.4 Step 3 — Switch back to generation (LM head)

After sufficient unlearning steps:

1. Remove the classifier head (C).
2. Restore/use the LM head (W_{\text{LM}}).
3. Generate text (baseball prompts and non-baseball prompts).
4. Evaluate:

   * Suppression of baseball content.
   * Overall fluency.
   * Non-baseball capabilities.

This is the pipeline you described.

---

## 2. The “mismatch” problem when switching from classifier head back to LM head

You are observing that after training with the classifier head and then switching back to the LM head, there is usually some **mismatch** (bad generations, odd probabilities, etc.).

There are three main sources of mismatch and corresponding fixes.

---

### 2.1 Mismatch 1: LM head and embeddings are no longer aligned

If your LM uses **weight tying** (common in GPT-like models):

* Before unlearning: (W_{\text{LM}} \approx E^\top) (or explicitly tied).
* You then change (E) during unlearning, **but leave (W_{\text{LM}}) fixed**.

Result:
The LM head is still projecting onto the **old basis**, while embeddings are now in a different place. This yields inconsistent next-token probabilities.

**Fix A: Re-tie or re-synchronize weights after unlearning**

If the model architecture allows weight tying, do:

* Either run with explicit tying: (W_{\text{LM}}) always references the current (E).
* Or, after unlearning:
  [
  W_{\text{LM}} \leftarrow E
  ]
  (or (W_{\text{LM}}^\top \leftarrow E), depending on implementation).

This ensures that the LM head is consistent with the new embeddings.

If the model was not using tying originally, you can still “manually retie” at the end.

---

### 2.2 Mismatch 2: Classifier head uses a different representation than LM

If your classifier head:

* Uses a **different pooling** (e.g., [CLS] token) than the LM (which uses each token’s hidden state),
* Or has extra normalization / non-linearities not present in LM output,

then embeddings may be optimized for **classifier behavior** that is not exactly aligned with LM behavior.

Two ways to reduce this:

**Fix B: Build the classifier on top of the LM head itself**

Instead of a separate classifier that sits directly on top of (h), define the classifier using the LM logits:

1. Compute LM logits:
   [
   \ell = W_{\text{LM}} h \in \mathbb{R}^V
   ]

2. Map LM logits to topic classes with a fixed, small linear or log-sum-exp:

   * Example: binary baseball vs non-baseball:
     [
     z_{\text{bb}} = \log\sum_{v \in V_{\text{bb}}} \exp(\ell_v)
     ]
     [
     z_{\text{other}} = \log\sum_{v \notin V_{\text{bb}}} \exp(\ell_v)
     ]

   or for multi-class topics, similar constructions.

3. Use these (z) as classifier logits and define the same loss as above.

Now the classifier is **derived from the LM head**, so there is no structural mismatch: you are directly training embeddings to change LM logits in the way you want.

**Fix C: Tie classifier head parameters to LM head**

If you do want a linear classifier, make its weights a fixed function of the LM head (e.g., learned before unlearning, then frozen), and continue to share the underlying hidden representation and scaling.

---

### 2.3 Mismatch 3: The LM drifts on non-baseball data

Because you train only on “forget” data (baseball-related examples) with a strong unlearning loss, embeddings may also drift in ways that hurt general language modeling.

**Fix D: Add a distillation / retention regularizer**

Use the original model as a teacher:

* For some non-baseball corpus:

  * Compute teacher LM probabilities (p^{\text{teacher}}(\cdot \mid x)) with original weights.
  * Compute student LM probabilities (p^{\text{student}}(\cdot \mid x)) with updated embeddings.
  * Add a KL regularizer:
    [
    L_{\text{retain}} = \text{KL}(p^{\text{teacher}} ,|, p^{\text{student}})
    ]

Overall loss:

[
L = L_{\text{unlearn}} + \lambda_{\text{retain}} L_{\text{retain}}
]

This keeps the LM behavior close to the original on non-baseball topics, while still pushing embeddings to kill the baseball logit.

---

## 3. Recommended “clean” design to avoid mismatch

If you want the concept to be clean and mechanically aligned:

1. **Never introduce a completely separate classifier head.**
   Instead derive the topic logit directly from LM logits using baseball-token subsets.

2. **Always keep LM head tied to embeddings.**

   * Either use explicit weight tying, or copy (E \rightarrow W_{\text{LM}}) after embedding updates.

3. **Use teacher–student regularization on non-baseball data.**

   * This prevents global degradation.

With this design:

* There is no structural mismatch when you “remove the classifier head and put the LM head back”, because the classifier was never structurally separate from the LM head; it was just a readout over LM logits.
* The only change is that you stop using the topic-loss and go back to pure generation.

---

