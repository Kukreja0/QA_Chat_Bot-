# LLM Theory — Basic to Advanced (Complete Notes)

---

## 1. What is an LLM, really?

An LLM (Large Language Model) is a neural network — specifically a **Transformer decoder** — trained to do one simple task extremely well: **predict the next token given all previous tokens.**

That's it. Everything else (chatting, coding, reasoning, summarizing) emerges from that one skill applied repeatedly, at massive scale.

Think of it like this: if you trained a model on "The capital of France is ___" billions of times across the entire internet, it would learn to say "Paris." If you trained it on billions of conversations, code snippets, essays, and Q&A pairs, it learns the *patterns* of language, facts, reasoning steps, and even style — all by learning "what word comes next" over and over.

**Key insight:** an LLM does NOT have a database of facts it looks up. It has learned **statistical patterns of language** so well that reciting facts, writing code, and reasoning all look like "predict the next most likely token" to the model internally.

---

## 2. How is an LLM trained? (The 3 Stages)

### Stage 1: Pretraining (the expensive, foundational stage)
- **Data:** Trillions of tokens scraped from the internet, books, code repositories, Wikipedia, etc.
- **Task:** Causal Language Modeling (CLM) — given tokens 1...N, predict token N+1. Repeat for every position in every document.
- **Compute:** Thousands of GPUs/TPUs running for weeks to months. This costs millions of dollars and is why only large companies (OpenAI, Google, Meta, Anthropic, etc.) do this from scratch.
- **What the model learns here:** grammar, facts, world knowledge, reasoning patterns, coding syntax, multiple languages — purely as a side effect of getting good at next-token-prediction.
- **Output of this stage:** a "base model" / "foundation model." It can complete text, but it's NOT good at following instructions. Ask a base model "What is the capital of France?" and it might respond with "What is the capital of Germany?" (because that's a statistically plausible continuation in a quiz-style document), not necessarily "Paris."

### Stage 2: Supervised Fine-Tuning / Instruction Tuning
- **Data:** Much smaller dataset (thousands to low millions of examples) of (instruction, ideal response) pairs, written/curated by humans.
- **Task:** Same next-token-prediction objective, but now ONLY on this curated instruction-response data.
- **What changes:** the model learns the *format* of being a helpful assistant — answering questions directly, following commands, refusing harmful requests, formatting code properly.
- **Output:** an "instruct" or "chat" model (e.g., `Llama-3-8B-Instruct` vs the base `Llama-3-8B`).

### Stage 3: Alignment (RLHF / DPO)
- **Goal:** make outputs not just "correctly formatted" but actually preferred by humans — helpful, harmless, honest.
- **RLHF (Reinforcement Learning from Human Feedback):**
  1. Humans rank multiple model outputs for the same prompt (best to worst).
  2. A separate "reward model" is trained to predict these human preference scores.
  3. The LLM is then fine-tuned using reinforcement learning (PPO algorithm) to maximize the reward model's score.
- **DPO (Direct Preference Optimization)** — the modern, cheaper alternative: skips training a separate reward model entirely and directly optimizes the LLM on preference pairs (chosen response vs rejected response) using a clever loss function. As of 2026, most companies use DPO (or its variants like ORPO) instead of full RLHF because it's cheaper, more stable, and gets comparable results.
- **Output:** the final assistant you interact with (ChatGPT, Claude, Gemini, etc.)

**Why this 3-stage pipeline matters for you:** when you see "fine-tuning" in a job description, 95% of the time they mean Stage 2-style fine-tuning (and usually via LoRA/QLoRA, not full fine-tuning) — NOT pretraining from scratch, and almost never full RLHF.

---

## 3. What happens when we "give new data" to an LLM? (Critical Concept)

This is where a LOT of confusion happens for newcomers. There are actually **THREE completely different things** people mean by "giving new data," and they behave very differently:

### Option A: Putting data in the prompt (Context / In-Context Learning)
- You paste text directly into your prompt: "Here is a document: [text]. Answer this question based on it: ..."
- **What happens internally:** NOTHING is learned or stored. The model's weights are completely unchanged. The text just becomes part of the input tokens that get processed through attention for THIS one request.
- **Lifespan:** the data is forgotten the instant the conversation/request ends (unless you paste it again).
- **Limit:** bounded by the context window (e.g., 128K–1M tokens depending on model).
- **This is what RAG uses** — it retrieves relevant chunks and stuffs them into the context window at query time.

### Option B: Fine-tuning (updating weights, even a few)
- You provide example (input, output) pairs and run a training loop.
- **What happens internally:** the model's weights (all of them in full FT, or a small adapter subset in LoRA) are actually mathematically updated via backpropagation and gradient descent.
- **Lifespan:** permanent (until you fine-tune again or replace the weights). The model now "behaves differently" on every future request, even with no special prompt.
- **Important misconception to avoid:** fine-tuning is BAD at teaching the model new facts reliably. If you fine-tune on "Our company's return policy is 30 days," the model may or may not reliably recall this exact fact later — it's more likely to *learn a style or behavior pattern* than to *memorize a fact precisely*. This is why RAG (Option A, dynamically) is preferred for facts/knowledge, and fine-tuning is preferred for behavior/format/tone.

### Option C: RAG (Retrieval-Augmented Generation) — combines A + a retrieval step
- A retrieval system (vector DB) finds the most relevant chunks of your data for the CURRENT query, and automatically inserts them into the prompt (Option A), at request time.
- **What happens internally:** same as Option A — no weights change. But now it's automated and scalable (you don't have to manually paste documents).
- **This is why "RAG vs fine-tuning" is such a common interview question** — they solve different problems and people often confuse them.

**One-line summary to remember:**
> Context (RAG) = temporary, perfect recall of what you give it, but limited by context window size.
> Fine-tuning = permanent change to behavior/style, but unreliable for precise fact recall, and can't easily be "updated" without retraining.

---

## 4. How does an LLM actually generate a response? (Step by step)

1. **Tokenization:** Your input text is split into tokens (subword pieces, via BPE/SentencePiece — recall Day 1). "Hello world" might become `["Hello", " world"]` or similar subword splits.
2. **Embedding:** Each token ID is converted into a dense vector (embedding) via a lookup table.
3. **Positional encoding** is added so the model knows token order.
4. **Pass through Transformer decoder layers:** Each layer applies self-attention (every token "looks at" all previous tokens to gather context) followed by a feed-forward network. This repeats for dozens of layers (e.g., 32 layers in a 7B model).
5. **Output logits:** The final layer produces a probability distribution over the ENTIRE vocabulary (e.g., 50,000–150,000 possible tokens) for "what should the next token be?"
6. **Sampling/decoding strategy** picks the actual next token from that probability distribution:
   - **Greedy decoding:** always pick the highest-probability token (deterministic, can be repetitive/boring).
   - **Temperature sampling:** higher temperature = flatter distribution = more random/creative output; lower temperature = more focused/deterministic.
   - **Top-k sampling:** only consider the k most likely tokens, then sample.
   - **Top-p (nucleus) sampling:** only consider tokens whose cumulative probability adds up to p (e.g., 0.9), then sample. Most production systems use top-p + temperature together.
7. **Repeat:** the newly generated token gets appended to the sequence, and the WHOLE process (steps 1-6) repeats to generate the NEXT token. This is why LLMs generate text one token at a time, and why longer responses take proportionally longer ("autoregressive generation").
8. **Stop condition:** generation halts when the model produces a special "end of sequence" token, or hits a max-token limit you set.

**Why this matters practically:**
- This is why setting `temperature=0` gives near-deterministic output (useful for production tasks like classification) and `temperature=0.7-1.0` gives more creative/varied output (useful for creative writing/chat).
- This is why LLMs can "hallucinate" — at every single token step, they're picking a *statistically plausible* next token, not "looking up" whether it's true. If the statistically plausible path leads to a fabricated fact, the model has no internal mechanism to stop and say "wait, I'm not sure this is true" unless it was specifically trained/prompted to do that.
- This is why longer outputs cost more (API pricing is per-token) and take more time (each token is a full forward pass through the network).

---

## 5. Key Architectural Concepts You Should Know

### Context Window
The maximum number of tokens (input + output combined, in most APIs) the model can "see" at once. Modern models range from 128K to 2M tokens. Bigger context = can paste more documents directly, but also slower and costlier per request, and models can have "lost in the middle" issues (worse recall for info buried in the middle of a huge context vs the start/end).

### Parameters
The number of learnable weights in the model (e.g., "7B" = 7 billion parameters). More parameters generally = more capability, but also more memory/compute needed. This is why a 7B model fits on a consumer GPU but a 70B+ model needs serious hardware (or quantization).

### Mixture of Experts (MoE) — modern architecture trend
Instead of every parameter being used for every token, MoE models have many "expert" sub-networks, and a routing mechanism picks only a few experts per token. This means a model can have huge TOTAL parameters (e.g., 1.6 trillion) but only a fraction ("active parameters", e.g., 49 billion) are actually computed per token — making it cheaper to run than its total size suggests. DeepSeek-V4 and many 2026-era frontier models use this.

### Quantization
Storing model weights in lower precision (e.g., 4-bit instead of 32-bit floating point) to drastically reduce memory usage, at a small cost to accuracy. This is THE technique that makes running/fine-tuning large models on consumer GPUs possible (this is what the "Q" in QLoRA stands for).

### Temperature, Top-k, Top-p
Covered above in generation — these are the "creativity dials" you control via API parameters.

### Hallucination
The model generates fluent, confident-sounding text that is factually wrong or unsupported. Root cause: the model is fundamentally a next-token predictor, not a fact-checker. Mitigated via RAG (ground answers in retrieved real text), better prompting ("only answer from the provided context, say 'I don't know' if not present"), and output verification layers.

### Context vs Parametric Knowledge
- **Parametric knowledge:** facts baked into the model's weights during pretraining (frozen at training cutoff, can be wrong/outdated, can't be easily updated).
- **Contextual knowledge:** facts provided at inference time via the prompt/RAG (always current, but limited by context window, must be provided every time).

---

## 6. Common Beginner Misconceptions (Read this carefully)

| Misconception | Reality |
|---|---|
| "The LLM looks up answers from a database" | No database. It's pattern-matching learned during training, expressed token-by-token. |
| "If I fine-tune on my company docs, the model will know all the facts perfectly" | Wrong tool. Fine-tuning teaches style/behavior, not reliable fact recall. Use RAG for facts. |
| "Bigger context window means I don't need RAG anymore" | Even with huge context, stuffing irrelevant info hurts accuracy/cost. RAG's *retrieval* (finding the RIGHT info) still matters even with big windows. |
| "The model remembers our previous conversation because it's smart" | The model is stateless. The ENTIRE conversation history is re-sent as text in the prompt every single time you send a new message. The "memory" is just your chat app resending history. |
| "Fine-tuning will make my model 'know' today's news" | Fine-tuning data is frozen at training time — it's still "static knowledge," just like pretraining. Only live retrieval (RAG/search) gives truly current info. |
| "More parameters always = better" | Diminishing returns + cost/speed tradeoffs. A well-tuned 8B model with good RAG can beat a 70B model with bad retrieval, for many practical tasks. |

---

## 7. Quick Glossary (for interview readiness)

- **Token:** smallest unit of text the model processes (subword piece).
- **Logits:** raw, unnormalized scores the model outputs before converting to probabilities.
- **Inference:** running the model to generate output (as opposed to training it).
- **Context window:** max tokens the model can process in one call.
- **Prompt engineering:** crafting input text to get better outputs, without changing weights.
- **Fine-tuning:** updating model weights on new data.
- **PEFT:** Parameter-Efficient Fine-Tuning — updating only a small subset of weights (e.g., LoRA).
- **RAG:** Retrieval-Augmented Generation — fetching relevant external data and inserting into the prompt before generation.
- **Embedding:** dense vector representation of text capturing meaning.
- **Quantization:** reducing weight precision to save memory.
- **Hallucination:** confidently generated but false/unsupported content.
- **Alignment:** the process of making model behavior match human values/preferences (RLHF/DPO).
