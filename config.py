"""Central configuration for the chatbot app."""
import os

# --- Model ---------------------------------------------------------------
# Open-source model: Qwen3-4B-Instruct-2507, 4-bit quantised (GGUF), Apache-2.0.
# Chosen to fit this machine: CPU-only inference (the MX230's 2 GB of VRAM is
# too small to be worth offloading to), ~2.5 GB on disk, ~3 GB resident, which
# is comfortable against 24 GB of RAM. It replaced Qwen2.5-3B, which ran at the
# same speed on this CPU but had weaker recall (it expanded "RAG" wrongly on
# every attempt). The 3B and 1.5B files are still on disk -- point MODEL_PATH
# at the 1.5B one for roughly double the speed at lower answer quality.
DEFAULT_MODEL = r"D:\llm-models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL)

N_CTX = int(os.getenv("N_CTX", "4096"))                # context window in tokens
N_THREADS = int(os.getenv("N_THREADS", "6")) or None   # 0 -> let llama.cpp decide
# 6 measured fastest on this i5-10210U (4 cores / 8 threads): benchmarked at
# 4.4 tok/s vs 2.9 tok/s for llama.cpp's own auto-detection, which under-uses
# the chip. Re-tune with N_THREADS=<n> if you move to different hardware.
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))     # CPU-only build

# --- Generation ----------------------------------------------------------
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "768"))   # room for long answers to finish
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.1"))

# --- Conversation --------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant having an ongoing conversation with a user.

Guidelines:
- Use the earlier turns of this conversation to resolve follow-up questions. References such as "it", "its components", "the second one" or "explain that further" refer back to what was just discussed.
- The user may change the subject at any point. Phrases like "forget that", "never mind", "different question" or "let's switch topics" mean they want to move on: answer the new question on its own terms and drop the old thread. They are never a reason to decline.
- Always attempt an ordinary question. Everyday requests (recipes, travel, code, definitions, small talk) are all fine to answer. Never reply that you cannot assist with a harmless request.
- If a follow-up is genuinely ambiguous, ask one short clarifying question.
- Only state facts you are confident in. If you are unsure what an acronym or term means, say so rather than inventing an expansion.
- Answer in clear, well-structured prose. Use short markdown lists only when they genuinely help. Keep answers focused rather than padded.
- Be concise. Aim for a few short paragraphs; go longer only when the user asks for detail. Do not pad answers with emoji, decorative headings or horizontal rules."""

# Leave room for the reply when packing history into the context window.
HISTORY_TOKEN_BUDGET = N_CTX - MAX_TOKENS - 256

# --- RAG ------------------------------------------------------------------
# Embedding model: BGE-small-en-v1.5 (MIT), 33M params, 384-dim, 512-token
# limit. Runs under the same llama.cpp runtime as the chat model, so RAG adds
# no new heavy dependency and the system stays fully offline.
EMBED_MODEL_PATH = os.getenv(
    "EMBED_MODEL_PATH", r"D:\llm-models\bge-small-en-v1.5-f16.gguf"
)
EMBED_N_CTX = int(os.getenv("EMBED_N_CTX", "512"))

# Each corpus keeps its own folder pair, so indexes are not mixed: retrieval
# across two unrelated reports pulls the wrong document into half the answers.
# Point these at another pair to switch corpus, e.g.
#   set DATA_DIR=data & set INDEX_DIR=index      (the GPO annual report)
DATA_DIR = os.getenv("DATA_DIR", "data_gep")     # source documents live here
INDEX_DIR = os.getenv("INDEX_DIR", "index_gep")  # vectors + chunk metadata

# Token budget for the retrieved passages in a document-mode prompt.
#
# This used to be sized to fill the context window (N_CTX - MAX_TOKENS - 700
# = 2,628 tokens), which is the wrong objective on a CPU. Measured on this
# machine, the model reads a prompt at 21 tokens/s and writes an answer at 6,
# so a 1,428-token prompt costs 68 seconds before a single word is generated
# -- roughly twice what the answer itself costs. Passages past the third or
# fourth are weak hits that rarely change the answer and always cost that
# time, so the budget is now set for latency rather than for capacity.
DOC_CONTEXT_TOKENS = int(os.getenv("DOC_CONTEXT_TOKENS", "900"))

# Answers in document mode are extractive -- a figure, a sentence, a citation
# -- so they do not need the headroom a free-form chat answer does, and every
# token costs a sixth of a second.
DOC_MAX_TOKENS = int(os.getenv("DOC_MAX_TOKENS", "320"))

# Sessions idle for longer than this are evicted (seconds).
SESSION_TTL = int(os.getenv("SESSION_TTL", str(6 * 60 * 60)))
