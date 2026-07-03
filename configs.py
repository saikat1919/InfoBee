CHROMA_COLLECTION_NAME = "shikkha_bondhu_rag"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 200

GROQ_PRIMARY_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL1 = "llama-3.1-8b-instant"
GROQ_FALLBACK_MODEL2 = "llama-4-scout-17b-16e-instruct" #Large context and reasoning
GROQ_FALLBACK_MODEL3 = "deepseek-r1-distill-llama-70b" #High reasoning and math

GENERATION_TOP_K = 10

MAX_CONTEXT_CHARS_PER_CHUNK = 800

INTENT_CLASSIFIER_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
INTENT_CLASSIFIER_FALLBACK_MODEL_NAME = "llama-3.3-70b-versatile"
TABLE_DEDUP_OVERLAP_THRESHOLD = 0.7
BOILERPLATE_MIN_REPEAT_FRACTION = 0.5