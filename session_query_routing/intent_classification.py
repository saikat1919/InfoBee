import re
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from configs import INTENT_CLASSIFIER_MODEL_NAME, INTENT_CLASSIFIER_FALLBACK_MODEL_NAME


class QueryIntent(str, Enum):
    FACTUAL = "FACTUAL"
    EXPLAIN = "EXPLAIN"



_EXPLAIN_PATTERNS = [
    r"^\s*why\b",
    r"^\s*how (does|do|did|is|are|was|were|can|could)\b",
    r"\bexplain\b",
    r"\bwalk me through\b",
    r"\bwhat('s| is| does) the (reason|logic|intuition) (behind|for)\b",
    r"\bwhat does .+ mean\b",
    r"\bcan you (clarify|elaborate)\b",
    r"\bin (simple|other|plain) (terms|words)\b",
    r"\bwhat('s| is) the difference between\b",
    r"\bbreak (this|that|it) down\b",
    r"\bhelp me understand\b",
]
_EXPLAIN_REGEX = re.compile("|".join(_EXPLAIN_PATTERNS), flags=re.IGNORECASE)

_FACTUAL_OVERRIDE_PATTERNS = [
    r"^\s*how (many|much|long|old|far|often|tall)\b",
    r"^\s*what (is|are|was|were)\b.*\b(value|number|amount|date|name|page|total)\b",
]
_FACTUAL_OVERRIDE_REGEX = re.compile("|".join(_FACTUAL_OVERRIDE_PATTERNS), flags=re.IGNORECASE)


def _regex_classify(query: str):
    if _FACTUAL_OVERRIDE_REGEX.search(query):
        return QueryIntent.FACTUAL
    if _EXPLAIN_REGEX.search(query):
        return QueryIntent.EXPLAIN
    return None


_CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for a document Q&A system.
Classify the user's question into exactly one of two labels:

FACTUAL - the user wants a specific fact, value, name, date, or statement
          that should be looked up directly from the source document.
EXPLAIN - the user wants reasoning, justification, or an explanation of why
          or how something works, even if the document doesn't spell out
          every step of the reasoning itself.

Respond with ONLY the single word FACTUAL or EXPLAIN. No punctuation, no
explanation, nothing else."""

_classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", _CLASSIFIER_SYSTEM_PROMPT),
    ("human", "{query}"),
])

_classifier_model = None


def _get_classifier_model():
    global _classifier_model
    if _classifier_model is None:
        hf_llm = HuggingFaceEndpoint(
            repo_id=INTENT_CLASSIFIER_MODEL_NAME,
            provider="auto",
            task="text-generation",
            max_new_tokens=4,
            do_sample=False,
        )
        primary = ChatHuggingFace(llm=hf_llm)

        fallback = ChatGroq(
            model=INTENT_CLASSIFIER_FALLBACK_MODEL_NAME,
            temperature=0,
            max_tokens=4,
        )

        _classifier_model = primary.with_fallbacks([fallback])
    return _classifier_model


def _llm_classify(query: str) -> QueryIntent:
    try:
        model = _get_classifier_model()
        chain = _classifier_prompt | model
        result = chain.invoke({"query": query})
        label = result.content.strip().upper()
        if "EXPLAIN" in label:
            return QueryIntent.EXPLAIN
        return QueryIntent.FACTUAL
    except Exception as e:
        print(f"[routing] intent classifier failed, defaulting to FACTUAL: {e}")
        return QueryIntent.FACTUAL


def classify_intent(query: str) -> QueryIntent:
    regex_result = _regex_classify(query)
    if regex_result is not None:
        return regex_result
    return _llm_classify(query)