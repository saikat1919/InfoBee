import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from configs import GROQ_PRIMARY_MODEL_NAME, GROQ_FALLBACK_MODEL1, GROQ_FALLBACK_MODEL2
from session_query_routing.intent_classification import classify_intent, QueryIntent

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONTEXTUALIZE_SYSTEM_PROMPT = """Given a chat history and the latest user question, \
which might reference context in the chat history, formulate a standalone \
question which can be understood without the chat history. Do NOT answer the \
question, just reformulate it if needed, and otherwise return it as is."""

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
])

FACTUAL_SYSTEM_PROMPT = """
You are AskMe, a question-answering assistant that answers strictly using the provided context. Follow these rules:
1. Only use information present in the context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, say so explicitly instead of guessing.
3. Be concise and direct. Do not repeat the question back.
4. If multiple people appear in the context, only use facts explicitly tied to the person named in the question. Never merge or borrow details from a different person's excerpt, even if the excerpts look similar.
5. You may use the chat history to understand what the user is referring to and to avoid repeating yourself, but the FACTS in your answer must still come only from the context below.

Context:
{context}
"""

EXPLAIN_SYSTEM_PROMPT = """
You are ShikkhaBondhu, a question-answering assistant helping the user(students) understand material from their uploaded document(s). The user is asking for an EXPLANATION, not just a fact lookup. Follow these rules:
1. Start from the retrieved context below -- quote or paraphrase the specific passage, equation, or statement the user is asking about, so they can see exactly what you're explaining.
2. If the context itself contains the explanation, use it.
3. If the context does NOT fully explain the "why" or "how", you MAY use your own general knowledge to fill in the reasoning -- but you MUST clearly separate this from the document content. Use a sentence like "The document doesn't explain this directly, but here's why this is generally true:" before adding outside reasoning.
4. Never silently blend outside knowledge into what looks like document content. The user must always be able to tell what came from their document versus what you added.
5. If multiple people or sources appear in the context, only use facts explicitly tied to the entity named in the question. Never merge or borrow details across different excerpts, even if they look similar.
6. Be clear and pedagogical, but don't pad the answer -- get to the explanation.
7. You may use the chat history to understand what the user is referring to.

Context:
{context}
"""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", FACTUAL_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", EXPLAIN_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])



def _build_model():
    groq_primary_model = ChatGroq(
        model=GROQ_PRIMARY_MODEL_NAME,
        temperature=0.4,
        max_tokens=512
    )

    primary_model = groq_primary_model

    fallback_model_1 = ChatGroq(
        model=GROQ_FALLBACK_MODEL1,
        temperature=0.4,
        max_tokens=512
    )

    fallback_model_2 = ChatGroq(
        model=GROQ_FALLBACK_MODEL2,
        temperature=0.4,
        max_tokens=512
    )


    tiers = [
        ("Groq_Primary", GROQ_PRIMARY_MODEL_NAME, primary_model),
        ("Groq_Fallback_1", GROQ_FALLBACK_MODEL1, fallback_model_1),
        ("Groq_Fallback_2", GROQ_FALLBACK_MODEL2, fallback_model_2),
    ]

    def _invoke_with_logged_tiers(messages, **kwargs):
        last_exc = None
        for tier_name, model_name, model in tiers:
            try:
                result = model.invoke(messages, **kwargs)
                if last_exc is not None:
                    logger.info("[generation] %s (%s) succeeded after earlier tier(s) failed.",
                                tier_name, model_name)
                return result
            except Exception as exc:
                logger.warning(
                    "[generation] TIER FAILED -- %s (%s): %s: %s",
                    tier_name, model_name, type(exc).__name__, exc,
                )
                last_exc = exc
                continue
        logger.error("[generation] ALL TIERS FAILED. Raising the last exception encountered.")
        raise last_exc

    return RunnableLambda(_invoke_with_logged_tiers)


def build_rag_chain(retriever):
    model = _build_model()

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_prompt
    )

    factual_chain = create_stuff_documents_chain(model, answer_prompt)
    explain_chain = create_stuff_documents_chain(model, explain_prompt)

    factual_rag_chain = create_retrieval_chain(history_aware_retriever, factual_chain)
    explain_rag_chain = create_retrieval_chain(history_aware_retriever, explain_chain)

    return {
        QueryIntent.FACTUAL: factual_rag_chain,
        QueryIntent.EXPLAIN: explain_rag_chain,
    }


def answer_question(rag_chains, query, chat_history):
    intent = classify_intent(query)
    chain = rag_chains[intent]
    result = chain.invoke({"input": query, "chat_history": chat_history})
    return result["answer"], intent