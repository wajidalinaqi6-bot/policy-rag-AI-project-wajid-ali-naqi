"""
rag_pipeline.py – RAG orchestration layer.

FIX: LangChain 0.1.x deprecated top-level imports like:
       from langchain.chat_models import ChatOpenAI
       from langchain.chains import LLMChain
     These now live in langchain_openai / langchain_anthropic /
     langchain_community and langchain.chains respectively.
     We use try/except to support both old and new import paths so the
     code works whether the user has langchain 0.1.x or 0.2.x installed.
"""

from typing import List, Dict, Any
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import config


class RAGPipeline:
    def __init__(self, vector_store, llm=None):
        self.vector_store = vector_store
        self.llm = llm
        self._initialize_llm()
        self._setup_prompt()

    def _initialize_llm(self):
        """Initialize the LLM based on available API keys.

        Priority: OpenAI → Groq → Anthropic.
        Raises ValueError if no key is configured.
        """
        if self.llm is not None:
            return

        if config.OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                from langchain.chat_models import ChatOpenAI  # 0.1.x fallback

            self.llm = ChatOpenAI(
                model=config.MODEL_NAME,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                api_key=config.OPENAI_API_KEY
            )

        elif config.GROQ_API_KEY:
            # FIX: Use ChatGroq directly instead of ChatOpenAI with Groq endpoint
            try:
                from langchain_groq import ChatGroq
            except ImportError:
                raise ImportError(
                    "langchain-groq is required for Groq API. "
                    "Run: pip install langchain-groq"
                )
            
            self.llm = ChatGroq(
                groq_api_key=config.GROQ_API_KEY,
                model_name="llama-3.3-70b-versatile",
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )

        elif config.ANTHROPIC_API_KEY:
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                from langchain.chat_models import ChatAnthropic  # 0.1.x fallback

            self.llm = ChatAnthropic(
                model="claude-3-haiku-20240307",
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                anthropic_api_key=config.ANTHROPIC_API_KEY
            )

        else:
            raise ValueError(
                "No LLM provider configured. "
                "Set OPENAI_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY in your .env file."
            )

    def _setup_prompt(self):
        """Set up the prompt template."""
        template = """You are a helpful assistant that answers questions about company policies and procedures.

Context information from company policy documents:
{context}

Question: {question}

Instructions:
- Only answer questions about company policies and procedures.
- If the question is not related to company policies, politely say:
  "I can only answer questions about our company policies and procedures."
- Always cite the source document(s) for your answer using the format [Source: filename].
- Keep your answer concise and within {max_length} words.
- If the information is not in the provided context, say so clearly.

Answer:"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
            partial_variables={"max_length": str(config.MAX_ANSWER_LENGTH)}
        )

    def _is_valid_topic(self, query: str) -> bool:
        """Return True if the query mentions at least one allowed policy topic."""
        query_lower = query.lower()
        return any(topic in query_lower for topic in config.ALLOWED_TOPICS)

    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents as numbered context blocks."""
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('filename', 'Unknown')
            parts.append(f"[Document {i}] (Source: {source})\n{doc.page_content}")
        return "\n\n".join(parts)

    def _format_citations(self, documents: List[Document]) -> List[Dict[str, str]]:
        """Build a deduplicated list of citation dicts for the frontend."""
        citations = []
        seen: set = set()
        for doc in documents:
            source = doc.metadata.get('filename', 'Unknown')
            if source not in seen:
                seen.add(source)
                snippet = doc.page_content
                citations.append({
                    'source':  source,
                    'content': (snippet[:200] + "…") if len(snippet) > 200 else snippet
                })
        return citations

    def answer(
        self,
        query: str,
        k: int = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """Answer a policy question using the RAG pipeline."""

        # Topic guard — refuse off-topic queries without hitting the LLM
        if not self._is_valid_topic(query):
            return {
                'answer':      "I can only answer questions about our company policies and procedures.",
                'citations':   [],
                'success':     True,
                'topic_valid': False
            }

        if k is None:
            k = config.TOP_K

        retrieved_docs = self.vector_store.similarity_search(query, k=k)

        if not retrieved_docs:
            return {
                'answer':      "I couldn't find relevant information in the company policy documents.",
                'citations':   [],
                'success':     True,
                'topic_valid': True
            }

        context = self._format_context(retrieved_docs)

        try:
            # FIX: Use the updated LangChain chain invocation pattern.
            # LLMChain.run() is deprecated in 0.2.x; use invoke() instead.
            # We try invoke() first and fall back to run() for 0.1.x.
            from langchain.chains import LLMChain
            chain = LLMChain(llm=self.llm, prompt=self.prompt)

            try:
                response = chain.invoke({'context': context, 'question': query})
                # invoke() returns a dict; the answer is under the output key
                answer = response.get('text', response.get('answer', '')).strip()
            except AttributeError:
                # Older LangChain 0.1.x path
                answer = chain.run({'context': context, 'question': query}).strip()

            citations = self._format_citations(retrieved_docs) if return_sources else []

            return {
                'answer':        answer,
                'citations':     citations,
                'success':       True,
                'topic_valid':   True,
                'retrieved_docs': len(retrieved_docs)
            }

        except Exception as e:
            return {
                'answer':    f"An error occurred while processing your question: {str(e)}",
                'citations': [],
                'success':   False,
                'topic_valid': True,
                'error':     str(e)
            }


def create_rag_pipeline(vector_store) -> RAGPipeline:
    """Factory — returns a configured RAGPipeline instance."""
    return RAGPipeline(vector_store)