"""Framework integrations for KowitoDB.

These adapters let KowitoDB act as a retriever / vector store inside popular
LLM frameworks. Import the submodule for the framework you use (each requires
that framework to be installed):

- LangChain::

      pip install "kowitodb[langchain]"
      from kowitodb.integrations.langchain import KowitoDBRetriever, KowitoDBVectorStore

- LlamaIndex::

      pip install "kowitodb[llamaindex]"
      from kowitodb.integrations.llamaindex import KowitoDBRetriever

The submodules are intentionally not imported here so that installing one
framework does not require the other.
"""

__all__ = ["langchain", "llamaindex"]
