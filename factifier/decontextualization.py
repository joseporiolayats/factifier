from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseLanguageModel

__all__ = ["MolecularFactsDecontextualizer"]

class MolecularFactsDecontextualizer:
    def __init__(self, llm: BaseLanguageModel):
        """
        Initialize the decontextualizer with any LangChain-compatible LLM.

        Args:
            llm (BaseLanguageModel): A LangChain-compatible language model.
        """
        self.llm = llm
        # Create prompt template
        self.decontext_prompt = ChatPromptTemplate.from_messages([
            ("user",
             "Resolve ambiguities in the subclaim using context from the paragraph.\n"
             "Subclaim: {subclaim}\n"
             "Context: {context}\n"
             "Output the decontextualized subclaim. Do not add new information.")
        ])
        # Create the LCEL chain
        self.chain = (
            {
                "subclaim": RunnablePassthrough(),  # Receives 'subclaim' from chain input
                "context": RunnablePassthrough()    # Receives 'context' from chain input
            }
            | self.decontext_prompt
            | self.llm
            | StrOutputParser()
        )

    def decontextualize(self, subclaim: str, context: str) -> str:
        """
        Decontextualize a subclaim using the provided context (synchronous).

        Args:
            subclaim (str): The subclaim to decontextualize.
            context (str): The context to resolve ambiguities.

        Returns:
            str: The decontextualized subclaim.
        """
        # Invoke the chain with the provided inputs
        return self.chain.invoke({"subclaim": subclaim, "context": context})

    async def decontextualize_async(self, subclaim: str, context: str) -> str:
        """
        Decontextualize a subclaim using the provided context (asynchronous).

        Args:
            subclaim (str): The subclaim to decontextualize.
            context (str): The context to resolve ambiguities.

        Returns:
            str: The decontextualized subclaim.
        """
        # Invoke the chain asynchronously with the provided inputs
        return await self.chain.ainvoke({"subclaim": subclaim, "context": context})


if __name__ == "__main__":
    import asyncio
    from langchain_openai import ChatOpenAI

    # Initialize with GPT-4
    llm = ChatOpenAI(model="gpt-4")
    decontextualizer = MolecularFactsDecontextualizer(llm=llm)


    # Decontextualize a subclaim asynchronously
    async def main():
        result = await decontextualizer.decontextualize_async(
            subclaim="The reaction rate increased",
            context="The experiment was conducted at 300K with platinum catalyst"
        )
        print(result)


    # Run the async function
    asyncio.run(main())