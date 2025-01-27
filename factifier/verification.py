# factifier/verification.py
# Verification (DnDScore)
# Verify subclaims agains reference documents using both atomic and decontextualized forms

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseLanguageModel

__all__ = ["DnDScoreVerifier"]

class DnDScoreVerifier:
    def __init__(self, llm: BaseLanguageModel):
        """
        Initialize the verifier with any LangChain-compatible LLM.

        Args:
            llm (BaseLanguageModel): A LangChain-compatible language model.
        """
        self.llm = llm
        # Create prompt template
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("user",
             "Reference: {reference}\n"
             "Context: {decontext_claim}\n"
             "Claim: {atomic_claim}\n"
             "Is the claim supported by the reference? Answer yes or no.")
        ])
        # Create the LCEL chain
        self.chain = (
            {
                "atomic_claim": RunnablePassthrough(),  # Receives 'atomic_claim' from chain input
                "decontext_claim": RunnablePassthrough(),  # Receives 'decontext_claim' from chain input
                "reference": RunnablePassthrough()  # Receives 'reference' from chain input
            }
            | self.verification_prompt
            | self.llm
            | StrOutputParser()
        )

    def verify(self, atomic_claim: str, decontext_claim: str, reference: str) -> bool:
        """
        Verify if the atomic claim is supported by the reference (synchronous).

        Args:
            atomic_claim (str): The atomic claim to verify.
            decontext_claim (str): The decontextualized claim for additional context.
            reference (str): The reference document to verify against.

        Returns:
            bool: True if the claim is supported, False otherwise.
        """
        # Invoke the chain with the provided inputs
        result = self.chain.invoke({
            "atomic_claim": atomic_claim,
            "decontext_claim": decontext_claim,
            "reference": reference
        })
        return result.strip().lower() == "yes"

    async def verify_async(self, atomic_claim: str, decontext_claim: str, reference: str) -> bool:
        """
        Verify if the atomic claim is supported by the reference (asynchronous).

        Args:
            atomic_claim (str): The atomic claim to verify.
            decontext_claim (str): The decontextualized claim for additional context.
            reference (str): The reference document to verify against.

        Returns:
            bool: True if the claim is supported, False otherwise.
        """
        # Invoke the chain asynchronously with the provided inputs
        result = await self.chain.ainvoke({
            "atomic_claim": atomic_claim,
            "decontext_claim": decontext_claim,
            "reference": reference
        })
        return result.strip().lower() == "yes"


if __name__ == "__main__":
    import asyncio
    from langchain_openai import ChatOpenAI

    # Initialize with GPT-4
    llm = ChatOpenAI(model="gpt-4")
    verifier = DnDScoreVerifier(llm=llm)


    # Verify a claim asynchronously
    async def main():
        is_supported = await verifier.verify_async(
            atomic_claim="The reaction rate increased",
            decontext_claim="The experiment was conducted at 300K with platinum catalyst",
            reference="The study found that platinum catalysts increase reaction rates at 300K."
        )
        print(f"Is the claim supported? {is_supported}")


    # Run the async function
    asyncio.run(main())