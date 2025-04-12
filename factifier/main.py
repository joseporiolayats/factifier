# factifier/main.py
# Main Pipeline
# Combine all processes into a simple-to-use pipeline

from typing import Dict, List, Union, Type

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from nltk.tokenize import sent_tokenize  # type: ignore

from .decomposition import DRNDDecomposer
from .decontextualization import MolecularFactsDecontextualizer
from .verification import DnDScoreVerifier
from .core import CoreFilter


__all__ = ["Factifier"]


class Factifier:
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        decomposer: Type[DRNDDecomposer] = DRNDDecomposer,
        decontextualizer: Type[
            MolecularFactsDecontextualizer
        ] = MolecularFactsDecontextualizer,
        verifier: Type[DnDScoreVerifier] = DnDScoreVerifier,
        core_filter: Type[CoreFilter] = CoreFilter,
    ):
        """
        Initialize the Factifier with the required components.

        Args:
            llm (BaseChatModel): The language model for generating subclaims.
            embeddings (Embeddings): The embeddings model for vectorizing subclaims.
            decomposer (DRNDDecomposer): The decomposer for splitting sentences into subclaims.
            decontextualizer (MolecularFactsDecontextualizer): The decontextualizer for resolving ambiguities.
            verifier (DnDScoreVerifier): The verifier for checking subclaims against a reference.
            core_filter (CoreFilter): The filter for removing redundant subclaims.
        """
        self.llm = llm
        self.embeddings = embeddings
        self.decomposer = decomposer()
        self.decontextualizer = decontextualizer(llm=self.llm)
        self.verifier = verifier(llm=self.llm)
        self.core_filter = core_filter(embeddings=self.embeddings)

    def pipeline(
        self, document: str, reference: str
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Run the full pipeline synchronously.

        Args:
            document (str): The input document to process.
            reference (str): The reference document for verification.

        Returns:
            Dict[str, Union[float, List[str]]]: A dictionary containing the verification score
                                               and the list of verified subclaims.
        """
        # Step 1: Decompose
        sentences = sent_tokenize(document)
        subclaims = [self.decomposer.decompose(sent) for sent in sentences]
        flat_subclaims = [claim for sublist in subclaims for claim in sublist]

        # Step 2: Decontextualize
        decontext_subclaims = [
            self.decontextualizer.decontextualize(c, document) for c in flat_subclaims
        ]

        # Step 3: Verify
        verified = [
            self.verifier.verify(claim, decontexted, reference)
            for claim, decontexted in zip(flat_subclaims, decontext_subclaims)
        ]

        # Step 4: Filter
        unique_subclaims = self.core_filter.filter(flat_subclaims)

        # Compute score
        score = sum(verified) / len(verified) if verified else 0
        return {"score": score, "verified_subclaims": unique_subclaims}

    async def pipeline_async(
        self, document: str, reference: str
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Run the full pipeline asynchronously.

        Args:
            document (str): The input document to process.
            reference (str): The reference document for verification.

        Returns:
            Dict[str, Union[float, List[str]]]: A dictionary containing the verification score
                                               and the list of verified subclaims.
        """
        import asyncio

        # Step 1: Decompose
        sentences = sent_tokenize(document)
        subclaims = [self.decomposer.decompose(sent) for sent in sentences]
        flat_subclaims = [claim for sublist in subclaims for claim in sublist]

        # Step 2: Decontextualize
        decontext_subclaims = await asyncio.gather(
            *[
                self.decontextualizer.decontextualize_async(c, document)
                for c in flat_subclaims
            ]
        )

        # Step 3: Verify
        verified = await asyncio.gather(
            *[
                self.verifier.verify_async(claim, decontexted, reference)
                for claim, decontexted in zip(flat_subclaims, decontext_subclaims)
            ]
        )

        # Step 4: Filter
        unique_subclaims = await self.core_filter.filter_async(flat_subclaims)

        # Compute score
        score = sum(verified) / len(verified) if len(verified) > 0 else 0
        return {"score": score, "verified_subclaims": unique_subclaims}


# if __name__ == "__main__":
#     import asyncio
#     from langchain_community.embeddings import HuggingFaceEmbeddings
#     from langchain_openai import ChatOpenAI
#
#     # Initialize components (same as above)
#     llm = ChatOpenAI(model="gpt-4")
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#
#     # Initialize Factifier
#     factifier = Factifier(
#         llm=llm,
#         embeddings=embeddings
#     )
#
#     # Run the pipeline asynchronously
#     async def main():
#         document = "Tarantino directed Pulp Fiction and it won an award."
#         reference = "Pulp Fiction, directed by Tarantino, won the Palme d'Or."
#         result = await factifier.pipeline_async(document, reference)
#         print(result)
#
#     # Run the async function
#     asyncio.run(main())
