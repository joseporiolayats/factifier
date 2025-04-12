# factifier/decomposition.py
# Decomposition (DR-ND Method)
# Split sentences into atomic subclaims using Russellian/Neo-Davidsonian parsing

from typing import List, Dict
import spacy

__all__ = ["DRNDDecomposer"]


class Decomposer:
    """
    Decomposer class for splitting sentences into atomic subclaims
    """

    pass


class DRNDDecomposer(Decomposer):
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the decomposer with a spaCy model for dependency parsing.

        Args:
            spacy_model (str): The name of the spaCy model to use for dependency parsing.
        """
        # For spacy model download use
        # python -m spacy download en_core_web_sm
        self.nlp = spacy.load(spacy_model)

    def _dependency_parsing(self, sentence: str) -> List[Dict[str, str | None]]:
        """
        Perform dependency parsing on a sentence using spaCy.

        Args:
            sentence (str): The sentence to parse.

        Returns:
            List[Dict[str, str | None]]: A list of dependencies, where each dependency is represented
                                  as a dictionary with keys "predicate", "subject", and "object".
        """
        doc = self.nlp(sentence)
        dependencies: List[Dict[str, str | None]] = []

        for token in doc:
            if token.dep_ in ("ROOT", "acl", "relcl"):  # Focus on predicates
                predicate = token.text
                subject = next(
                    (
                        child.text
                        for child in token.children
                        if child.dep_ in ("nsubj", "nsubjpass")
                    ),
                    None,
                )
                obj = next(
                    (
                        child.text
                        for child in token.children
                        if child.dep_ in ("dobj", "attr", "prep")
                    ),
                    None,
                )

                if subject or obj:  # Only include meaningful dependencies
                    dependencies.append(
                        {"predicate": predicate, "subject": subject, "object": obj}
                    )

        return dependencies

    def _logical_form_generation(
        self, dependencies: List[Dict[str, str | None]]
    ) -> List[str]:
        """
        Map dependencies to Neo-Davidsonian event semantics.

        Args:
            dependencies (List[Dict[str, str]]): A list of dependencies from dependency parsing.

        Returns:
            List[str]: A list of logical forms in Neo-Davidsonian style.
        """
        logical_forms = []
        for dep in dependencies:
            predicate = dep["predicate"]
            subject = dep["subject"]
            obj = dep["object"]
            if subject and obj:
                logical_forms.append(f"{predicate}({subject}, {obj})")
            elif subject:
                logical_forms.append(f"{predicate}({subject})")
            elif obj:
                logical_forms.append(f"{predicate}({obj})")
        return logical_forms

    def _atomic_fact_extraction(self, logical_forms: List[str]) -> List[str]:
        """
        Isolate predicates and arguments as standalone claims.

        Args:
            logical_forms (List[str]): A list of logical forms in Neo-Davidsonian style.

        Returns:
            List[str]: A list of atomic subclaims.
        """
        return logical_forms  # In this case, logical forms are already atomic

    def decompose(self, sentence: str) -> List[str]:
        """
        Decompose a sentence into atomic subclaims using DR-ND method (synchronous).

        Args:
            sentence (str): The sentence to decompose.

        Returns:
            List[str]: A list of atomic subclaims.
        """
        # Step A: Dependency Parsing
        dependencies = self._dependency_parsing(sentence)
        # Step B: Logical Form Generation
        logical_forms = self._logical_form_generation(dependencies)
        return self._atomic_fact_extraction(logical_forms)

    async def decompose_async(self, sentence: str) -> List[str]:
        """
        Decompose a sentence into atomic subclaims using DR-ND method (asynchronous).

        Args:
            sentence (str): The sentence to decompose.

        Returns:
            List[str]: A list of atomic subclaims.
        """
        # For async, we can use the same logic as the synchronous version
        # since spaCy itself is not async. However, this method is provided
        # for consistency with LangChain's async ecosystem.
        return self.decompose(sentence)


if __name__ == "__main__":
    import asyncio

    # Initialize the decomposer
    decomposer = DRNDDecomposer()

    # Decompose a sentence asynchronously
    async def main():
        sentence = "Tarantino directed Pulp Fiction and it won an award."
        atomic_facts = await decomposer.decompose_async(sentence)
        print(atomic_facts)

    # Run the async function
    asyncio.run(main())
