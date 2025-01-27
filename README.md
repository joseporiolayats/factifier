# Factifier
Coding the paper:

[DnDScore: Decontextualization and Decomposition for Factuality Verification in Long-Form Text Generation](https://arxiv.org/abs/2412.13175)


**Factifier** is a robust, modular pipeline for decomposing, verifying, and filtering factual claims from text documents. It combines state-of-the-art natural language processing techniques to extract atomic subclaims, resolve ambiguities, verify claims against reference documents, and remove redundancies. Designed for flexibility and scalability, Factifier is ideal for applications in fact-checking, knowledge extraction, and document analysis.

---

## Features

- **Decomposition**: Splits sentences into atomic subclaims using dependency parsing and Neo-Davidsonian semantics.
- **Decontextualization**: Resolves ambiguities in subclaims by leveraging document context.
- **Verification**: Checks subclaims against reference documents for factual accuracy.
- **Filtering**: Removes redundant subclaims using semantic similarity clustering.
- **Async Support**: Provides both synchronous and asynchronous methods for efficient processing.
- **Model Agnostic**: Works with any LangChain-compatible language models and embeddings.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/joseporiolayats/factifier.git
   cd factifier
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## Usage

### Quick Start

```python
from factifier import Factifier, DRNDDecomposer, MolecularFactsDecontextualizer, DnDScoreVerifier, CoreFilter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize components
decomposer = DRNDDecomposer()
llm = ChatOpenAI(model="gpt-4")
decontextualizer = MolecularFactsDecontextualizer(llm=llm)
verifier = DnDScoreVerifier(llm=llm)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
core_filter = CoreFilter(embeddings=embeddings)

# Initialize Factifier
factifier = Factifier(
    decomposer=decomposer,
    decontextualizer=decontextualizer,
    verifier=verifier,
    core_filter=core_filter,
)

# Run the pipeline
document = "Tarantino directed Pulp Fiction and it won an award."
reference = "Pulp Fiction, directed by Tarantino, won the Palme d'Or."
result = factifier.pipeline(document, reference)
print(result)
```

### Async Example

```python
import asyncio

async def main():
    result = await factifier.pipeline_async(document, reference)
    print(result)

asyncio.run(main())
```

---

## Pipeline Overview

1. **Decomposition**:
   - Splits sentences into atomic subclaims using dependency parsing and logical form generation.

2. **Decontextualization**:
   - Resolves ambiguities in subclaims by incorporating document context.

3. **Verification**:
   - Verifies subclaims against a reference document using a language model.

4. **Filtering**:
   - Removes redundant subclaims using semantic similarity clustering.

---

## Output

The pipeline returns a dictionary with two keys:
- **`score`**: A float representing the percentage of verified subclaims.
- **`verified_subclaims`**: A list of unique, verified subclaims.

Example output:
```python
{
    "score": 1.0,  # 100% of subclaims verified
    "verified_subclaims": [
        "direct(Tarantino, Pulp_Fiction)",
        "won(Pulp_Fiction, Palme_d'Or)"
    ]
}
```

---

## Customization

- **Language Models**: Use any LangChain-compatible LLM (e.g., OpenAI, Hugging Face).
- **Embeddings**: Swap out embeddings models (e.g., `all-MiniLM-L6-v2`, `SentenceTransformer`).
- **Clustering Parameters**: Adjust DBSCAN parameters (`eps`, `min_samples`) in `CoreFilter`.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **spaCy**: For dependency parsing.
- **LangChain**: For LLM and embeddings integration.
- **Hugging Face**: For pre-trained models and embeddings.

---

**Factifier** simplifies the process of extracting and verifying factual claims, making it a powerful tool for researchers, developers, and fact-checkers alike. Give it a try and let us know how it works for you! 🚀




