# factifier/core.py
# Core Filtering
# Remove redundant claims using semantic similarity

from typing import List
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings
from sklearn.cluster import DBSCAN
import numpy as np

__all__ = ["CoreFilter"]

class CoreFilter:
    def __init__(self, embeddings: Embeddings, eps: float = 0.3, min_samples: int = 1):
        """
        Initialize the core filter with a LangChain-compatible embeddings model.

        Args:
            embeddings (Embeddings): A LangChain-compatible embeddings model.
            eps (float): The maximum distance between two samples for them to be considered
                        as in the same neighborhood (DBSCAN parameter).
            min_samples (int): The number of samples in a neighborhood for a point to be
                              considered as a core point (DBSCAN parameter).
        """
        self.embeddings = embeddings
        self.eps = eps
        self.min_samples = min_samples

    def filter(self, subclaims: List[str]) -> List[str]:
        """
        Remove redundant claims using semantic similarity (synchronous).

        Args:
            subclaims (List[str]): A list of subclaims to filter.

        Returns:
            List[str]: A list of unique subclaims after filtering.
        """
        # Generate embeddings for the subclaims
        embeddings = self.embeddings.embed_documents(subclaims)
        # Perform clustering using DBSCAN
        clusters = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(embeddings)
        # Get unique cluster indices
        unique_indices = list(set(clusters))
        # Return unique subclaims based on cluster indices
        return [subclaims[i] for i in unique_indices]

    async def filter_async(self, subclaims: List[str]) -> List[str]:
        """
        Remove redundant claims using semantic similarity (asynchronous).

        Args:
            subclaims (List[str]): A list of subclaims to filter.

        Returns:
            List[str]: A list of unique subclaims after filtering.
        """
        # Generate embeddings for the subclaims asynchronously
        embeddings = await self.embeddings.aembed_documents(subclaims)
        # Perform clustering using DBSCAN
        clusters = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(embeddings)
        # Get unique cluster indices
        unique_indices = list(set(clusters))
        # Return unique subclaims based on cluster indices
        return [subclaims[i] for i in unique_indices]