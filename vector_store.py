import faiss

class VectorStore:
    def __init__(self, dimension):
        """
        Initialize a FAISS vector store.

        Args:
            dimension (int): The dimension of the embeddings.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add(self, embeddings, texts):
        """
        Add embeddings and their corresponding texts to the store.

        Args:
            embeddings (np.ndarray): 2D array of embeddings.
            texts (list of str): The corresponding texts.
        """
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding, k=5):
        """
        Perform a similarity search.

        Args:
            query_embedding (np.ndarray): The embedding of the query.
            k (int): Number of nearest neighbors to return.

        Returns:
            list of tuples: List of (score, text) for the top-k matches.
        """
        distances, indices = self.index.search(query_embedding, k)
        texts = [self.texts[i[0]] for i in indices]
        embeddings = self.index.reconstruct_batch(indices.flatten())
        return texts, embeddings, distances, indices
