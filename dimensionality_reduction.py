import faiss

def reduce_pca(embeds, dim_in, dim_out):
    pca = faiss.PCAMatrix(dim_in, dim_out)
    pca.train(embeds)
    return pca, pca.apply(embeds)

def reduce_naive(embeds, dim_in, dim_out):
    embeds_reduced = embeds[:, :dim_out]
    class NaiveReducer:
        def __init__(self, dim):
            self.dim = dim
        
        def apply(self, e):
            return e[:, :self.dim]
    
    return NaiveReducer(dim_out), embeds_reduced
