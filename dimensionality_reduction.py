import faiss

def reduce_pca(embeds, dim_in, dim_out):
    pca = faiss.PCAMatrix(dim_in, dim_out)
    pca.train(embeds)
    return pca, pca.apply(embeds)