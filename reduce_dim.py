from sklearn.decomposition import PCA
import itertools

def get_N_or_less_from_generator(stream, N):
    return list(itertools.islice(stream, N))

def reduce_dim(vectors, dim_output = 32, fit_size=100000):
    dim_input = vectors.shape[1]
    if dim_output >= dim_input:
        return vectors
    vectors_train = vectors[:fit_size]
    pca = PCA(n_components=dim_output)
    pca.fit(vectors_train)
    vectors_transformed = pca.transform(vectors)
    return vectors_transformed