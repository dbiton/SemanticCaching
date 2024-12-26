from sklearn.decomposition import PCA, FastICA, FactorAnalysis, TruncatedSVD
from sklearn.discriminant_analysis import StandardScaler
from sklearn.manifold import MDS
from sklearn.random_projection import GaussianRandomProjection

def autoscale(embdes):
    scaler = StandardScaler() 
    return scaler.fit_transform(embdes)

def reduce_pca(embeds, dim):
    pca = PCA(n_components=dim)
    return pca.fit_transform(autoscale(embeds))

def reduce_svd(embeds, dim):
    svd = TruncatedSVD(n_components=dim)
    return svd.fit_transform(autoscale(embeds))

def reduce_msd(embeds, dim):
    mds = MDS(n_components=dim)
    return mds.fit_transform(autoscale(embeds))

def reduce_msd_no_scale(embeds, dim):
    mds = MDS(n_components=dim)
    return mds.fit_transform(embeds)

def reduce_grp(embeds, dim):
    rp = GaussianRandomProjection(n_components=dim)
    return rp.fit_transform(autoscale(embeds))