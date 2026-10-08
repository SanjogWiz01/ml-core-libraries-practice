from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# K-Means 80/20 practice file
X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

if __file__.endswith('/1.py') or __file__.endswith('\\1.py'):
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    print(model.fit_predict(X)[:20])
elif __file__.endswith('/2.py') or __file__.endswith('\\2.py'):
    model = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    print('centers:', model.cluster_centers_)
    print('inertia:', model.inertia_)
elif __file__.endswith('/3.py') or __file__.endswith('\\3.py'):
    model = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    print(model.predict([[0, 0], [5, 5]]))
elif __file__.endswith('/4.py') or __file__.endswith('\\4.py'):
    model = KMeans(n_clusters=4, random_state=42, n_init=10); labels=model.fit_predict(X)
    plt.scatter(X[:,0],X[:,1],c=labels); plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],marker='X',s=200); plt.show()
elif __file__.endswith('/5.py') or __file__.endswith('\\5.py'):
    print([(k, KMeans(n_clusters=k,random_state=42,n_init=10).fit(X).inertia_) for k in range(1,9)])
elif __file__.endswith('/6.py') or __file__.endswith('\\6.py'):
    vals=[KMeans(n_clusters=k,random_state=42,n_init=10).fit(X).inertia_ for k in range(1,9)]; plt.plot(range(1,9),vals,marker='o'); plt.xlabel('K'); plt.ylabel('Inertia'); plt.show()
elif __file__.endswith('/7.py') or __file__.endswith('\\7.py'):
    scores={k:silhouette_score(X,KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X)) for k in range(2,9)}; print(scores); print('best K:',max(scores,key=scores.get))
elif __file__.endswith('/8.py') or __file__.endswith('\\8.py'):
    X2=np.column_stack([np.random.default_rng(42).normal(50000,10000,500),np.random.default_rng(1).normal(30,5,500)]); Z=StandardScaler().fit_transform(X2); print(np.bincount(KMeans(n_clusters=3,random_state=42,n_init=10).fit_predict(Z)))
elif __file__.endswith('/9.py') or __file__.endswith('\\9.py'):
    rng=np.random.default_rng(42); X2=np.column_stack([rng.normal(50000,10000,500),rng.normal(30,5,500)]); print('raw:',np.bincount(KMeans(n_clusters=3,random_state=42,n_init=10).fit_predict(X2))); print('scaled:',np.bincount(KMeans(n_clusters=3,random_state=42,n_init=10).fit_predict(StandardScaler().fit_transform(X2))))
elif __file__.endswith('/10.py') or __file__.endswith('\\10.py'):
    for k in range(2,7):
        m=KMeans(n_clusters=k,random_state=42,n_init=10); y=m.fit_predict(X); print(k,m.inertia_,silhouette_score(X,y))
elif __file__.endswith('/11.py') or __file__.endswith('\\11.py'):
    a=KMeans(n_clusters=4,init='k-means++',n_init=10,random_state=42).fit_predict(X); b=KMeans(n_clusters=4,init='k-means++',n_init=10,random_state=42).fit_predict(X); print('reproducible:',np.array_equal(a,b))
elif __file__.endswith('/12.py') or __file__.endswith('\\12.py'):
    from sklearn.cluster import MiniBatchKMeans
    Xbig=np.random.default_rng(42).normal(size=(10000,10)); m=MiniBatchKMeans(n_clusters=5,batch_size=256,random_state=42,n_init=10); print(m.fit_predict(Xbig).shape)
elif __file__.endswith('/13.py') or __file__.endswith('\\13.py'):
    rng=np.random.default_rng(42); C=np.column_stack([rng.normal(50000,12000,500),rng.normal(35,8,500),rng.normal(30000,9000,500)]); Z=StandardScaler().fit_transform(C); y=KMeans(n_clusters=4,random_state=42,n_init=10).fit_predict(Z); print([(i, C[y==i].mean(axis=0).round(1)) for i in range(4)])
elif __file__.endswith('/14.py') or __file__.endswith('\\14.py'):
    m=KMeans(n_clusters=4,random_state=42,n_init=10); y=m.fit_predict(X); from sklearn.metrics import silhouette_samples; s=silhouette_samples(X,y); print('overall:',s.mean()); print([(i,s[y==i].mean()) for i in range(4)])
elif __file__.endswith('/15.py') or __file__.endswith('\\15.py'):
    Z=StandardScaler().fit_transform(X); scores={k:silhouette_score(Z,KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(Z)) for k in range(2,9)}; k=max(scores,key=scores.get); m=KMeans(n_clusters=k,random_state=42,n_init=10); y=m.fit_predict(Z); print('best K:',k,'silhouette:',scores[k],'counts:',np.bincount(y))
