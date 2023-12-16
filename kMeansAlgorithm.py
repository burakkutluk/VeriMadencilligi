import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Veri oluşturma
data, _ = make_blobs(n_samples=200, centers=5, random_state=42)

# Veriyi görselleştirme
plt.scatter(data[:, 0], data[:, 1])
plt.title('Oluşturulan Veri Noktaları')
plt.show()

# K-Means algoritması
def k_means(data, k=5, max_iters=100):
    # Veriyi rastgele küme merkezleriyle başlatma
    centroids = data[np.random.choice(len(data), k, replace=False)]
    
    for _ in range(max_iters):
        # Her veri noktasını en yakın küme merkezine atama
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Yeni küme merkezlerini hesaplama
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Küme merkezlerini güncelleme
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

# K-Means modelini kullanma
labels, centroids = k_means(data, k=5)

# Sonuçları görselleştirme
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='red')
plt.title('K-Means Kümeleme Sonuçları')
plt.show()
