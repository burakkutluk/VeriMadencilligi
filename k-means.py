import numpy as np
import matplotlib.pyplot as plt

# Veri oluşturma
np.random.seed(42)
data = np.random.randn(200, 2) + np.array([2, 2])
data = np.vstack([data, np.random.randn(200, 2)])
data = np.vstack([data, np.random.randn(200, 2) + np.array([5, -1])])

# Veriyi görselleştirme
plt.scatter(data[:, 0], data[:, 1])
plt.title("Oluşturulan Veri Noktaları")
plt.show()


# K-Means algoritması
def k_means(data, k=3, max_iters=100):
    # Veriyi rastgele küme merkezleriyle başlatma
    centroids = data[np.random.choice(len(data), k, replace=False)]

    for _ in range(max_iters):
        # Her veri noktasını en yakın küme merkezine atama
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


labels, centroids = k_means(data, k=3)

# Sonuçları görselleştirme
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis")
plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, color="red")
plt.title("K-Means Kümeleme Sonuçları")
plt.show()
