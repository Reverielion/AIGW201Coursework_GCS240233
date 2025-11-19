import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class PCAGMM:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.pca = PCA(n_components=2, random_state=1)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        self.covariance_types = ['full', 'tied', 'diag', 'spherical']
        self.best_score = -2
        self.best_params = ""

        for n_components in range(4, 7):
            self.y_cat = pd.qcut(self.y.rank(method='first'), q=n_components, labels=[str(i) for i in range(1, n_components + 1)])
            for covariance_type in self.covariance_types:
                self.gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=1)
                self.cluster_labels = self.gmm.fit_predict(self.X_pca)
                self.silhouette = silhouette_score(self.X_pca, self.cluster_labels)
                if self.best_score < self.silhouette:
                    self.best_score = self.silhouette
                    plt.figure(figsize=(6, 5))
                    plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], c=self.cluster_labels, cmap="viridis", s=10)
                    plt.title(f"PCA+GMM Cluster Plot: {n_components} Clusters ({covariance_type})")
                    plt.xlabel("Principal Component 1")
                    plt.ylabel("Principal Component 2")
                    plt.colorbar(label="Cluster")
                    plt.savefig("bestplot.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    PCAGMM(r"C:\Users\Asus\PycharmProjects\CourseworkAI\Week 5 - housing_dataset.csv")
