import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_clusters(df, model):
    centers = model.cluster_centers_

    plt.figure(figsize=(8, 6))

    colors = ['red', 'orange', 'green']
    labels = ['Low Performance', 'Medium Performance', 'High Performance']

    for i in range(3):
        plt.scatter(
            df[df["cluster"] == i]["attendance"],
            df[df["cluster"] == i]["final_score"],
            label=labels[i],
            alpha=0.7
        )

    # 聚类中心
    plt.scatter(
        centers[:, 0],
        centers[:, 2],
        s=300,
        marker='X',
        edgecolors='black',
        linewidths=2,
        label='Centroids'
    )

    plt.xlabel("Attendance (%)")
    plt.ylabel("Final Score (%)")
    plt.title("Clustering Results of Student Performance")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/figure1.png")
    plt.show()


def plot_pca(df):
    X = df[["attendance", "assignment", "final_score"]]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    colors = ['red', 'orange', 'green']
    labels = ['Low Performance', 'Medium Performance', 'High Performance']

    for i in range(3):
        plt.scatter(
            X_pca[df["cluster"] == i, 0],
            X_pca[df["cluster"] == i, 1],
            label=labels[i],
            alpha=0.7
        )

    plt.title("PCA Visualization of Student Clusters")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/figure2_pca.png")
    plt.show()