from src.generate_data import generate_data
from src.clustering import run_kmeans
from src.visualization import plot_clusters, plot_pca

def main():
    # 1. 生成数据
    df = generate_data()

    # 2. 聚类分析
    df, model = run_kmeans(df)

    # 3. 可视化
    plot_clusters(df, model)
    plot_pca(df)

if __name__ == "__main__":
    main()