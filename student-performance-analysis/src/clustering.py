from sklearn.cluster import KMeans

def run_kmeans(df):
    X = df[["attendance", "assignment", "final_score"]]

    model = KMeans(n_clusters=3, random_state=0)
    df["cluster"] = model.fit_predict(X)

    # 按成绩排序（更专业）
    order = df.groupby("cluster")["final_score"].mean().sort_values().index
    mapping = {order[i]: i for i in range(3)}
    df["cluster"] = df["cluster"].map(mapping)

    return df, model