import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")
plt.style.use("ggplot")


####### DIFERENTES GRÁFICOS ÚTILES################

def boxplot_describe(df_plot, col_x="weekday", col_y="speed", n_variables=True):
    """
    Plot a boxplot and returns a describe of the variable.
    Args: a dataframe, a numerical column (col_y), a factor column (col_x)
    and a n_variables argument.

    Returns: If n_variables is set as true, several boxplots are depicted according to the values
    of col_x and a dataframe with the describe of the col_y column is given grouped by the col_x values
    If False, only a boxplot is plotted with the numerical values

    """

    plt.figure(figsize=(15, 8))

    if n_variables:
        if len(df_plot[col_x].value_counts().index) > 4:
            # df = px.data.tips()
            fig = px.box(df_plot, x=col_x, y=col_y)
        else:

            fig = px.box(df_plot, x=col_x, y=col_y, color=col_x)

        fig.show()

        # ax = sns.boxplot(x=col_x, y=col_y,data=df_plot)

        return df_plot[[col_y, col_x]].groupby(col_x).describe().T
    else:
        fig = px.box(df_plot, y=col_y)
        fig.show()


def draw_silhouette(df2_k, columns_to_k, range_n_clusters=[2, 3, 4, 5, 6, 7, 8, 9, 10], fig_size=(15, 8)):
    """
    This function draws a plot with the silhouette coefficiente for every sample grouped by cluster
    Args: a dataframe, list with columns of the independent variables to cluster, a list with the number of cluters to compare
    and the figsize.
    Returns. A plot with every cluster calculated

    """

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(figsize=fig_size)
        # ax1=plt.plot()

        # fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(df2_k[columns_to_k]) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(df2_k[columns_to_k])

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(df2_k[columns_to_k], cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(df2_k[columns_to_k], cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()


def elbow_plot(df2_k, columns_to_k, range_n_clusters=[2, 3, 4, 5, 6, 7, 8, 9, 10], fig_size=(15, 8)):
    """
    A function to plot the elbow.inertia plot for clustering,
    Args: a dataframe, list with columns of the independent variables to cluster, a list with the number of cluters to compare
    and the figsize.
    Returns. A plot of the elbow function

    """

    sse = {}
    plt.figure(figsize=fig_size)
    for k in range_n_clusters:
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df2_k[columns_to_k])

        # df2_kmeans["clusters"] = kmeans.labels_

        # print(df2_kmeans["clusters"])

        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center

    # plt.figure()

    df_I = pd.DataFrame.from_dict(sse, orient='index', columns=["Inertia"]).reset_index()
    df_I.columns = ["Number of clusters", "Inertia"]

    df_I.sort_values(by="Inertia", ascending=False, inplace=True)

    fig = px.line(df_I, y="Inertia", x="Number of clusters", title='Optimal number of clusters')
    fig.show()


def draw_roc(model, X, y, titulo="Roc curve", figsize=(12, 8)):
    """
    Function to plot the ROC Curve
    ARgs: A machine learning model, a dataframe with the independent variables (X), a series or colection with the target
    variable (y), the graph title and figsize
    Returns:  A plot with the roc curve


    """

    y_test_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, umbrales = roc_curve(y_true=y
                                   ,
                                   y_score=y_test_proba)  # Aquí se extraen los ratios fpr y tpr que se pintarán después.
    plt.figure(figsize=figsize)
    plt.plot([0, 1], [0, 1], "r--")
    plt.plot(fpr, tpr, "b")
    plt.fill_between(fpr, tpr, alpha=0.4, color="yellow")
    # plt.title(titulo)
    plt.show()
    # return roc_auc_score(y_true=y,y_score=y_test_proba) ##bonus
