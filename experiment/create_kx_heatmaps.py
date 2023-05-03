import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    ax.set_xlabel("Hash Functions")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("False Positive Rate")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, shrink=0.5)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

file = "daisy-BF_k_insert"
# file = "daisy-BF_k_lookup"
data = pd.read_csv(f'./data/plots/{file}.csv')
# data = data.reindex(sorted(data.columns), axis=1)
print(data.head())
data = pd.read_csv(f'./data/plots/{file}.csv', index_col=0)
# data = data.reindex(sorted(data.columns), axis=1)
print(data.head())

rows = data["FPR_actual"]
print(rows)
rows = data["FPR_actual"].apply(lambda x: "{:.1E}".format(x))
# rows = data["FPR_actual"].apply(lambda x: "{:.1f}*10^{}".format(*("{:.1e}".format(x).split('e'))))
print(rows)
# print(rows)

print(f"rows: {len(rows)}")

kx_data = data.drop(["size", "FPR_target", "FPR_actual"], axis=1)
cols = kx_data.columns.astype(int)
sorter = np.argsort(cols)
cols = cols[sorter]
print(cols)
kx_data = kx_data[cols.astype(str)]
# data.columns = cols.astype(str)
print(kx_data.head())
# kx_data.fillna(0, inplace=True)
num_elements = kx_data.iloc[0].sum()
# print(kx_data.head())
# kx_data.to_numpy()
# kx_data = np.nan_to_num(kx_data)
kx_data = kx_data * 100 / num_elements
# print(kx_data.head())
kx_data = kx_data.round(2)
print(kx_data.head())
# print(kx_data)
# cols = list(range(kx_data.shape[1]))
print(f"cols: {len(cols)}")
fig, ax = plt.subplots(figsize=(15,15))

im, cbar = heatmap(kx_data, rows, cols, ax=ax,
                   cmap="Wistia", cbarlabel="% of elements")

texts = annotate_heatmap(im, valfmt="{x:.1f}")

fig.tight_layout()
plt.savefig(f'./distributions/img/heatmaps/{file}.png')
