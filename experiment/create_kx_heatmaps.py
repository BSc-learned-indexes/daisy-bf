import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', action="store", dest="data_path", type=str, required=True, help="name of the dataset")

parser.add_argument('--is_daisy', action="store", dest="is_daisy", type=bool, required=False, help="True if the file is from the daisy bloom filter", default=False)


def heatmap(data, row_labels, col_labels, x_ax_label="", y_ax_label="", ax=None,
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
    ax.set_xlabel(x_ax_label)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(y_ax_label)

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

args = parser.parse_args()
FILE = args.data_path
IS_DAISY = args.is_daisy
data = pd.read_csv(f'./data/plots/{FILE}.csv', index_col=0)
rows = data["FPR_actual"].apply(lambda x: "{:.2E}".format(x))
if IS_DAISY:
    matrix = data.drop(["size", "FPR_target", "FPR_actual"], axis=1)
else:
    matrix = data.drop(["size", "FPR_actual"], axis=1)
cols = matrix.columns.astype(int)
sorter = np.argsort(cols)
cols = cols[sorter]
matrix = matrix[cols.astype(str)]
num_elements = matrix.iloc[0].sum()
matrix = matrix * 100 / num_elements
matrix = matrix.round(2)
fig, ax = plt.subplots(figsize=(15,15))
if IS_DAISY:
    im, cbar = heatmap(matrix, rows, cols, "Number of Hash Functions", "False Positive Rate", ax=ax, cmap="Wistia", cbarlabel="% of elements")
else:
    im, cbar = heatmap(matrix, rows, cols, "Region", "False Positive Rate", ax=ax, cmap="Wistia", cbarlabel="% of elements")


texts = annotate_heatmap(im, valfmt="{x:.1f}")

fig.tight_layout()
plt.savefig(f'./distributions/img/heatmaps/{FILE}.png')
