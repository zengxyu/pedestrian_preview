import numpy as np
from matplotlib import pyplot as plt


json_filename_template = "c_{}+s_{}_speed_{}.json"


def plot_two_stack_bar_chart(
        colors,
        x_names,
        values,
        values2,
        legend_labels,
        ax,
        title=None,
        show_extra_info=True,
        legend_loc="lower right",
        save_path=None,
        width=0.5,
        fontsize=18,
        xlabel=None,
        ylabel=None,
        round_digit=1,
        top_names=None
):
    def auto_fill_text(starts, heights, offset, i):
        xcenters = starts + heights / 2
        for x, (y, c) in enumerate(zip(xcenters, heights)):
            if c > 5:
                ax.text(
                    x + offset,
                    y,
                    str(round(c, round_digit)),
                    ha="center",
                    va="center",
                    color="black",
                    rotation=0,
                    fontsize=fontsize - 1,
                )
        for x, _ in enumerate(xcenters):
            if top_names is not None:
                ax.text(
                    x + offset,
                    103,
                    top_names[i],
                    ha="center",
                    va="center",
                    color="black",
                    rotation=0,
                    fontsize=fontsize,
                )

    r1 = np.arange(len(x_names))
    offset = width * 1.1
    r2 = [x + offset for x in r1]

    data_cum = values.cumsum(axis=1)
    data_cum2 = values2.cumsum(axis=1)
    if show_extra_info:
        ax.yaxis.set_visible(True)
    else:
        ax.yaxis.set_visible(False)
    # ax.set_xticklabels(labels=x_names, rotation=0)
    ax.set_ylim(0, np.max(np.sum(values, axis=1)) + 8)

    for i, label in enumerate(legend_labels):
        heights = values[:, i]
        starts = data_cum[:, i] - heights

        heights2 = values2[:, i]
        starts2 = data_cum2[:, i] - heights2
        # color = colors[i] if isinstance(colors, list) else colors
        ax.bar(r1, heights, bottom=starts, width=width, label=label, color=colors[i])
        auto_fill_text(starts, heights, 0, 0)
        ax.bar(r2, heights2, bottom=starts2, width=width, color=colors[i])
        auto_fill_text(starts2, heights2, offset, 1)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    plt.xticks([r + width / 2 for r in range(len(x_names))], labels=x_names, size=fontsize)

    if show_extra_info:
        ax.legend(loc=legend_loc, fontsize=fontsize)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=500)
