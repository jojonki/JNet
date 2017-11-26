import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_heat_matrix(context,
                     query,
                     attn_data,
                     ans_pair,
                     fig_size=[60, 15],
                     title='Attention Matrix',
                     output_file='attention_matrix.png',
                     pred=None):
    '''
        data: (x, y) # (context_len, query_len)
    '''
    label_fs = 60 # title, a and y label titles
    x_fs = 20
    y_fs = 30

    scale = 10.0
    data = attn_data.cpu().data.numpy()
    data = data[:len(context), :len(query)] # remove zero padding
    data = data.T
    data *= scale # for vivid color

    # https://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
    fig, ax = plt.subplots(facecolor='w')
    ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.8)

    # Format
    fig = plt.gcf()
    fig.set_size_inches(fig_size[0], fig_size[1]) # W, H

    # turn off the frame
    ax.set_frame_on(False)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    # ax.xaxis.tick_top()

    # Set the labels
    ax.set_xticklabels(context, minor=False, fontsize=x_fs)
    ax.set_yticklabels(query, minor=False, fontsize=y_fs)
    plt.xticks(rotation=90)
    ax.set_xlabel('Paragraph', fontsize=label_fs)
    ax.set_ylabel('Question', fontsize=label_fs)

    ax.grid(False)

    # Turn off all the ticks memory
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # change label color
    if pred:
        [t.set_color('deepskyblue') for (i, t) in enumerate(ax.xaxis.get_ticklabels()) if i == pred]
    [t.set_color('red') for (i, t) in enumerate(ax.xaxis.get_ticklabels()) if i >= ans_pair[0] and i <= ans_pair[1]]
    plt.title(title, fontsize=label_fs, y=1.02)
    plt.tight_layout()

    # plt.show()
    print('Save attention figure', output_file)
    plt.savefig(output_file)
    plt.close()
