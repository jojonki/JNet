import numpy as np
import matplotlib.pyplot as plt


def plot_heat_matrix(context, query, attn_data, ans_pair, fig_size=[15, 55], output_file='attention_matrix.pdf', scale=10.0):
    '''
        data: (x, y) # (context_len, query_len)
    '''

    data = attn_data.cpu().data.numpy()
    data = data[:len(context), :len(query)]
    data *= scale # for vivid color
    # ans_pair = [ans_pair[0].cpu().data[0], ans_pair[1].cpu().data[0]]

    fig_size[1] = max(fig_size[1], int(fig_size[1] * len(context)/200)) # TODO Hard code

    # https://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
    fig, ax = plt.subplots()
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
    ax.xaxis.tick_top()

    # Set the labels
    ax.set_xticklabels(query, minor=False)
    ax.set_yticklabels(context, minor=False)
    plt.xticks(rotation=90)

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
    [t.set_color('red') for (i, t) in enumerate(ax.yaxis.get_ticklabels()) if i>=ans_pair[0] and i<ans_pair[1]]

    # plt.show()

    print('Save attention figure', output_file)
    plt.savefig(output_file)
