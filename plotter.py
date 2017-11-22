import numpy as np
import matplotlib.pyplot as plt


def plot_heat_matrix(context, query, data, fig_size=(15, 55), flip=False):
    print('context', len(context))
    print('query', len(query))
    print('data', data.shape)
    data = data[:len(context), :len(query)]
    print(data)
    '''
        data: (x, y) # (context_len, query_len)
    '''

    # https://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.8)

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
    plt.show()
