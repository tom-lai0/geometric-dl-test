import matplotlib.pyplot as plt


def plot_3d_scatter(d, dir):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        d[:, 0],
        d[:, 1],
        d[:, 2],
        c = d[:, dir]
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.xlim(-1.4, 1.4)
    plt.ylim(-1.4, 1.4)
    fig.show()


def save_3d_project_fig(f, ax, d, dir, path):
    ax1, ax2, ax3 = ax
    ax1.cla(); ax2.cla(); ax3.cla()

    ax1.scatter(d[:, 0], d[:, 1], c=dir[:, 2], marker='.')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('z')

    ax2.scatter(d[:, 0], d[:, 2], c=dir[:, 1], marker='.')
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    ax2.set_title('y')

    ax3.scatter(d[:, 1], d[:, 2], c=dir[:, 0], marker='.')
    ax3.set_xlabel('y')
    ax3.set_ylabel('z')
    ax3.set_title('x')
    f.savefig(path)
    ax1.cla(); ax2.cla(); ax3.cla()
