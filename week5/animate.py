# import packages

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# for binding arguments to functions
from functools import partial

# ==============================================================================

# initialize graphing

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'r')

def init(xlim_1, ylim_1, xlim_2, ylim_2):
    """
    Initialize axes for plotting by setting X and Y limits
    """
    ax.set_xlim(xlim_1, xlim_2)
    ax.set_ylim(ylim_1, ylim_2)
    return ln,

# ==============================================================================

# convenience function for generating polygons

def genpoly(vertices, n):
    """
    Generate a polygon with `n` points using the co-ordinates given, and return
    the points in it.
    """

    lists = []

    n_vert = vertices.shape[0]
    points_per_vertex = n // n_vert

    t = np.linspace(0, 1, points_per_vertex + 1)[:-1]
    ones = np.ones(points_per_vertex)

    for i in range(n_vert - 1):
        lists.append(np.column_stack(
            [
                t * vertices[i + 1][j] + (ones - t) * vertices[i][j]
                for j in [0, 1]
            ]
        ))

    # put all the leftover points in the final edge
    t = np.linspace(0, 1, n - points_per_vertex * (n_vert - 1))
    ones = np.ones(n - points_per_vertex * (n_vert - 1))

    lists.append(np.column_stack(
        [
            t * vertices[0][j] + (ones - t) * vertices[n_vert - 1][j]
            for j in [0, 1]
        ]
    ))

    return np.concatenate(lists)

# ==============================================================================

# linear interpolation morphing

def lerp_morph():
    total_frames = 1600
    polygon_points = 840
    low_poly = 3
    high_poly = 8

    def update(polygons, num_pause, frame):
        """
        Linear interpolate between polygons. Pause for `num_pause` between the
        forward and backward animations.
        """
        i = int(np.floor(frame))

        # lists of points to interpolate between
        x1 = None
        x2 = None
        y1 = None
        y2 = None

        if i < len(polygons) - 1:
            # forward animation
            x1 = polygons[i][:, 0]
            y1 = polygons[i][:, 1]

            x2 = polygons[i + 1][:, 0]
            y2 = polygons[i + 1][:, 1]

        elif i < len(polygons) - 1 + num_pause:
            last = polygons[-1]
            ln.set_data(last[:, 0], last[:, 1])

            # pause
            return ln,

        elif i < 2 * len(polygons) - 2 + num_pause:
            # backward animation
            x1 = polygons[2 * len(polygons) - i - 2 + num_pause][:, 0]
            y1 = polygons[2 * len(polygons) - i - 2 + num_pause][:, 1]

            x2 = polygons[2 * len(polygons) - i - 3 + num_pause][:, 0]
            y2 = polygons[2 * len(polygons) - i - 3 + num_pause][:, 1]

        else:
            # pause again
            return ln,

        j = frame - i

        xdata, ydata = x2 * j + x1 * (1 - j), y2 * j + y1 * (1 - j)
        ln.set_data(xdata, ydata)
        return ln,

    polygons = []

    for i in range(low_poly, high_poly + 1):
        points = 2 * np.pi * np.arange(0, i) / i

        polygons.append(genpoly(
            np.column_stack([
                np.cos(points),
                np.sin(points)
            ]),
            polygon_points
        ))

    ani = FuncAnimation(fig,
                        partial(update, polygons, 2),
                        frames=np.linspace(0, 2 * (high_poly - low_poly + 2), total_frames)[:-1],
                        init_func=partial(init, -1.5, -1.5, 1.5, 1.5),
                        blit=True,
                        interval=5,
                        repeat=True)
    plt.show()

# ==============================================================================

# splitting points and morphing
def get_intermediate_vertices(vert1, vert2, step):
    new_verts = []

    for i in range(len(vert1)):
        new_verts.append(
            np.array([
                step * vert2[i][j] + (1 - step) * vert1[i][j]
                for j in [0, 1]
            ])
        )

        new_verts.append(
            np.array([
                step * vert2[i + 1][j] + (1 - step) * vert1[i][j]
                for j in [0, 1]
            ])
        )

    return np.array(new_verts)

def split_morph():
    total_frames = 1600
    low_poly = 3
    high_poly = 8
    num_pause = 2

    def update(polygons, num_pause, frame):
        """
        Create intermediate polygons by splitting points.

        num_pause is the amount of time paused between forward/backward cycles
        """
        i = int(np.floor(frame))
        num_poly = len(polygons)

        if i < num_poly - 1:
            # forward animation
            vert = get_intermediate_vertices(polygons[i], polygons[i + 1], frame - i)

        elif i < num_poly + num_pause - 1:
            vert = polygons[num_poly - 1]

        elif i < 2 * num_poly + num_pause - 2:
            # forward animation
            j = 2 * num_poly - i + num_pause - 2

            vert = get_intermediate_vertices(polygons[j - 1], polygons[j], 1 - (frame - i))

        else:
            # pause again
            vert = polygons[0]

        vert = np.concatenate([vert, [vert[0]]])
        ln.set_data(vert[:, 0], vert[:, 1])
        return ln,

    polygons = []

    for i in range(low_poly, high_poly + 1):
        # add a regular polygon to the list of polygons
        points = 2 * np.pi * np.arange(0, i) / i

        polygons.append(
            np.column_stack([
                np.cos(points),
                np.sin(points)
            ])
        )

    ani = FuncAnimation(fig,
                        partial(update, polygons, num_pause),
                        frames=np.linspace(0, 2 * (high_poly - low_poly + num_pause),
                                        total_frames),
                        init_func=partial(init, -1.5, -1.5, 1.5, 1.5),
                        blit=True,
                        interval=5,
                        repeat=True)
    plt.show()


choice = input("""Enter which animation you want to play:
1 - simple morphing
2 - given animation on moodle\n""")

if choice == "1":
    lerp_morph()
elif choice == "2":
    split_morph()
else:
    print("invalid choice")
