# Adapted from: https://matplotlib.org/examples/widgets/lasso_selector_demo.html

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

import os, sys

from scipy.integrate import trapz
from skimage.io import imread
from six.moves import input


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool highlights
    selected points by fading them out (i.e., reducing their alpha values).
    If your collection has alpha < 1, this tool will permanently alter them.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.01):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, self.Npts).reshape(self.Npts, -1)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':

    plt.ion()
    file_path = input("The path of the image: ")
    img = imread(file_path)
    y, x = img.shape[0], img.shape[1]
    data = np.mgrid[0:x, 0:y]
    selected_pts = []

    fig, ax = plt.subplots()

    ax.imshow(img)
    pts = ax.scatter(data[0], data[1], s = 1, alpha = 0.1)
    plt.show()

    while(True):
        
        flag = input("Press Y to begin selection, press N to exit: ")
        if flag == "N":
            break

        selector = SelectFromCollection(ax, pts)
        input('Press Enter to accept selected points')

        print("Selected points:")
        selected_pts.append(selector.xys[selector.ind])
        print(selector.xys[selector.ind].shape)
        selector.disconnect()

    mask = np.vstack(selected_pts).astype(int)
    print("Total selected points: ", mask.shape[0])
    img_mask = np.zeros((y, x))
    img_mask[mask[:, 1], mask[:, 0]] = 1

    out_path = input("The path of the mask: ")
    np.save(out_path, img_mask)
    # Block end of script so you can check that the lasso is disconnected.
    input('Press Enter to quit')