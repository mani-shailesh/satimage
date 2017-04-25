import matplotlib.cm as cm
import numpy as np
import numpy.ma as ma
import pylab as pl

from keras.models import model_from_json
from mpl_toolkits.axes_grid1 import make_axes_locatable


def restore_model(json_filepath, weights_filepath):
    """
    Restore the pretrained model.
    :param json_filepath: Path of the .json file containing model architecture
    :param weights_filepath: Path of the .h5 file containing weights of pretrained model
    :return `model` object
    """
    print("Loading model from disk...")
    json_file = open(json_filepath, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_filepath)
    print("Done")
    return model


# -----------------------------------------------------------------------------
# https://github.com/julienr/
# ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic
