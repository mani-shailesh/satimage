import os
import keras.backend as K
import numpy as np
import pandas as pd
import math
import matplotlib.cm as cm
import numpy.ma as ma
import pylab as pl

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import misc
from keras.models import model_from_json

BATCH_SIZE = 20

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


def img_data_generator(file_paths, batch_size):
    """
    Data generator for the model.
    :param file_paths: List of paths for the images
    :param batch_size: Batch size to be used for prediction
    """
    while True:
        x_train = []
        for file_path in file_paths:
            img = misc.imread(file_path)
            x_train.append(img)
            if len(x_train) == batch_size:
                x_to_yield = np.array(x_train, dtype=np.float32)
                if K.image_dim_ordering() == "th":
                    x_to_yield = x_to_yield.transpose((0, 3, 1, 2))
                yield x_to_yield
                x_train = []
        if len(x_train) > 0:
            x_to_yield = np.array(x_train, dtype=np.float32)
            if K.image_dim_ordering() == "th":
                x_to_yield = x_to_yield.transpose((0, 3, 1, 2))
            yield x_to_yield
                

def generate_predictions(model, img_dir, out_filepath, batch_size=BATCH_SIZE):
    """
    Generate predictions for the model and save them to the specified path.
    :param model: The pretrained model object
    :param img_dir: The directory containing images that are to be fed to the model
    :param out_filepath: File path to write the predictions
    :param batch_size: Batch size to be used for generating predictions
    """
    file_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    steps = int(len(file_paths) / batch_size)
    if len(file_paths) % batch_size > 0:
        steps += 1
    data_generator_obj = img_data_generator(file_paths, batch_size)
    
    print("Generating predictions...")
    predictions = model.predict_generator(data_generator_obj,
                                          val_samples=steps * batch_size,
                                          pickle_safe=True)
    
    pd_dict = dict()
    order = ['region_code']
    pd_dict['region_code'] = [os.path.split(f)[1].split('.')[0] for f in file_paths]
    for ii in range(len(predictions)):
        predictions[ii] = np.array(predictions[ii], dtype=np.float32)

        for idx in range(predictions[ii].shape[-1]):
            pd_dict[str(ii) + "_" + str(idx)] = np.transpose(predictions[ii])[idx]
            order.append(str(ii) + "_" + str(idx))
            
    compare = pd.DataFrame(data=pd_dict)
    
    compare = compare[order]
    compare.to_csv(out_filepath, index=False)
    print("Done")

    
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
    
    
def show_filter_responses(model, layer_index, input_img_path, save_dir=None, filter_index=None, dpi=100.0, save_original=False):
    """
    Show and save the filter responses for all or a selected filter at given layer.
    :param model: pre-trained model object
    :param layer_index: index of the layer
    :param input_img_path: path of the input image
    :param save_dir: path of directory to save the filters/original image. Filters are only displayed but not saved if this is None
    :param filter_index: index of the filter in the given layer. All filter responses are displayed if this is None
    :param dpi: DPI of the display
    """
    input_img = np.array(misc.imread(input_img_path), dtype=np.float32)
    if K.image_dim_ordering() == "th":
        input_img = input_img.transpose((2, 0, 1))
    
    layer = model.layers[layer_index]
    inputs = [K.learning_phase()] + model.inputs
    _layer_f = K.function(inputs, [layer.output])

    def layer_f(X):
        # The [0] is to disable the training phase flag
        return _layer_f([0] + [X])

    if K.image_dim_ordering() == "th":
        display_image = input_img.transpose((1, 2, 0))
    else:
        display_image = np.copy(input_img)
    display_image = deprocess_image(display_image, alter_dim=False)

    pl.figure(
        figsize=(display_image.shape[0] / dpi, display_image.shape[1] / dpi),
        dpi=dpi
    )
    nice_imshow(pl.gca(), display_image, cmap=cm.binary)

    images = np.array([input_img])
    c1 = layer_f(images)
    c1 = np.squeeze(c1)

    if K.image_dim_ordering() == "tf":
        c1 = c1.transpose((2, 0, 1))

    print("c1 shape : ", c1.shape)

    if filter_index is None:
        grid_dim = int(math.ceil(math.sqrt(c1.shape[0])))
        out_img = make_mosaic(c1, grid_dim, grid_dim)
    else:
        out_img = c1[filter_index]
    
    if save_dir is not None:
        prefix = "layer_" + str(layer_index)
        if filter_index is not None:
            prefix += "_filter_" + str(filter_index)
        if save_original:
            misc.imsave(os.path.join(save_dir, prefix + "_input.png"), display_image)
        misc.imsave(os.path.join(save_dir, prefix + "_output.png"), deprocess_image(out_img, alter_dim=False))

    pl.figure(
        figsize=(out_img.shape[0] / dpi, out_img.shape[1] / dpi),
        dpi=dpi
    )
    pl.suptitle(layer.name)
    nice_imshow(pl.gca(), out_img, cmap=cm.binary)

    
# -----------------------------------------------------------------------------

# https://blog.keras.io/
# how-convolutional-neural-networks-see-the-world.html

# http://ankivil.com/visualizing-deep-neural-networks-classes-and-features/

# -----------------------------------------------------------------------------
def deprocess_image(x, alter_dim=True):
    """
    Utility function to convert a tensor into a valid image
    """
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if alter_dim and K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
