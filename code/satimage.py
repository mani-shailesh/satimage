import os
import keras.backend as K
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.cm as cm
import math

from scipy import misc

import util

BATCH_SIZE = 20


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
    display_image = util.deprocess_image(display_image, alter_dim=False)

    pl.figure(
        figsize=(display_image.shape[0] / dpi, display_image.shape[1] / dpi),
        dpi=dpi
    )
    util.nice_imshow(pl.gca(), display_image, cmap=cm.binary)

    images = np.array([input_img])
    c1 = layer_f(images)
    c1 = np.squeeze(c1)

    if K.image_dim_ordering() == "tf":
        c1 = c1.transpose((2, 0, 1))

    print("c1 shape : ", c1.shape)

    if filter_index is None:
        grid_dim = int(math.ceil(math.sqrt(c1.shape[0])))
        out_img = util.make_mosaic(c1, grid_dim, grid_dim)
    else:
        out_img = c1[filter_index]
    
    if save_dir is not None:
        prefix = "layer_" + str(layer_index)
        if filter_index is not None:
            prefix += "_filter_" + str(filter_index)
        if save_original:
            misc.imsave(os.path.join(save_dir, prefix + "_input.png"), display_image)
        misc.imsave(os.path.join(save_dir, prefix + "_output.png"), util.deprocess_image(out_img, alter_dim=False))

    pl.figure(
        figsize=(out_img.shape[0] / dpi, out_img.shape[1] / dpi),
        dpi=dpi
    )
    pl.suptitle(layer.name)
    util.nice_imshow(pl.gca(), out_img, cmap=cm.binary)

