import os
from keras.models import model_from_json
import keras.backend as K

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
    steps = int(len(file_paths) / BATCH_SIZE)
    if len(file_paths) % BATCH_SIZE > 0:
        steps += 1
    data_generator_obj = img_data_generator(file_paths, batch_size)
    
    print("Generating predictions...")
    predictions = model.predict_generator(data_generator_obj,
                                          steps=steps,
                                          pickle_safe=True,
                                          verbose=1)
    
    pd_dict = dict()
    pd_dict['region_code'] = [os.path.split(f)[1].split('.')[0] for f in file_paths]
    for ii in range(len(predictions)):
        predictions[ii] = np.array(predictions[ii], dtype=np.float32)

        for idx in range(predictions[ii].shape[-1]):
            pd_dict[str(ii) + "_" + str(idx)] = np.transpose(predictions[ii])[idx]

    compare = pd.DataFrame(data=pd_dict)
    compare.to_csv(out_filepath)
    print("Done")
