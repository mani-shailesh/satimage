import os
from keras.models import model_from_json


def restore_model(json_filepath, weights_filepath):
    print("Loading model from disk...")
    json_file = open(json_filepath, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_filepath)
    print("Done")
    return model


def generate_predictions(model, data_generator_obj, num_val_points, save_dir):
    print("Generating predictions...")
    predictions = model.predict_generator(data_generator_obj,
                                          val_samples=num_val_points,
                                          pickle_safe=True)
    
    pd_dict = dict()
    for ii in range(len(predictions)):
        predictions[ii] = np.array(predictions[ii], dtype=np.float32)

        for idx in range(predictions[ii].shape[-1]):
            pd_dict[str(ii) + "_" + str(idx)] = np.transpose(predictions[ii])[idx]

    compare = pd.DataFrame(data=pd_dict)
    compare.to_csv(os.path.join(save_dir, "compare.csv"))
    print("Done")
