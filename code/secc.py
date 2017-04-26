import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.stats import pearsonr

BATCH_SIZE = 50


def generate_predictions(model, developmental_filepath, village_info_filepath, out_filepath, batch_size=BATCH_SIZE):
    """
    Generate predictions of income level for each sub-district and save the results at specified path.
    :param model: pre-trained 'Model' object
    :param developmental_filepath: path of file containing values of developmental parameters for each village
    :param village_info_filepath: path of file containing information (including 'subdistrict_code' and 'num_households') 
    for each village
    :param out_filepath: path of file to write the predictions
    :param batch_size: Batch size to be used for generating predictions
    """
    print("Reading developmental parameter values...")
    village_data_dict = {}
    data = pd.read_csv(developmental_filepath)
    header_list = list(data)[1:]
    for ii, row in data.iterrows():
        village_code = row['village_code']
        village_data_dict[village_code] = {}
        for header in header_list:
            param_idx, column_idx = int(header.split('_')[0]), int(header.split('_')[1])
            if param_idx not in village_data_dict[village_code]:
                village_data_dict[village_code][param_idx] = {}
            village_data_dict[village_code][param_idx][column_idx] = row[header]
    print("Done.")

    print("Aggregating values at sub-district level...")
    subdistrict_data_dict = {}
    data = pd.read_csv(village_info_filepath)
    for ii, row in data.iterrows():
        village_code = row['village_code']
        subdistrict_code = row['subdistrict_code']
        num_households = row['num_households']
        if village_code in village_data_dict:
            if subdistrict_code not in subdistrict_data_dict:
                subdistrict_data_dict[subdistrict_code] = {
                    'num_households': 0,
                }
            village_dict = village_data_dict[village_code]
            subdistrict_data_dict[subdistrict_code]['num_households'] += num_households
            for param_idx in village_dict:
                if param_idx not in subdistrict_data_dict[subdistrict_code]:
                    subdistrict_data_dict[subdistrict_code][param_idx] = {}
                for column_idx in village_dict[param_idx]:
                    if column_idx not in subdistrict_data_dict[subdistrict_code][param_idx]:
                        subdistrict_data_dict[subdistrict_code][param_idx][column_idx] = 0
                    subdistrict_data_dict[subdistrict_code][param_idx][column_idx] += \
                        village_dict[param_idx][column_idx] * num_households

    data_matrix = []
    subdistrict_code_list = []
    for subdistrict_code in subdistrict_data_dict:
        data_row = []
        subdistrict_dict = subdistrict_data_dict[subdistrict_code]
        num_households = subdistrict_dict.pop('num_households')
        for param_idx in sorted(subdistrict_dict.keys()):
            for column_idx in sorted(subdistrict_dict[param_idx].keys()):
                data_row.append(subdistrict_dict[param_idx][column_idx] / num_households)
        data_matrix.append(data_row)
        subdistrict_code_list.append(subdistrict_code)

    data_matrix = np.array(data_matrix)
    print("Done.")

    print("Predicting income level values...")
    predictions = model.predict(data_matrix, batch_size=batch_size)
    print("Done.")

    print("Writing predictions to file...")
    param_idx = 0
    pd_dict = dict()
    order = ['subdistrict_code']
    pd_dict['subdistrict_code'] = subdistrict_code_list
    predictions = np.array(predictions, dtype=np.float32)
    for column_idx in range(predictions.shape[-1]):
        pd_dict[str(param_idx) + "_" + str(column_idx)] = np.transpose(predictions)[column_idx]
        order.append(str(param_idx) + "_" + str(column_idx))

    compare = pd.DataFrame(data=pd_dict)

    compare = compare[order]
    compare.to_csv(out_filepath, index=False)
    print("Done.")


def compare_income_predictions(original_filepath, predicted_filepath):
    """
    Compare actual and predicted income levels
    :param original_filepath: Path of the file containing actual income level values
    :param predicted_filepath:  Path of the file containing predicted income level values
    :return: 
    """
    original_subdistrict_dict = {}
    original_values = []
    predicted_values = []
    data_original = pd.read_csv(original_filepath)
    data_predicted = pd.read_csv(predicted_filepath)
    header_list = list(data_predicted)[1:]
    for ii, row in data_original.iterrows():
        original_subdistrict_dict[row['subdistrict_code']] = [row[header] for header in header_list]
    for ii, row in data_predicted.iterrows():
        predicted_values.append([row[header] for header in header_list])
        original_values.append(original_subdistrict_dict[row['subdistrict_code']])
    original_values = np.array(original_values)
    predicted_values = np.array(predicted_values)

    print("Correlation for each class:")
    print("[0] " + str(pearsonr(original_values[:, 0], predicted_values[:, 0])))
    print("[1] " + str(pearsonr(original_values[:, 1], predicted_values[:, 1])))
    print("[2] " + str(pearsonr(original_values[:, 2], predicted_values[:, 2])))

    print("\nPoverty prediction after thresholding on class [0]: ")
    t = 0.1
    while t < 1.0:
        p1m = np.copy(original_values[:, 0])
        p1m[p1m >= t] = 1
        p1m[p1m < t] = 0
        frac = np.sum(p1m) / len(p1m)
        ot = [1 if i >= t else 0 for i in original_values[:, 0]]
        pt = [1 if i >= t else 0 for i in predicted_values[:, 0]]
        print(
            "Threshold: " + str(t)
            + " Accuracy: " + str(accuracy_score(ot, pt))
            + " Baseline: " + str(max(frac, 1 - frac))
            + " Precision: " + str(precision_score(ot, pt))
            + " Recall: " + str(recall_score(ot, pt))
        )
        t += 0.1
