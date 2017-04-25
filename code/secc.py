import pandas as pd
import numpy as np

BATCH_SIZE = 50


def generate_predictions(model, developmental_filepath, region_info_filepath, out_filepath, batch_size=BATCH_SIZE):
    """
    Generate predictions of income level for each sub-district and save the results at specified path.
    :param model: pre-trained 'Model' object
    :param developmental_filepath: path of file containing values of developmental parameters for each region
    :param region_info_filepath: path of file containing information (including 'tehsil_code' and 'num_households') 
    for each region
    :param out_filepath: path of file to write the predictions
    :param batch_size: Batch size to be used for generating predictions
    """
    print("Reading developmental parameter values...")
    region_data_dict = {}
    data = pd.read_csv(developmental_filepath)
    header_list = list(data)[1:]
    for ii, row in data.iterrows():
        region_code = row['region_code']
        region_data_dict[region_code] = {}
        for header in header_list:
            param_idx, column_idx = int(header.split('_')[0]), int(header.split('_')[1])
            if param_idx not in region_data_dict[region_code]:
                region_data_dict[region_code][param_idx] = {}
            region_data_dict[region_code][param_idx][column_idx] = row[header]
    print("Done.")

    print("Aggregating values at sub-district level...")
    tehsil_data_dict = {}
    data = pd.read_csv(region_info_filepath)
    for ii, row in data.iterrows():
        region_code = row['region_code']
        tehsil_code = row['tehsil_code']
        num_households = row['num_households']
        if region_code in region_data_dict:
            if tehsil_code not in tehsil_data_dict:
                tehsil_data_dict[tehsil_code] = {
                    'num_households': 0,
                }
            region_dict = region_data_dict[region_code]
            tehsil_data_dict[tehsil_code]['num_households'] += num_households
            for param_idx in region_dict:
                if param_idx not in tehsil_data_dict[tehsil_code]:
                    tehsil_data_dict[tehsil_code][param_idx] = {}
                for column_idx in region_dict[param_idx]:
                    if column_idx not in tehsil_data_dict[tehsil_code][param_idx]:
                        tehsil_data_dict[tehsil_code][param_idx][column_idx] = 0
                    tehsil_data_dict[tehsil_code][param_idx][column_idx] += region_dict[param_idx][column_idx]

    data_matrix = []
    tehsil_code_list = []
    for tehsil_code in tehsil_data_dict:
        data_row = []
        tehsil_dict = tehsil_data_dict[tehsil_code]
        num_households = tehsil_dict.pop('num_households')
        for param_idx in sorted(tehsil_dict.keys()):
            for column_idx in sorted(tehsil_dict[param_idx].keys()):
                data_row.append(tehsil_dict[param_idx][column_idx] / num_households)
        data_matrix.append(data_row)
        tehsil_code_list.append(tehsil_code)

    data_matrix = np.array(data_matrix)
    print("Done.")

    print("Predicting income level values...")
    predictions = model.predict(data_matrix, batch_size=batch_size)
    print("Done.")

    print("Writing predictions to file...")
    param_idx = 0
    pd_dict = dict()
    order = ['tehsil_code']
    pd_dict['tehsil_code'] = tehsil_code_list
    predictions = np.array(predictions, dtype=np.float32)
    for column_idx in range(predictions.shape[-1]):
        pd_dict[str(param_idx) + "_" + str(column_idx)] = np.transpose(predictions)[column_idx]
        order.append(str(param_idx) + "_" + str(column_idx))

    compare = pd.DataFrame(data=pd_dict)

    compare = compare[order]
    compare.to_csv(out_filepath, index=False)
    print("Done.")
