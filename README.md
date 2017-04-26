<p align="center">
<b><a href="#introduction">Introduction</a></b>
|
<b><a href="#data">Data</a></b>
|
<b><a href="#instructions">Instructions</a></b>
|
<b><a href="#filters">Filters</a></b>
|
<b><a href="#acknowledgements">Acknowledgements</a></b>
</p>

<a href="https://github.com/agarwalt/satimage">
<div align="center">
	<img src="readme_images/header_img.png">
</div>
</a><br>

#### Introduction

This repository accompanies the manuscript "Predicting Poverty and Developmental Statistics from Satellite Images using Multi-task Deep Learning" and contains the code and model weights for two prediction tasks: 

1. Predict, using a multi-task fully convolutional deep neural network (<a href="models/developmental/model.png" target="_blank">PNG</a>, <a href="models/developmental/best_model_architecture.json" target="_blank">JSON</a>, <a href="https://www.dropbox.com/s/187e6zp2or2s9ni/best_model_weights.h5?dl=0" target="_blank">Weights</a>), three developmental parameters - the main material of the roof, source of lighting and source of drinking water - from satellite imagery.
2. Predict, using a simple four-layer fully-connected neural network (<a href="models/income_poverty_pd/model.png" target="_blank">PNG</a>, <a href="models/income_poverty_pd/best_model_architecture.json" target="_blank">JSON</a>, <a href="https://www.dropbox.com/s/ml3hkms3nlx0k0u/best_model_weights.h5?dl=0" target="_blank">Weights</a>), the income levels (a direct indicator of poverty) using the predicted developmental parameter outputs of the first model.
2. Predict, using a simple four-layer fully-connected neural network (<a href="models/income_poverty_cd/model.png" target="_blank">PNG</a>, <a href="models/income_poverty_cd/best_model_architecture.json" target="_blank">JSON</a>, <a href="" target="_blank">Weights</a>), the income levels (a direct indicator of poverty) using the actual developmental parameter values.


#### Data

We obtained the Census of India (2011) data from these websites: 

1. Primary Census Abstract Data Tables: <a href="http://censusindia.gov.in/pca/pcadata/pca.html" target="_blank">http://censusindia.gov.in/pca/pcadata/pca.html</a>
2. Percentage of Households to Total Households by Amenities and Assets: <a href="http://www.censusindia.gov.in/2011census/HLO/HL_PCA/Houselisting-housing-HLPCA.html" target="_blank">http://www.censusindia.gov.in/2011census/HLO/HL_PCA/Houselisting-housing-HLPCA.html</a>
3. Socio-Economic and Caste Census: <a href="http://www.secc.gov.in/statewiseEmploymentAndIncomeReport?reportType=Employment%20and%20Income" target="_blank">http://www.secc.gov.in/statewiseEmploymentAndIncomeReport?reportType=Employment%20and%20Income</a>

We utilized Google's Geocoding API (https://developers.google.com/maps/documentation/geocoding/) to obtain coordinates of the center of a region from its address in the census data.  

Further, we used Google Static Maps API (https://developers.google.com/maps/documentation/static-maps/) to extract 1920x1920 satellite images for the villages at the "16" zoom level.

We provide a small subset of our dataset in this repository to test both models.


#### Instructions

###### Predicting developmental parameters:
1. <a href="data/region_info.csv" target="_blank">data/region_info.csv</a> contains the centre latitudes and longitudes for all the regions in shared dataset. Use Google Static Maps API (https://developers.google.com/maps/documentation/static-maps/) to extract 1920x1920 satellite images for these regions at the 16 zoom level. Make sure that the name of each image is `region_code.png` e.g. for region with `region_code = 12345`, name of the image must be `12345.png`
2. Download the model weights from https://www.dropbox.com/s/187e6zp2or2s9ni/best_model_weights.h5?dl=0 and place the downloaded file in `models/developmental`
3. `cd code` and launch the interactive python shell using `ipython` or any other python notebook of your choice.  
4. Load the pre-trained multi-task fully-convolutional model using: 
    ```python
    import util
    model = util.restore_model('../models/developmental/best_model_architecture.json', '../models/developmental/best_model_weights.h5')
    ``` 
5. Generate and save predictions of developmental parameters from the downloaded images (assuming that they are saved in `images` directory) using:
    ```python
    import satimage
    satimage.generate_predictions(model, '../images', '../data/predicted_developmental.csv')
    ```
    The file `data/predicted_developmental.csv` should now contain the predictions of developmental parameters for those regions whose images were provided.
    
###### Visualizing filter responses:
1.  Load the model using steps `1-4` listed earlier and then to see and save filter responses for a given image at a specified layer use the following snippet:
    ```python
    import satimage
    layer_index = 19
    filter_index = None
    input_img_path = '../images/12345.png'
    save_dir = '../images'
    satimage.show_filter_responses(model, layer_index, input_img_path, save_dir, filter_index)
    ```
    Vary the `layer_index`, `filter_index` and `input_img_path` to see the filters at specific layer for specific image. A copy of the filter responses will also be saved at `save_dir`.
    
###### Predicting income level and poverty:
1.  Download the model weights from https://www.dropbox.com/s/ml3hkms3nlx0k0u/best_model_weights.h5?dl=0 and https://www.dropbox.com/s/ml3hkms3nlx0k0u/best_model_weights.h5?dl=0 and place the downloaded file in `models/income_poverty_pd` and `models/income_poverty_cd` respectively.
3. `cd code` and launch the interactive python shell using `ipython` or any other python notebook of your choice.  
4. Load the pre-trained income level predictions models using: 
    ```python
    import util
    model_pd = util.restore_model('../models/income_poverty_pd/best_model_architecture.json', '../models/income_poverty_pd/best_model_weights.h5')
    model_cd = util.restore_model('../models/income_poverty_cd/best_model_architecture.json', '../models/income_poverty_cd/best_model_weights.h5')
    ``` 
5. Generate and save predictions of income level from the developmental parameters (assuming that they have already been predicted and saved) using:
    ```python
    import secc
    secc.generate_predictions(model_pd, '../data/predicted_developmental.csv', '../data/region_info.csv', '../data/pd_tehsil_income.csv')
    secc.generate_predictions(model_cd, '../data/data_developmental.csv', '../data/region_info.csv', '../data/cd_tehsil_income.csv')
    ```
    The file `data/pd_tehsil_income.csv` should now contain the predictions of income levels using predicted developmental parameters while the file `data/cd_tehsil_income.csv` should contain the predictions of income levels using actual values of the developmental parameters.
6. To compare these predicted results against the actual ones and also to see the accuracy of povery prediction using these predictions use the following snippet:
    ```python
    secc.compare_income_predictions('../data/data_tehsil_income.csv', '../data/cd_tehsil_income.csv')  # For model trained on actual data
    secc.compare_income_predictions('../data/data_tehsil_income.csv', '../data/pd_tehsil_income.csv')  # For model trained on predicted data
    ```

#### Filters

We provide filter responses for our first (multi-task) model.

##### Shared Layers

In the multi-task model's first block, consistent with the observations reported in the literature, filters learn edges with different orientations. The figure below shows differently oriented edges for different filters for a particular region.

<div align="center">
	<img src="filter_responses/shared_1.png">
</div>
<br>

##### Task-specific Layers

The following figures illustrate the filter responses for the task specific branches for each of the three tasks in the multi-task model. In each figure, the larger images show different areas of two regions (`A` and `B`). The smaller images show filter activations for: 

1. Roof type 
2. Source of lighting, and,
3. Source of drinking water. 

In contrast to the activations for shared layers' filters, the activations for the task-specific branches in the multi-task model highlight objects of relevance to the respective tasks.

<div align="center">
	<img src="filter_responses/task_specific_1.png">
</div>
<br>

<div align="center">
	<img src="filter_responses/task_specific_2.png">
</div>
<br>

<div align="center">
	<img src="filter_responses/task_specific_3.png">
</div>
<br>


#### Acknowledgements

The authors are grateful to NVIDIA Corporation for donating the TITAN X GPUs used for this research.
