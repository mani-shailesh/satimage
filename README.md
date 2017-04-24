<p align="center">
<b><a href="#introduction">Introduction</a></b>
|
<b><a href="#filter%20responses">Filter Responses</a></b>
|
<b><a href="#todo">Todo</a></b>
</p>

<a href="https://github.com/agarwalt/satimage">
<div align="center">
	<img src="readme_images/header_img.png">
</div>
</a><br>

#### Introduction

This repository accompanies the manuscript "Predicting Poverty and Developmental Statistics from Satellite Images using Multi-task Deep Learning" and contains the code and model weights for two prediction tasks: 

1. Predict, using a multi-task fully convolutional deep neural network, three developmental parameters - the main material of the roof, source of lighting and source of drinking water - from satellite imagery.
2. Predict, using a simple four-layer fully-connected neural network, the income levels (a direct indicator of poverty) using the predicted developmental parameter outputs of the first model.

#### Filter Responses

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

#### Todo

Add code for running sample data through the models.
