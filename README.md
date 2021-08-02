[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"

# Udacity DSND Captstone Project - Dog Breed Classifier Using Transfer Learning


## Project Overview and Motivation

This is the Captstone Project of the Udacity Data Science Nano Degree.  In this project we will build a Dog Breed Multiclass Classifier using CNNs (Convolutional Neural Networks).

The consists of the next task: given an image, the algorithm must return an output of what it thinks to be... if is it an human, a dog or if any other image is interted (an error message will be displayed).  

If a dog or human is predicted on the image it will return back the breed of the dog (even if it is human, just for fun, resembling the dog breed).

There are two datasets of dogs and humans.  The dog dataset consists of 133 categories of dogs and with a total of 8351 images itself divided in 6680 training images, 835 validation images and 836 testing images for model metrics.

<p align="center">
<b>Figure 1 - Dog Breed Classification Output</b>
<img src = "./readme_imgs/overview.PNG?raw=true" width="75%"/>
</p>

On the notebook we will explore several tasks including a vanilla deep learning model and finally explore transfer learning methods and data agumentation for increase accuracy.

You can find [the blog post here](https://daqsyspty.com/2021/08/dog-breed-classifier-using-cnns-and-transfer-learning/) for more reading details.


## Project Libraries Requirements

- opencv-python==3.2.0.6
- h5py==2.6.0
- matplotlib==2.0.0
- numpy==1.12.0
- scipy==0.18.1
- tqdm==4.11.2
- keras==2.0.2
- scikit-learn==0.18.1
- pillow==4.0.0
- ipykernel==4.6.1
- tensorflow==1.0.0



## File Descriptions

Six (6) folders are present on the repository.

- <b>bottleneck_features</b>: it will contain VGG16 bottleneck features implemented in step 4 of the dog_app notebook.

- <b>haarcascateds</b>: has the OpenCV default front face detector used on step 1 of the dog_app notebook.

- <b>images</b>: it contains the dataset of images.

- <b>readme_images</b>: the image of the overview and results that we are displaying in this readme.

- <b>requirements</b>: contains all necessary files that will help you to get up and running you application if you want to clone the project and run on your computer, the data science nanodegree project folder or a cloud service like AWS.

- <b>saved_models</b>: contain the files for the pretrained ResNet50, VGG16 and Inception models used in steps 2, 4 and 5 of the dog_app notebook.

There are also five (5) files related to the project:

- <b>.gitignore</b>: if the project is downloaded and then migrated to the repository, this file will ensure that big data files remain on your local machine and will not get up to the remote repository.

- <b>dog_app.html</b>: the output project in HTML format of the dog_app.ipynb.

- <b>dog_app.ipynb</b>: The notebook of the datascience capstone project.

- <b>extract_bottleneck_features.py</b>: contains functions for downloading pretrained models and for the transfer learning section.

- <b>REAMDE.md</b>: this file as the information on motivation, file descriptions, how to run and results.

## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

7. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
8. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

12. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. __Unless requested, do not modify code that has already been included.__

## Major Parts of the Project in CNN

We first started with a small approach using a vanilla implementation of a neural network.

<p align="center">
<b>Figure 2 - Vanilla Convolutional Neural Network </b>
<img src = "./readme_imgs/vanilla_cnn.PNG?raw=true" width="75%"/>
</p>

Later we explored with a VGG16 pre-trained model without data augmentation method

<p align="center">
<b>Figure 3 - VGG16 Pretrained Model Using Transfer Learning Technique </b>
<img src = "./readme_imgs/vgg16_pretrained.PNG?raw=true" width="75%"/>
</p>

Finally we used an Inception model to do transfer learning with data augmentation.

<p align="center">
<b>Figure 4 - Inception Model Using Transfer Learning Technique </b>
<img src = "./readme_imgs/inception_tl.PNG?raw=true" width="75%"/>

</p>
<p align="center">
<b>Figure 5 - Data Augmentation Pipeline </b>
<img src = "./readme_imgs/data_augmentation.PNG?raw=true" width="75%"/>
</p>

## Results

We can show below the image of the results of the model:

<p align="center">
<b>Figure 6 - First 3 results of the Dog Breed Classifier Output </b>
<img src = "./readme_imgs/results_1.PNG?raw=true" width="75%"/>
</p>

<p align="center">
<b>Figure 8 - Last 3 results of the Dog Breed Classifier Output </b>
<img src = "./readme_imgs/results_2.PNG?raw=true" width="75%"/>
</p>

Below you will se a table of the results of the training, validation and test datasets over different models and its metrics.

<center>

Table 1.  Training Results using Different CNN Configurations.

| - | Vanilla CNN  |  VGG16 | Inception |
|---|---|---|---|
|  Data Augmentation | No | Yes | Yes |
|  Training Loss | 3.1444  |  7.7105 | 0.0366  |
|  Training Accuracy |  0.2401 |  0.5115 | 0.9915  |
|  Validation Loss |  3.6114 |  8.4304 | 0.5816  | 
|  Validation Accuracy |  0.1310 |  0.4156 | 0.8546  |
| Test Accuracy Metric | 17.58%| 41.86% | 81.1% |
</center>

<details open>
<summary> :iphone: <b>Having Problems?<b></summary>

<p align = "center">

[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/riawa)
[<img src="https://img.shields.io/badge/telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"/>](https://t.me/issaiass)
[<img src="https://img.shields.io/badge/instagram-%23E4405F.svg?&style=for-the-badge&logo=instagram&logoColor=white">](https://www.instagram.com/daqsyspty/)
[<img src="https://img.shields.io/badge/twitter-%231DA1F2.svg?&style=for-the-badge&logo=twitter&logoColor=white" />](https://twitter.com/daqsyspty) 
[<img src ="https://img.shields.io/badge/facebook-%233b5998.svg?&style=for-the-badge&logo=facebook&logoColor=white%22">](https://www.facebook.com/daqsyspty)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/riawe)
[<img src="https://img.shields.io/badge/tiktok-%23000000.svg?&style=for-the-badge&logo=tiktok&logoColor=white" />](https://www.linkedin.com/in/riawe)
[<img src="https://img.shields.io/badge/whatsapp-%23075e54.svg?&style=for-the-badge&logo=whatsapp&logoColor=white" />](https://wa.me/50766168542?text=Hello%20Rangel)
[<img src="https://img.shields.io/badge/hotmail-%23ffbb00.svg?&style=for-the-badge&logo=hotmail&logoColor=white" />](mailto:issaiass@hotmail.com)
[<img src="https://img.shields.io/badge/gmail-%23D14836.svg?&style=for-the-badge&logo=gmail&logoColor=white" />](mailto:riawalles@gmail.com)

</p

</details>

<details open>
<summary> <b>License<b></summary>
<p align = "center">
<img src= "https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-sa.svg" />
</p>
</details>
