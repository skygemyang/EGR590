**AI for Everyone Final Report**

Flora Shi, Zhichen Guo and Tianqi Yang

**Introduction**

The diagnosis of blood-based diseases often involves identifying and characterizing patient blood samples. Traditional identification of blood cells is based on manual observation of features such as cell size, nuclear shape and the presence of cytoplasmic granules. This is a time and labor-intensive work, and its results are highly variable because of the differences between technician&#39;s hands and measuring standards. Therefore, it is important for clinical trials to have an automated and consistent method to detect and classify blood cells in microscope images. In this project, we will make use a state-of-the-art machine learning model, You only look once (YOLO) [1], to detect four types of white blood cells (Eosinophil, Lymphocyte, Monocyte, and Neutrophil) in provided images. We choose the YOLOv3 as it surpasses previous YOLO versions and most other object-detection systems considering speed and accuracy factors together.

**Methods**

_Image dataset preparation_

We obtained blood cell images from two sources. One is the Kaggle Blood Cell Images Dataset [2], which provides around 400 original blood cell microscope images and 3000 augmented images for each white blood cell type. The other one is the Acute Lymphoblastic Leukemia Image Database (ALL-IDB) [3], which provides 108 blood cell images collected from ALL patients or healthy volunteer. The augmented Kaggle images are used for model training and validation, and the types and positions of white blood cells in these images were labeld using LabelImg [4] software. We first used 1200 labeled augmented Kaggle images (1000 images for training dataset and 200 images for validation dataset) to train the model. To improve the model&#39;s detection range, we tried to increase the number of labeled images to 2000 (1600 images for training dataset, and 400 images for validatation dataset). After training, we randomly chose 20 pre-augmented Kaggle images and 20 ALL-IBD images to test the trained model.

_Model preparation_

All our work in this project was done in the GPU environment provided by Google Colab. After downloading the Darknet package [5], the open source Yolov3 framework, we did

the following editions followed two online protocols [5,6]:

1. To fit yolov3 with Colab environment, we changed GPU= 1, CUDNN=1 and OPENCV=1 in the Makefile building the Darknet package in Google Colab.
2. We add our blood cell images and labels into the data/ folder, and created &quot;train.txt&quot;, &quot;valid.txt&quot; and &quot;test.txt&quot; files to define directory pathes of the images.
3. We also created &quot;bloodcells.data&quot; and &quot;bloodcells.name&quot; files to define the number and names of classes.
4. To fit yolov3 model with our costumed dataset, we changed basic parameters in the &quot;yolov3.cfg&quot; configuration file:

- &#39;batch=1&#39; to &#39;batch=64&#39;
- &#39;subdivisions=1&#39; to &#39;subdivisions=16&#39;
- &#39;max\_batches = 500200&#39; to &#39;max\_batches = 4000&#39;
- &#39;steps=400000,450000&#39; to &#39;steps=3000,3500&#39;
- &#39;filters=255&#39; to &#39;filters=27&#39;

5. To improve the model in capturing detailed features of images, we tried to modify some optional parameters in the &quot;yolov3.cfg&quot; configuration file:

- &#39;width=416&#39; to &#39;width=608&#39;
- &#39;height=416&#39; to &#39;height=608&#39;
- &#39;saturation = 1.5&#39; to &#39;saturation = 1.6&#39;
- &#39;exposure = 1.5&#39; to &#39;exposure = 1.6&#39;
- &#39;ignore\_thresh = .7&#39; to &#39;ignore\_thresh = .65&#39;

Weights are saved after 1000, 2000, 3000 and 4000 training batches. We also saved log file to tract average loss and mean average precision (mAP@0.5) values during the training.

_GitHub repository_

We created a GitHub repository (https://github.com/skygemyang/EGR590.git) to store the datasets, scripts and results related to this project. Results can be reproduced by following steps in yolov3\_bloodcells.ipynb file in our GitHub repository.

**Results**

We failed to train the model when setting resizing width and height standards for input images at 608 because of the memory limitation on Google Colab. All other trainings based on 1200 training images, or 2000 training images, or 2000 training images with modification of saturation/exposure/ignore\_thresh parameters are successfully completed in about 5 hours on Google Colab. In all training conditions, the average loss stops decreasing after ~300 batches

(Figure 1), yet the mean average precision (mAP@0.5) does not reach a plateau over 0.99 until ~3300 iterations (Figure 2). Therefore, we chose the weights obtained after training 4000 batches of images to test the model.

![](RackMultipart20200502-4-oze822_html_698cc7ffee20a59c.png)

Figure 1. Track of average loss during training in three different setting.

![](RackMultipart20200502-4-oze822_html_6a6254eb2724fa59.png)

Figure 2. Track of average mAP@0.5 during training in three different setting.

All three trained model successfully predicted the correct cell type and positions in most pre-augmented Kaggle images. We found that increasing the number of training images improves the model&#39;s ability to detect cells with slight abnormal structure (Figure 3A). However, neither increasing training images nor modification of saturation/exposure/ignore\_thresh parameters can help the model to tell two adhesive cells (Figure 3B).Finally, all three trained model were not able to predict the correct position or bounding box of single white blood cells in ALL-IBD images (Figure 4).

![](RackMultipart20200502-4-oze822_html_14969d59efa22b04.png)

Figure 3. Prediction result. (A) Increasing training image numbers helped the model to detect a lymphocyte with large cytoplasm area. (B) Modification of saturation, exposure and ignore\_thresh parameters did not help the model to differentiate adhesive cells.

![](RackMultipart20200502-4-oze822_html_6461d94663864346.png)

Figure 4. Models trained on Kaggle datasets cannot detect single white blood cells in ALL-IDB images.

**Discussion**

The most probable reason why our trained models cannot differentiate two adhesive cells is that all our training images only contain one target cell. Therefore, the model did not learn enough features to detect more than one cell in one image. Comparing to Kaggle images, the ALL-IDB images have multiple target cells in each image with different resolution and high background. We think the best way to improve the prediction acurady is to include images from both dataset in the training and validation datasets.

In sum, we conclude that YOLO is a potential deep learning model to achieve automatic detection of white blood cells image microscope images. The accuracy of prediction is highly dependent on the number and representativeness of training images.

**Reference**

[1] YOLO: Real-Time Object Detection. Redmon, J. 2019. [online] Available at:

[https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)

[2] Kaggle.com. 2020. Blood Cell Images. [online] Available at:

[https://www.kaggle.com/paultimothymooney/blood-cells](https://www.kaggle.com/paultimothymooney/blood-cells)

[3] ALL-IDB. 2010. [online] Available at:

[https://homes.di.unimi.it/scotti/all/](https://homes.di.unimi.it/scotti/all/)

[4] GitHub. 2020. Tzutalin/Labelimg. [online] Available at:

[https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg)

[5] GitHub. 2020. Alexeyab/Darknet. [online] Available at:

[https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)

[6] Medium. 2020. How To Train Yolov3 On Google Colab To Detect Custom Objects

(E.G: Gun Detection). [online] Available at:

[https://medium.com/@quangnhatnguyenle/how-to-train-yolov3-on-google-colab-to-detect-custom-objects-e-g-gun-detection-d3a1ee43eda1](https://medium.com/@quangnhatnguyenle/how-to-train-yolov3-on-google-colab-to-detect-custom-objects-e-g-gun-detection-d3a1ee43eda1)
