# Semantic-Segmentation

Deep convolutional neural networks have been applied to a broad range of problems and tasks within Computer Vision. In this project, we will explore semantic segmentation task. Semantic segmentation means that every pixel in an image is classified as belonging to some category.

Details on the underlying technical approaches and results can be found [here](https://drive.google.com/file/d/1y61lZVzo-qwaNk2PqyKvexRQm6e0QCSn/view?usp=sharing)



### Installation

This code uses Python 3.6.

- Install dependencies
```bash
$ pip install -r requirements.txt
```

### Data

We will be using TAS500 for the task of semantic segmentation. This dataset has pixelwise annotation for 9 coarse and 23 fine-grained object categories. The statistics of the dataset can be found [here](https://mucar3.de/icpr2020-tas500/). The main goal of this
challenge is to recognize objects from a number of visual object classes in realistic scenes (i.e., not pre-segmented objects). In this project, we will be using 9 categories. 

- Download [link](https://drive.google.com/uc?export=download&id=1dCxJNWGbQT-tbuDTD6nQvmWPwGS9MPwT) 
- Extract files to ```data```.
- The contents of ```data``` should be the following:
```
data
 │  .gitignore
 │
 └──tas500v1.1
```


*Note: all python calls below must be run from ```./``` i.e. home directory of the project*
### Execution

The following will run the program with default config.
```bash
$ python main.py 
```

You can make the necessary changes in config.yaml.


### Results

The results will be stored in the `experiments` folder with the `experiment_name` provided in the config file.