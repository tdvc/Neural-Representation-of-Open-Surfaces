# Neural Representation of Open Surfaces

<img src="/images/interpolations.gif" align="center" width="40%"><img src="/images/MGN_SSDF_lv_analysis.png" align="center" width="35%">

This GitHub Repository contains the code behind the paper [Neural Representation of Open Surfaces](https://www.thorshammer.dk/papers/Neural_Representation_of_Open_Surfaces.pdf). In this work we show how it is possible to encode shapes with boundary curves (open surfaces) within a simple MLP and an associated latent space by representing the shapes as the zero level set of a learned Semi-Signed Distance Field. This can be used to represent a variety of shapes, interpolate between shapes (shown as an interpolation between a pair of pants and a shirt in the GIF to the left), cluster shapes according to similiar topology and geometry (shown in the image to the right) and to do shape completion.

For more details about the project, please visit the [project page](https://www.thorshammer.dk/projectpages/ssdf.html) for this work.

## Citation

If you find our work useful, please cite the paper: 

```
@article {10.1111:cgf.14916,
    journal={Computer Graphics Forum},
    title={{Neural Representation of Open Surfaces}},
    author={Christiansen, Thor V. and B{\ae}rentzen, Jakob Andreas and Paulsen, 
      Rasmus R. and Hannemose, Morten R.},
    year={2023},
    publisher={The Eurographics Association and John Wiley & Sons Ltd.},
    ISSN= {1467-8659},
    DOI={10.1111/cgf.14916}
    }
```

## Installations

In order to use the code, different packages are required. Specifically, [PyGEL](http://www2.compute.dtu.dk/projects/GEL/PyGEL/) and [libIGL](https://libigl.github.io). Depending on your system, these libraries can be installed via pip by using the following commands:

```
pip install PyGEL3D
```

and 

```
python -m pip install libigl
```

Alternatively, you can build the libraries yourself. For the PyGEL library a guide is provided [here](https://github.com/janba/GEL) and for the libIGL library, a guide is [here](https://github.com/libigl/libigl-python-bindings).

Moreover, PyTorch is required, which can be installed from [here](https://pytorch.org). Lastly, a couple of standard packages are needed, which can be installed by running the following line inside the directory of this code:

```
pip install -r requirements.txt
```

## Usage 
In order to use this code, you need to split your files into a training set and a test set. Then you should place the training meshes inside the train folder under experiment -> your experiment -> data -> train. The same thing goes for the test meshes, which should be put into the test folder at the same location as the train folder.

Note: It is assumed that the meshes are aligned on beforehand (This can be done manually, using landmarks or simple ICP). Moreover, it is assumed that the meshes have been centered around the origin and scaled in such a way that they fit inside the unit sphere. Lastly, the files should be triangle meshes in .obj file format.

The preprocessing step can be done using the Jupyter Notebook "Data Preprocssing". Simply provide the path to your meshes in the Jupyter Notebook and make a list of the names of the training files and a list of the names of the test files inside the notebook. Then all the meshes will be centered and scaled as well as saved in the respective folders. For more information see the Notebook.

### Train and Inference
It is possible to train two different kinds of networks. One network that learns two signals: SSDF and UDF and another network which approximates the GWN. To train the SSDF network, simply go into the folder "your_experiment" and run: 

```
python ../../train.py "your_experiment" "ssdf"
```

If you want to train the GWN network, just replace "ssdf" with "gwn". 

Similarly, if you want to do inference, go into the folder "your_experiment" and run: 

```
python ../../test.py "your_experiment" "ssdf".
```

Also, just replace "ssdf" with "gwn", if you want to do inference for the gwn based network.

### Surface Reconstruction
When you are done training, you need to reconstruct the meshes in the training samples in order to find the threshold "k" for the gradient length. Please see the [paper](https://www.thorshammer.dk/papers/Neural_Representation_of_Open_Surfaces.pdf) for more details. 

Running surface reconstruction:

```
python ../../shape_reconstruction.py "your_experiment" "train" "ssdf"
```
If you want to reconstruct the surfaces of the training samples for the GWN network, just replace "ssdf" with "gwn". 

### Finding GWN gradient threshold k
Afterwards you should just run: 

```
python ../../find_gwn_threshold.py "your_experiment" "ssdf".
```

Also, just replace "ssdf" with "gwn", if you want to find the threshold k for the gwn based network.

Then you can reconstruct the closed surfaces in the test set by running: 

```
python ../../shape_reconstruction.py "your_experiment" "test" "ssdf"
```
And to recover the true surfaces (the shapes with boundary curves) you need to run: 

```
python ../../open_shapes.py "your_experiment" "test" "ssdf"
```

### Interpolations
If you want to interpolate between two different meshes, go to "your_experiment" and run the following command:

```
python ../../interpolations.py "your_experiment" "ssdf" "test" "file1" "file2" n
```

Here "your_experiment" is the name of the folder in which your data is. Similar to before "ssdf" can be exchanged with "gwn", if you want to interpolate between two shapes, where the network uses the "gwn" as an implicit surface representation. "file1" is the name of the first shape (without the .obj extension), "file2" is the name of the second file you want to interpolate between (again without the .obj extension) and n is the number of interpolations including the original meshes. Notice: Before running the interpolations script, you need to first reconstruct the shapes in the training set and then find the threshold "k". 

### Configurations
If you want to change the configurations for your experiment e.g. the number of epochs, the number of points sampled each for each shape etc., go into the "cfgs" folder in "your_experiment" folder, find the file "default.yaml" and change the parameters.

## Questions and Problems
If you have any questions or encounter any problems with the code, please do not hesitate to contact me on [tdvc@dtu.dk](tdvc@dtu.dk).
