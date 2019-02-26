# Style Stack
A library for style similarity search for art and design images.

## Contents
- **analysis.ipynb**: a walkthrough of the the EDA, development, and analysis of my similarity search method.
Note that there is some dynamic inheritence modificaiton and method declaration (monkey patching) to enhance the story-like flow of the notebook, which is not included in the `analysts.py` class versions, and would not be included in production code.
- **analysts.py**: classes used in analysis and EDA
- **style_stack.py**: primary similarity search classes for production


## Usage
### Build `GramStack`
Import `GramStack` and choose a model from `keras.applications`
```python
from keras.applications.vgg16 import VGG16
from stylestack.gram_stack import GramStack
```
Set up arguments
```python
image_dir = '../data/my_data'
model = VGG16(weights='imagenet', include_top=False)
layer_range = ('block1_conv1', 'block2_pool')
```
Build `GramStack`
```python
stack = GramStack.build(image_dir, model, layer_range)
```
### Query `GramStack`
Set weighting for embedding layers in similarity search. Any layers not specified will be weighted as 0, or all layers can be used by specifying `None`.
```python
embedding_weights = {
    'block1_conv1': 1,
    'block3_conv2': 0.5,
    'block3_pool': .25
}
```
Set other arguments. Use `write_output` to output results JSON to `/output/`.
```python
image_path = '../data/my_data/cat_painting.jpg'
n_results = 5
write_output = True
```
Query `GramStack`
```python
results = stack.query(image_path, embedding_weights, n_results, write_output)
```
### Save to disk
```
stack.save(lib_name='my_data')
```
### Load from disk
```
GramStack.load(lib_name='my_data')
```
Once the `GramStack` is loaded, it can be queried and behaves the same as when it was built.


## Installation
To install the package above, pleae run:

### GPU
```shell
conda install faiss-gpu -c pytorch
conda install requirements.txt
```

### CPU
```shell
conda install faiss-cpu -c pytorch
conda install requirements.txt
```

#### Requisites

- conda

#### Dependencies

- [faiss](https://github.com/facebookresearch/faiss/wiki)
- [joblib](https://joblib.readthedocs.io/en/latest/)
- [keras](keras.io)
- [matplotlib](https://matplotlib.org/)
- [numpy](http://www.numpy.org/)
- [pillow](https://python-pillow.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

