# Style Stack
A library for scalable style similarity search for art and design images.
Note: Readme under construction

## Contents
- **Insight_Project_Framework** : Put all source code for production within structured directory
- **tests** : Put all source code for testing in an easy to find location
- **configs** : Enable modification of all preset variables within single directory (consisting of one or many config files for separate tasks)
- **data** : Include example a small amount of data in the Github repository so tests can be run to validate installation
- **build** : Include scripts that automate building of a standalone environment
- **static** : Any images or content to include in the README or web framework if part of the pipeline

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

## Requisites

- conda

#### Dependencies

- [Streamlit](streamlit.io)
- [faiss](https://github.com/facebookresearch/faiss/wiki)
- [matplotlib](https://matplotlib.org/)
- [pillow](https://python-pillow.org/)

#### Installation
To install the package above, pleae run:
```shell
pip install -r requiremnts
```

## Build Environment
- Include instructions of how to launch scripts in the build subfolder
- Build scripts can include shell scripts or python setup.py files
- The purpose of these scripts is to build a standalone environment, for running the code in this repository
- The environment can be for local use, or for use in a cloud environment
- If using for a cloud environment, commands could include CLI tools from a cloud provider (i.e. gsutil from Google Cloud Platform)
```
# Example

# Step 1
# Step 2
```

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- **DO NOT STORE CREDENTIALS IN THE CONFIG DIRECTORY!!**
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)


## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
