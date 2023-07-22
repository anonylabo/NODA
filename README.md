# GTFormer: a Geospatial Temporal Transformer for Crowd Flow Prediction

This repository is the official implementation of [GTFormer: a Geospatial Temporal Transformer for Crowd Flow Prediction](https://arxiv.org/abs/2030.12345). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Experiment

1. Install the following:
   ```
   Python 3.10.6
   torch 2.0.1
   geopandas 0.10.2
   scikit_mobility 1.3.1
   rtree 1.0.1
   ```

2. We provide the experiment scripts of all models under the folder ./scripts. You can reproduce the experiment results by:
   ```
   ./scripts/NYC/GTFormer.sh
   ./scripts/DC/GTFormer.sh
   ``` 


## Results

Our model achieves the following performance on OD flow predicton:

<img src="https://github.com/kodakoda-koda/GTFormer/assets/87755637/e18d0a43-036a-480a-b471-6adaac0bf04b" width="500">

And our model achieves the following performance on IO flow prediction:

<img src="https://github.com/kodakoda-koda/GTFormer/assets/87755637/e312dca7-7198-4d86-a21e-5b01fd521175" width="350">


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
