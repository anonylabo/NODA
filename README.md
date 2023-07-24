# GTFormer: a Geospatial Temporal Transformer for Crowd Flow Prediction

This repository is the official implementation of [GTFormer: a Geospatial Temporal Transformer for Crowd Flow Prediction](https://arxiv.org/abs/2030.12345). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Experiment

1. Download data from [here](https://drive.google.com/drive/folders/1B9WRpkfHn48VfkaHjnErgQ5yb8Vv6PSj?usp=drive_link) and put into /data/NYC or /data/DC.


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
