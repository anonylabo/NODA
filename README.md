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

Our model achieves the following performance on OD flow problem:
 
| Interval(min) | 60 | 45 | 30 | 15 || 60 | 45 | 
| ------------- | -- | -- | -- | -- || -- | -- |
| CrowdNet | 1.05 | 0.88 | 0.65 | 0.41 || 0.057 | 0.048 |
| GTFormer | 0.94 | 0.80 | 0.59 | 0.38 || 0.054 | 0.046 |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
