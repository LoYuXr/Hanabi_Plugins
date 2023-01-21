# Plug-in Modules for the Hanabi Learning Environment


## Instructions

the repository implements our proposed HCIC module and GOIR module, based on the Rainbow DQN architecture.

The Rainbow agent is derived from the [Dopamine framework](https://github.com/google/dopamine) which is based on Tensorflow. We recommend you consult the
[Tensorflow documentation](https://www.tensorflow.org/install)for additional details.

The HCIC module is implemented in `./ToMmodel`, written on PyTorch and trained individually. We adopt [onnx](https://github.com/onnx/tutorials) to transform the model (.pt) into Tensorflow format (.pb). You can refer to [this website](https://towardsdatascience.com/converting-a-simple-deep-learning-model-from-pytorch-to-tensorflow-b6b353351f5d) for the tutorials.
## Dependencies
* scikit-build
* gin-config
* absl-py
* tensorflow-gpu (version 1 or version 2 are both acceptable)
* tf_slim (for tf.__version__ > 1.15.0)
* tensorflow_addons (for tf.__version__ > 1.15.0)
* torch
* onnx == 1.8.0
* numpy

### Installation

First we pre install some dependencies for the backbone model. If you don't have a GPU resource, replace `tensorflow-gpu` with `tensorflow` (see [Tensorflow instructions](https://www.tensorflow.org/install/install_linux)
for details).

This assumes you already installed the learning environment as detailed in the root README.

```
pip install scikit-build gin-config absl-py tensorflow-gpu numpy
```
If you install Tensorflow 2, further installation should be conducted:
```
pip install tf_slim tensorflow_addons
```

For HCIC module, please install:
```
pip3 install torch 
```
To transfer PyTorch model to Tensorflow model, please install:
```
pip install onnx == 1.8.0
```

## Getting Started
We provide our trained HCIC module, tom.onnx, tom3.onnx, and tom4.onnx, which are trained based on 2-player, 3-player and 4-player settings.
First, you should download the onnx-tensorflow folder to the base directory. Then run the following commands for setup.
```
cd onnx-tensorflow # 0.1.6
pip install -e .
```

After that, you can return to this directory and transfer them to Tensorflow models by running:
```
python cvt_onnx_to_pb.py
```
Remember to change the model name according to your needs.

Finally, we can run the RL agent. You can designate the game configuration in `configs/hanabi_rainbow.gin`, where you may change the player numbers and running iterations. Specifically, to change the number of players, please change the code as follows. The entry point to run a Rainbow agent on the Hanabi environment is `train.py`.

| |2|3|4|
|----|----|----|----|
|`create_environment.num_players` in `configs/hanabi_rainbow.gin` |2|3|4|
|Line 8 `num_players` in `ToMmodel/dataset.py`|2|3|4|
|Line 193 `max_discard` in `ToMmodel/dataset.py`|20|20|10|

Assuming you are running from the agent directory `hanabi_learning_environment/agents/rainbow`, you can control the activation of HCIC and GOIR modules by running the following command. Please first change the `path_to_pb` to the path of the `.pb` model corresponding to the number of people. 
```
python -um train  \
  --base_dir=/tmp/hanabi_rainbow \
  --gin_files='configs/hanabi_rainbow.gin' \
  --hcic=True \
  --goir=True \
  --path_to_pb='./tomcuda.pb'
```

All the arguments must be provided.

More generally, most parameters are easily configured using the
[gin configuration framework](https://github.com/google/gin-config).

## Training a new HCIC module

Our HCIC module is implemented and trained under Pytorch, therefore you might need to train the new HCIC module in a Pytorch environment.

### Dependencies
* torch
* numpy
* tqdm

### Getting Started

After you prepared the dependencies and datasets, you can train the HCIC module by running:

```python
python ToMmodel/train.py --batch_size <batch_size> --epochs <epochs> --lr <learning_rate> --look_back <length of action sequence> --data_path <path to the dataset> --save_path <path to save the model> --device <device>
```

The detailed description of the arguments can be found in `train.py`. Specifically, you might need to modify the code when changing the numbers of players.

