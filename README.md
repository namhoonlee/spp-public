# A Signal Propagation Perspective for Pruning Neural Networks at Initialization
This repository contains code for the paper [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://arxiv.org/abs/1906.06307) (ICLR 2020).

## Prerequisites

### Dependencies
* tensorflow >= 1.14
* python >= 3.6
* packages in `requirements.txt`

### Datasets
Put the following datasets in your preferred location (e.g., `./data`).
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [Tiny-ImageNet](https://tiny-imagenet.herokuapp.com/)

## Usage
To run the code (MLP-7-linear on MNIST by default):
```sh
$ python main.py --path_data=./data
```
To enforce approximate dynamical isometry while checking signal propagation on network:
```sh
$ python main.py --path_data=./data --check_jsv --enforce_isometry
```
See `main.py` to run with other options.

## Citation
If you use this code for your work, please cite the following:
```
@inproceedings{lee2020signal,
  title={A signal propagation perspective for pruning neural networks at initialization},
  author={Lee, Namhoon and Ajanthan, Thalaiyasingam and Gould, Stephen and Torr, Philip HS},
  booktitle={ICLR},
  year={2020},
}
```

## License
This project is licensed under the MIT License.
See the [LICENSE](https://github.com/namhoonlee/snip-public/blob/master/LICENSE) file for details.
