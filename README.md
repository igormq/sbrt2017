# Towards an end-to-end speech recognizer for Portuguese using deep neural networks
This repository contains the implementation of the SBRT 2017 paper entitled **Towards an end-to-end speech recognizer for Portuguese using deep neural networks**.

## Training a character-based all-neural Brazilian Portuguese speech recognition model

The model was trained using four datasets: [CSLU Spoltech (LDC2006S16)](https://catalog.ldc.upenn.edu/LDC2006S16), [Sid](#acknowledgements), [VoxForge](http://www.voxforge.org), and [LapsBM1.4]( http://www.laps.ufpa.br/falabrasil/). Only the CSLU dataset is paid.

#### Setting up the (partial) Brazilian Portuguese Speech Dataset (BRSD)

You can download the freely available datasets with the provided script (it may take a while):

```bash
$ cd data; sh download_datasets.sh
```

Next, you can preprocess it into an hdf5 file. Click [here](extras/make_dataset.py) for more information.

```bash
$ python -m extras.make_dataset --parser brsd
```

#### Training the network

You can train the network with the `main.py` script. For more usage information see [this](main.py). To train with the default parameters:

```bash
$ python main.py train --dataset .datasets/brsd/data.h5
```

## Pre-trained model

You may download a pre-trained [sbrt2017](models.py) over the full brsd dataset (including the CSLU dataset):

```bash
$ cd data; sh download_model.sh
```

Also, you can evaluate the model against the **brsd** test set

```bash
$ python main.py eval --model data/models/sbrt2017.h5 --dataset .datasets/brsd/data.h5
```

## Requirements

* Python 2.7
* Numpy
* Scipy
* Pyyaml
* HDF5
* Unidecode
* Librosa
* Tensorflow
* Keras

## Acknowledgements
* [python_speech_features](https://github.com/jameslyons/python_speech_features) for the [audio preprocessing](preprocessing/audio.py)
* [Google Magenta](https://github.com/tensorflow/magenta) for the [hparams](core/hparams.py)
* @robertomest for helping me with everything
* SANTOS, S. C. B.; ALCAIM, A. "Reduced Sets of Subword Units for Continuous Speech Recognition of Portuguese". Electronics Letters, v.36, p.586 588, 2000.

## License
See [LICENSE](LICENSE) for more information
