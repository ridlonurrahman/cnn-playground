# CNN-PLAYGROUND

This github is initially intended for Final Coursework of [Machine Learning Practical](http://www.inf.ed.ac.uk/teaching/courses/mlp/) (MLP) 2016-2017. We used an architecture similar to VGGNet16 for classifying CIFAR-10/100 dataset. We also compared AlexNet Accuracy with VGGNet16.

## Experimental Setup

Weused TensorFlow 1.0 GPU version running on Amazon EC2 p2.xlarge instance(four vCPUs Intel Xeon E5-2686, 61 GB of RAM, and one Nvidia Tesla k80 with 12GB of VRAM.)

## Building The Network

[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) and [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) were designed originally for ImageNet dataset with 224x224 images size as its input. So, We did some adjustment to make them work with CIFAR dataset. Therefore, the architectures used here are not entirely similar to their paper.

### AlexNet
we cannot use the 11x11 convolutional filter because of small 32x32 images input. We use a 5x5 convolutional filter with stride 1 and the very similar max pooling with 3x3 filter size and stride 2 and followed by local response normalisation. We only utilize two convolutional layers because after the second max pooling, the spatial size is already small enough. Then, it is connected by two fully connected layers with 256 hidden units.

### VGGNet
We also cannot use the exactly same as VGG16 design because after the third or fourth max pooling, the spatial size of our dataset is already small. It is useless to subsample the spatial size that cannot be subsampled again. In addition, also decrease the number of neurons in the fully connected layers to 1024 hidden units per layer. For the complete architecture please refers to report.pdf

### Why not GoogLeNet, ResNet, or DenseNet?
(i) Considering the results of the CIFAR dataset in Kaggle, we can achieve very high accuracy (more than 90% in CIFAR-10) using VGGNet. (ii) Unlike ImageNet, CIFAR dataset is relatively small by the number of the images and small by the resolution. Hence, complex architectures such as resnet cannot give its full potential because small 32x32 RGB images do not have many features to be learned. Those architectures perhaps achieve a better accuracy, but only with a small
margin. (iii) The complexity of those architectures are high and training very deep network such as resnet takes much longer time

### Improving The Accuracy
We tried to improve the accuracy by trying to implement [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf), [Batch Normalization](http://proceedings.mlr.press/v37/ioffe15.pdf), and [ELU](https://arxiv.org/abs/1511.07289). We also try an [approach](https://arxiv.org/abs/1412.6806) to replace the max-pooling layer with conventional layer that acts like pooling.

## Result
TL;DR: achieving 90.18% accuracy on CIFAR-10 dataset. This is quick summary from the report.

We don’t expect much from our AlexNet-two-convolutional-layers performance. We try to improve the accuracy by varying the convolutional filter depth. We know that deeper convolutional filter gives a better representation from the squeezed spatial information. Yet, the performance is still significantly outperformed by VGGNet.

We learned two lessons from training VGGNet. The first is the network depth is beneficial for classification accuracy. And the second is the training deeper neural network is harder. We faced vanishing gradient problem when we trained VGG16.

We applied dropout and batch normalisation to address this problem. We choose dropout with probability of 0.25 for our convolutional layers and 0.5 for our fully-connected layers. We found that higher dropout probability in convolutional layers lead to inefficiency as it slows down the training and does
not prevent co-adaptation. Dropout suddenly becomes inefficient when regularizing convolutional layers because it does not have many parameters. But,
with low dropout probability, the network is quite well-regularized. For the fully-connected layers, we choose dropout probability of 0.5 because it results
in equal probability distribution for the ”smaller network” caused by dropping out neurons. After all, Sristava and Hinton recommend 0.25 as the dropout
probabilty for the convolutional layer and 0.5 as the dropout probability for the fully connected layers. Batch normalisation also help to reducing overfitting by addressing internal covariate shift problem. It applies the whitening data to every hidden layers input so the network converges faster. Employing both dropout and batch normalisation, we achieve the validation accuracy of 90.03%.

We have tried to replace the pooling layer with convolutional layers that act like pooling. We do not see any improvement, so we keep pooling in our architecture. The reason why we do not have the improvement is because we use a different architecture with the paper and the convolutional layers that act as pooling layers somehow cannot learn the necessary invariant better than max-pooling. Then, we also tried ELU activation function to replace ReLU. ELU can help to push the mean of unit activation closer to zero, hence, it speeds up the learning process. ELU was able to slightly increase the classification performance in our architecture to 90.65% on validation set, and 90.18% on test set.

## Built With

* [Tensorflow](https://github.com/tensorflow/tensorflow) - Library for computation using data flow graph

## Acknowledgments

* [mlppractical](https://github.com/CSTR-Edinburgh/mlpractical) - MLP course repo
* [cifar_input.py](https://github.com/tensorflow/models/blob/master/resnet/cifar_input.py) - CIFAR data provider
