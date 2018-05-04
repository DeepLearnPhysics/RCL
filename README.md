# RCL
This is a testbed for prototyping/testing Recurrent Convolutional Layer for Recurrent CNN (RCNN).
The implementation is based on one of the first [RCNN paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf)
For clarity, RCL is not quite RNN+CNN (or at least how I would think about it).
It is really CNN but with recurrent information path.
To me, RNN+CNN means a network design that contains both RNN and CNN but implemented separately at a granular level.
RCL on the other hand implements recurrent infomration path for individual convolution layer.

