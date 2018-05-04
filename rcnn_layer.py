import tensorflow.python.platform
import tensorflow.contrib.layers as L
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn

def conv_layer(inputs, num_outputs, kernel_size, stride, normalizer_fn=None, activation_fn=nn.relu, trainable=True, scope='noname'):

	net = slim.conv2d(inputs        = inputs,
		num_outputs   = num_outputs,
		kernel_size   = kernel_size,
		stride        = stride,
		normalizer_fn = normalizer_fn,
		activation_fn = activation_fn,
		trainable     = trainable,
		scope         = scope,
		reuse         = reuse
		)
	return net

def rcnn_layer(inputs, num_outputs, kernel_size, stride, normalizer_fn=None, activation_fn=nn.relu, trainable=True, scope='noname'):

	CL = conv_layer(inputs,
		num_outputs,
		kernel_size,
		stride,
		normalizer_fn,
		activation_fn=None,
		trainable=trainable,
		scope=scope,
		reuse=False)
	state = nn.relu(CL,name='{:s}_activation'.format(scope))

	RL = conv_layer(state,
		num_outputs,
		kernel_size,
		stride=1,
		normalizer_fn=normalizer_fn,
		activation_fn=None,
		trainable=trainable,
		scope='{:s}_recurrent'.format(scope),
		reuse=False)
	state = nn.relu(CL+RL,name='{:s}_recurrent_activation0'.format(scope))

	RL = conv_layer(state,
		num_outputs,
		kernel_size,
		stride=1,
		normalizer_fn=normalizer_fn,
		activation_fn=None,
		trainable=trainable,
		scope='{:s}_recurrent'.format(scope),
		reuse=True)
	state = nn.relu(CL+RL,name='{:s}_recurrent_activation1'.format(scope))

	RL = conv_layer(state,
		num_outputs,
		kernel_size,
		stride=1,
		normalizer_fn=normalizer_fn,
		activation_fn=None,
		trainable=trainable,
		scope='{:s}_recurrent'.format(scope),
		reuse=True)

	state = CL+RL
	if activation_fn:
		state = activation_fn(state,name='{:s}_recurrent_activation2'.format(scope))
		return state








