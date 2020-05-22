
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, \
	BatchNormalization, ReLU, Flatten, Layer, Input
from tensorflow.keras.regularizers import l2

def residual(x, name, filters, kernel_size, c, upscale=False):
	res = Conv2D(
		filters,
		kernel_size,
		padding='same',
		data_format='channels_first',
		activation='relu',
		use_bias=False,
		kernel_regularizer=l2(c),
		name=name+'_conv0')(x)
	res = BatchNormalization(name=name+'_bn0')(res)
	res = Conv2D(
		filters,
		kernel_size,
		padding='same',
		data_format='channels_first',
		use_bias=False,
		kernel_regularizer=l2(c),
		name=name+'_conv1')(res)
	if upscale:
		x = Conv2D(
			filters,
			1,
			padding='same',
			data_format='channels_first',
			use_bias=False,
			kernel_regularizer=l2(c),
			name=name+'_match')(x)
	x = ReLU()(x + res)
	x = BatchNormalization(name=name+'_bn1')(x)
	return x

def init_model(name, n_layers, filters, head_filters, c, 
			   height, width, ac_dim, depth):
	s = Input(shape=(depth, height, width),
			  dtype='float32', name='state')
	x = residual(s, 'res0', filters, 3, c, upscale=True)
	for i in range(1, n_layers):
		x = residual(x, 'res{}'.format(i), filters, 3, c)

	p = Conv2D(
		head_filters,
		3,
		padding='same',
		data_format='channels_first',
		activation='relu',
		use_bias=False, 
		kernel_regularizer=l2(c),
		name='p_conv')(x)
	p = BatchNormalization(name='p_bn')(p)
	p = Flatten(name='p_flatten')(p)
	p = Dense(ac_dim, activation='softmax',
		kernel_regularizer=l2(c), name='policy')(p)

	v = Conv2D(
		head_filters * 2,
		3,
		padding='same',
		data_format='channels_first',
		activation='relu',
		use_bias=False,
		kernel_regularizer=l2(c),
		name='v_conv')(x)
	v = BatchNormalization(name='v_bn')(v)
	v = Flatten(name='v_flatten')(v)
	v = Dense(256, activation='relu', kernel_regularizer=l2(c), name='v_dense')(v)
	v = Dense(1, activation='tanh', kernel_regularizer=l2(c), name='value')(v)
	return tf.keras.Model(inputs=s, outputs=[p, v], name=name)

class Agent():
	def __init__(self, path, n_layers, filters, head_filters, c,
			height, width, ac_dim, depth, lr, td, device):
		self.path = path
		self.nnet = init_model(
			'model', n_layers, filters, head_filters, c,
			height, width, ac_dim, depth)
		self.nnet.compile(
			optimizer=keras.optimizers.Adam(lr),
			loss=['categorical_crossentropy', 'MSE'])
		self.nnet.summary()
		self.td = td

	def forward(self, s):
		ps, vs = self.nnet.predict_on_batch(s)
		return ps.numpy().flatten(), vs.numpy().flatten()

	def update(self, s, pi, z):
		return np.mean(self.nnet.train_on_batch(s, {'policy': pi, 'value': z}))

	def save(self, i):
		folder = '{}/{:03d}/model{}'.format(self.path, i,
            '_td' if self.td else '')
		keras.models.save_model(self.nnet, folder)
		print('Model saved.')
		
	def load(self, i):
		folder = '{}/{:03d}/model{}'.format(self.path, i,
            '_td' if self.td else '')
		self.nnet = keras.models.load_model(folder)
		print('Model loaded.')
