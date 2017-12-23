# coding: utf-8

import os
import sys
import argparse
import pickle
		
import gzip
import numpy as np

from keras.layers import Dense, Input, Lambda, BatchNormalization, Activation, Dropout, GaussianDropout, Reshape, GlobalAveragePooling1D
from keras.models import Model
from keras.layers import Embedding
from keras.regularizers import l2
import keras.backend as K
import keras
import tensorflow as tf

np.random.seed(1337)

parser = argparse.ArgumentParser(description='QA classifier')
parser.add_argument('--train', action='store_true')
parser.add_argument('--data', type=str, default='data.tsv.gz')
parser.add_argument('--embeddings', type=str, default='ruscorpora.emb.gz')
parser.add_argument('--num_epochs', type=int, help = "Epochs to train", default=10)
args = parser.parse_args()

EMBEDDINGS_SIZE=300
EMBEDDINGS_COUNT=400000

MAX_PWORDS=256
MAX_QWORDS=32

BATCH_SIZE = 256
RGL = l2(1e-5)

def load_embeddings():
	word_pos = {}
	embeddings = [np.zeros(EMBEDDINGS_SIZE, dtype='float32')]*2
	with gzip.open(args.embeddings, 'r') as f:
		print >> sys.stderr, 'Loading embedding matrix.'
		pos = 2
		num = f.readline() #first line - sizes
		for line in f:
			word, values = line.split(' ', 1)
			embeddings.append(np.fromstring(values, sep=' ', dtype='float32'))
			word_pos[word.decode('utf8')] = pos
			pos += 1
			if pos > EMBEDDINGS_COUNT:
				break
		print >> sys.stderr, 'Loaded embedding matrix.'
	return word_pos, np.asarray(embeddings)

if args.train:
	Word_id, Embeddings = load_embeddings()
	print Embeddings.shape
	pickle.dump(Word_id, gzip.open("wordids.pckl.gz", "w"))
else:
	Word_id = pickle.load(gzip.open("wordids.pckl.gz"))
	Embeddings = None

def encode_word(word):
	return Word_id.get(word, 1)

def is_good_word(word):
	return word in Word_id	

def parse_text(text):
	return np.fromiter(map(Word_id.get, filter(is_good_word, text.split())), dtype=np.int32)

def frame_text(words, size):
	if words.shape[0] < size:
		return np.pad(words, (0, size - words.shape[0]), mode="constant", constant_values=0)
	else:
		return words[:size]

def load_data(fname):
	questions = []
	answers = []
	with gzip.open(fname) as f:
		for line in f:
			q, a = line.strip().decode('utf8').split('\t')
			questions.append(frame_text(parse_text(q), MAX_QWORDS))
			answers.append(frame_text(parse_text(a), MAX_PWORDS))
	return [np.asarray(questions), np.asarray(answers)]

X_train = load_data(args.data)
Y_train = np.zeros((X_train[0].shape[0], 1, 2))

from keras.layers.pooling import _GlobalPooling1D 
class GlobalMaskedAveragePooling1D(_GlobalPooling1D):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(GlobalMaskedAveragePooling1D, self).__init__(**kwargs)
	def compute_mask(self, x, input_mask=None):
		return input_mask
	def call(self, x, mask=None):
		if mask is None:
			return K.mean(x, axis=1)
		else:
			mask = K.cast(K.expand_dims(mask, axis=-1), K.floatx())
			return K.sum(x * mask, axis=1) / (K.epsilon() + K.sum(mask, axis=1))

def generate_pos(x):
	pos = tf.cumsum(K.ones_like(x), axis=1) - 1
	return K.minimum(pos // 16, 15)

def build_text_model():
	word_input = Input(shape=(None,), dtype='int32')

	emb = [Embeddings] if not Embeddings is None else None
	word_embedding = Embedding(len(Word_id)+2, EMBEDDINGS_SIZE, weights=emb, mask_zero=True, trainable=False)

	words = word_embedding(word_input)
	words = Dropout(0.25)(words)
	
	pos = Lambda(generate_pos)(word_input)
	pos = Embedding(16, 4, embeddings_regularizer=RGL)(pos)
	words = keras.layers.Concatenate()([words, pos])

	words = Dense(128, kernel_initializer='orthogonal', kernel_regularizer=RGL)(words)

	text = GlobalMaskedAveragePooling1D()(words)
	text = GaussianDropout(0.05)(text)
	text = Lambda(lambda x : K.l2_normalize(x, axis=-1))(text)

	return Model(inputs=word_input, outputs=text)

def build_model():
	pword_input = Input(shape=(None,), dtype='int32')
	qword_input = Input(shape=(None,), dtype='int32')

	text_model = build_text_model()

	text = text_model(pword_input)
	query = text_model(qword_input)

	#pack text and query vectors in batch axis
	#result = keras.layers.Concatenate(axis=0)([text, query])	
	result = Lambda(lambda x : K.stack(x, axis=-1))([text, query])	

	return Model(inputs=[pword_input, qword_input], outputs=result), text_model

def calc_distance(y_pred):
	#unpack text and query
	text = y_pred[...,0]
	query = y_pred[...,1]

	#make distance matrix
	t = K.expand_dims(text, axis=1)
	q = K.expand_dims(query, axis=0)
	return K.sqrt(K.sum(tf.squared_difference(t,q), axis=-1))

def loss(y_true, y_pred):
	distance_matrix = calc_distance(y_pred)

	#diag is true distances, others are false (assuming text-query relation is unique)
	true_distance = tf.diag_part(distance_matrix)

	eye = tf.eye(tf.shape(true_distance)[0])
	false_distance = K.min(distance_matrix + eye * 10, axis=-1)

	#ranking loss	
	return K.maximum(0.2 - false_distance + true_distance, 0)

def acc(y_true, y_pred):
	distance_matrix = calc_distance(y_pred)

	#diag is true distances, others are false (assuming text-query relation is unique)
	true_distance = tf.diag_part(distance_matrix)

	eye = tf.eye(tf.shape(true_distance)[0])
	false_distance = K.min(distance_matrix + eye * 10, axis=-1)
	#top 1 accuracy
	return K.cast(false_distance > true_distance, K.floatx())

if args.train:
	model, text_model = build_model()
	model.compile(loss=loss, optimizer='rmsprop', metrics=[acc])
	model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
		epochs=args.num_epochs, verbose=2,
		validation_split=0.25)
	text_model.save_weights("model")

