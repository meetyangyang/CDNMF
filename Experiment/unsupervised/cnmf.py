import logging as log
import numpy as np
from sklearn import decomposition
import joblib
from pymf.cnmf import CNMF
# --------------------------------------------------------------

class SklNMF:
	"""
	Wrapper class backed by the scikit-learn package NMF implementation.
	"""
	# def __init__( self, max_iters = 200, init_strategy = "random", random_seed = 0, alpha_W = 0.0, alpha_H = 0.0, l1_ratio = 1.0): #改进
	def __init__( self, max_iters = 200, init_strategy = "random", random_seed = 0, beta_loss="kullback-leibler", solver ="mu", alpha_W = 0.0, alpha_H = 0.0, l1_ratio = 1.0): #改进
		self.max_iters = 200
		self.init_strategy = init_strategy
		self.W = None
		self.H = None
		self.random_seed = random_seed
		self.alpha_W = alpha_W #改进
		self.alpha_H = alpha_H #改进
		self.l1_ratio = l1_ratio #改进
		self.model = None
		self.beta_loss = beta_loss #改进
		self.solver= solver #改进

	def apply( self, X, k = 2):
		"""
		Apply NMF to the specified document-term matrix X.
		"""
		self.W = None
		self.H = None

		## nmf 作为 model
		# model = decomposition.NMF(init=self.init_strategy, n_components = k, max_iter = self.max_iters, random_state = self.random_seed, alpha_W=self.alpha_W, alpha_H=self.alpha_H, l1_ratio=self.l1_ratio, beta_loss = self.beta_loss, solver = self.solver) #改进
		# self.W = model.fit_transform(X)
		# self.H = model.components_	
		# self.model = model

		## cnmf 作为 model
		model2 = CNMF(X, num_bases = k)
		model2.factorize(niter = self.max_iters)
		self.W, self.H = model2.W, model2.H
		self.model = model2

	def rank_terms( self, topic_index, top = -1 ):
		"""
		Return the top ranked terms for the specified topic, generated during the last NMF run.
		"""
		if self.H is None:
			raise ValueError("No results for previous run available")
		# NB: reverse
		top_indices = np.argsort( self.H[topic_index,:] )[::-1]
		# truncate if necessary
		if top < 1 or top > len(top_indices):
			return top_indices
		return top_indices[0:top]

	def generate_partition( self ):
		if self.W is None:
			raise ValueError("No results for previous run available")
		return np.argmax( self.W, axis = 1 ).flatten().tolist()	

# --------------------------------------------------------------

def generate_doc_rankings( W ):
	'''
	Rank document indices, based on values in a W factor matrix produced by NMF.
	'''
	doc_rankings = []
	k = W.shape[1]
	for topic_index in range(k):
		w = np.array( W[:,topic_index] )
		top_indices = np.argsort(w)[::-1]
		doc_rankings.append(top_indices)
	return doc_rankings

def save_nmf_results( out_path, doc_ids, terms, term_rankings, partition, W, H, topic_labels=None ):
	"""
	Save output of NMF using Joblib. Note that we use the scikit-learn bundled version of joblib.
	"""
	# no labels? generate some standard ones
	if topic_labels is None:
		topic_labels = []
		for i in range( len(term_rankings) ):
			topic_labels.append( "C%02d" % (i+1) )
	log.info( "Saving NMF results to %s" % out_path )
	joblib.dump((doc_ids, terms, term_rankings, partition, W, H, topic_labels), out_path ) 


def load_nmf_results( in_path ):
	"""
	Load NMF results using Joblib. Note that we use the scikit-learn bundled version of joblib.
	"""
	(doc_ids, terms, term_rankings, partition, W, H, labels) = joblib.load( in_path )
	return (doc_ids, terms, term_rankings, partition, W, H, labels)

