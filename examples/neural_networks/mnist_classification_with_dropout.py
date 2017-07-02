"""
=============================================================================
Performance improvement in MNIST classification with dropout as a regularizer
=============================================================================

Srivastava et al (2014) illustrated performance increase with use of dropout
method as a regularizer. Here, we demonstrate this approach for MNIST data
set.
"""
print(__doc__)

from sklearn.datasets import fetch_mldata
from sklearn.neural_network.multilayer_perceptron_dropout import MLPClassifier
from sklearn.utils import shuffle
import numpy as np

mnist = fetch_mldata("MNIST original")
# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

X_train, y_train = shuffle(X_train, y_train, random_state=1)

max_iter = 100
n_iter = 50
lr = 0.005
n_samples = X_train.shape[0]
batch_size = 128
n_hidden = 512
tol = 1e-18

classifiers = {
	'mlp_standard':
    MLPClassifier(early_stopping=False,hidden_layer_sizes=(n_hidden, n_hidden),
                  alpha=0, max_iter=max_iter,
                    solver='adam', learning_rate='invscaling', verbose=10, tol=tol, random_state=1,
                    learning_rate_init=lr, activation='relu'),
	'mlp_l2_0.0001':
    MLPClassifier(early_stopping=False,hidden_layer_sizes=(n_hidden, n_hidden),
                  alpha=0.0001, max_iter=max_iter,
                    solver='adam', learning_rate='invscaling', verbose=10, tol=tol, random_state=1,
                    learning_rate_init=lr, activation='relu'),
	'mlp_dropout': MLPClassifier(early_stopping=False,
                              hidden_layer_sizes=(n_hidden, n_hidden), alpha=0,
                              max_iter=max_iter,
                solver='adam', learning_rate='invscaling', verbose=10, tol=tol, random_state=1,
                learning_rate_init=lr, activation='relu', dropout=(0, 0.2, 0.2)),
}

#def test_loss(clf, X_test, y_test):
#	X_test, y_test = clf._validate_input(X_test, y_test, incremental=True)
#	y_prob = clf.predict_proba(X_test)
#	loss_func_name = clf.loss
#	loss = LOSS_FUNCTIONS[loss_func_name](y_test, y_prob) 
#	return loss
#
for clf_name in classifiers:
	print("FITTING CLASSIFIER", clf_name)
	classifiers[clf_name].fit(X_train, y_train )
	accuracy_score = classifiers[clf_name].score(X_test, y_test)
	print("Accuracy score", accuracy_score)

#classifier_stats = {}
#for clf_name in classifiers:
#	stats = {'total_fit_time': 0.0,
#		 'time_track': [0],
#		 'accuracy_history': [(0,0)],
#		 'n_train': 0,
#		 'test_loss' : [],
#		 'train_loss' : [],
#		 'accuracy_test' : [],
#		 }
#	classifier_stats[clf_name] = stats
#
#for idx_pass in range(n_iter):
#	for batch_slice in gen_batches(n_samples, batch_size):
#		for clf_name in classifiers:
#			print("FITTING CLASSIFIER", clf_name)
#			classifiers[clf_name].partial_fit(X_train[batch_slice], y_train[batch_slice], classes=np.unique(y_train))
#			accuracy_score = classifiers[clf_name].score(X_test, y_test)
#			print("Accuracy score", accuracy_score)
#			classifier_stats[clf_name]['accuracy_test'].append(accuracy_score)
#			classifier_stats[clf_name]['test_loss'].append(test_loss(classifiers[clf_name], X_test,y_test))
#			classifier_stats[clf_name]['train_loss'].append(test_loss(classifiers[clf_name], X_train[batch_slice],y_train[batch_slice]))
#
#
#plt.figure()
#plt.semilogy(classifier_stats['mlp_standard']['train_loss'], 'r',lw=2, label="Trainining Loss for MLP Standard")
#plt.semilogy(classifier_stats['mlp_l2_0.0001']['train_loss'], 'g', lw=2, label= "Trainining Loss for MLP with L2 reg (a = 0.0001)")
#plt.semilogy(classifier_stats['mlp_dropout']['train_loss'], 'k', lw=2, label= "Trainining Loss for MLP with Dropout (p = 0.5)")
#plt.semilogy(classifier_stats['mlp_standard']['test_loss'], 'r--',lw=2, label="Test Loss for MLP Standard")
#plt.semilogy(classifier_stats['mlp_l2_0.0001']['test_loss'], 'g--', lw=2, label= "Test Loss for MLP with L2 reg (a = 0.0001)")
#plt.semilogy(classifier_stats['mlp_dropout']['test_loss'], 'k--', lw=2, label= "Test Loss for MLP with Dropout (p = 0.5)")
#
#plt.legend(loc="best")
#plt.show()
#
#plt.figure()
#plt.semilogy(classifier_stats['mlp_standard']['accuracy_test'], 'r',lw=2, label="Test accuracy for MLP Standard")
#plt.semilogy(classifier_stats['mlp_l2_0.0001']['accuracy_test'], 'g', lw=2, label= "Test accuracy for MLP with L2 reg (a = 0.0001)")
#plt.semilogy(classifier_stats['mlp_dropout']['accuracy_test'], 'k', lw=2, label= "Test accuracy for MLP with Dropout (p = 0.5)")
#plt.legend(loc="best")
#plt.show()
#
#
#
#
#
















