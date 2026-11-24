# %%
# # Neural Network from Scratch
#
# A two-layer neural network implemented from scratch using **NumPy**, supporting both **regression** (Linear + MSE) and **binary classification** (Sigmoid + Binary Cross-Entropy), benchmarked against `scikit-learn` models.

# %%
import numpy as np

class NeuralNetwork():
    # Set random seed
    np.random.seed(42)

    def __init__(self, features, targets, hidden_units, out_activation="linear", loss_fn="mse"):
        self.targets = targets
        self.features = features
        self.input_units = features.shape[1]
        self.hidden_units = hidden_units
        self.out_activation = out_activation
        self.loss_fn = loss_fn

        # Initialize weights and biases with random values (array of random numbers)
        self.w_input_hidden = np.random.randn(self.input_units, self.hidden_units)
        self.b_hidden = np.random.randn(self.hidden_units)
        self.w_hidden_out = np.random.randn(self.hidden_units, 1)
        self.b_out = np.random.randn(1)

    # Activation function
    def activition(self, z, activation_name):
        if activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif activation_name == 'relu':
            return np.maximum(0, z)
        elif activation_name == 'linear':
            return z
        else:
            raise Exception('Unknown activation function')

    def activition_derivitive(self, z, activation_name):
        if activation_name == 'sigmoid':
            return z * (1 - z)
        elif activation_name == 'relu':
            return np.where(z > 0, 1, 0)
        elif activation_name == 'linear':
            return 1
        else:
            raise Exception('Unknown activation function')

    # Forward Propagation
    def forward_pass(self, features):
        # Input layer
        self.input_layer = features

        # Hidden layer
        self.hidden_layer = self.activition(
            np.dot(self.input_layer, self.w_input_hidden) + self.b_hidden, "relu"
        )

        # Output layer
        self.output_layer = self.activition(
            np.dot(self.hidden_layer, self.w_hidden_out) + self.b_out, self.out_activation
        )

        return self.output_layer

    def error(self, actual, predicted):
        if self.loss_fn == 'mse':
            return np.mean(np.square(actual - predicted))
        elif self.loss_fn == 'cross_entropy':
            return -np.mean(
                actual * np.log(predicted + 0.00001) + (1 - actual) * np.log(1 - predicted + 0.00001)
            )
        else:
            raise Exception('Unknown error function')

    def error_derivitive(self, actual, predicted):
        if self.loss_fn == 'mse':
            return 2 * (predicted - actual) / actual.size
        elif self.loss_fn == 'cross_entropy':
            return (predicted - actual) / (predicted * (1 - predicted + 0.00001) * actual.size)
        else:
            raise Exception('Unknown activation function')

    # Backpropagation
    def backward_pass(self, features, actual, predicted, lr_rate):
        # Output layer
        self.out_delta = self.error_derivitive(actual, predicted) * self.activition_derivitive(
            predicted, self.out_activation
        )
        self.b_out -= lr_rate * np.sum(self.out_delta, axis=0)
        self.w_hidden_out -= lr_rate * np.dot(self.hidden_layer.T, self.out_delta)

        # Hidden layer
        self.hidden_delta = np.dot(self.out_delta, self.w_hidden_out.T) * self.activition_derivitive(
            self.hidden_layer, "relu"
        )
        self.b_hidden -= lr_rate * np.sum(self.hidden_delta, axis=0)
        self.w_input_hidden -= lr_rate * np.dot(features.T, self.hidden_delta)

        # Return weights and biases
        return self.w_input_hidden, self.b_hidden, self.w_hidden_out, self.b_out

    # Training
    def train(self, features, targets, lr_rate, num_epochs):
        for step in range(num_epochs):
            predicted = self.forward_pass(features)
            weights_ih, biases_h, weights_ho, bias_o = self.backward_pass(
                features, targets, predicted, lr_rate
            )
            if step % 500 == 0:
                print('Epoch: {}, Loss: {:.3f}'.format(step, self.error(targets, predicted)))
        print('Trainig complete!')
        return weights_ih, biases_h, weights_ho, bias_o

    # Prediction
    def predict(self, features):
        if self.loss_fn == "mse":
            return self.forward_pass(features)
        elif self.loss_fn == 'cross_entropy':
            return np.where(self.forward_pass(features) > 0.5, 1, 0)

# %%
# ## 1. Regression Experiment

# %%
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

reg_features, reg_targets = make_regression(n_samples=1000, n_features=3, noise=20, random_state=42)
reg_targets = reg_targets.reshape(-1, 1)

train_features, test_features, train_targets, test_targets = train_test_split(
    reg_features, reg_targets, random_state=42
)

feature_scaler = StandardScaler()
train_features = feature_scaler.fit_transform(train_features)
test_features = feature_scaler.transform(test_features)

# %%
custom_nn = NeuralNetwork(
    train_features, train_targets, hidden_units=128, out_activation='linear', loss_fn='mse'
)
weights_ih, biases_h, weights_ho, bias_o = custom_nn.train(
    train_features, train_targets, lr_rate=0.001, num_epochs=5000
)

# %%
# Weights and biases
print('Input-hidden weights:\n', weights_ih)
print('Hidden-output weights:\n', weights_ho)
print('Hidden biases:\n', biases_h)
print('Output bias:\n', bias_o)

# %%
# Accuracy
from sklearn.metrics import r2_score

reg_predictions = custom_nn.predict(test_features)
print('R2 Score: {:.3f}'.format(r2_score(test_targets, reg_predictions)))

# %%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_features, train_targets)
lin_predictions = lin_reg.predict(test_features)
print('R2 Score: {:.3f}'.format(r2_score(test_targets, lin_predictions)))

# %%
# MlpRegressor
from sklearn.neural_network import MLPRegressor

sklearn_mlp_reg = MLPRegressor(
    hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=5000, random_state=42
)
train_targets = train_targets.reshape(-1)
sklearn_mlp_reg.fit(train_features, train_targets)

# %%
train_targets.shape

# %%
mlp_reg_predictions = sklearn_mlp_reg.predict(test_features)
test_targets = test_targets.reshape(-1)
print('R2 Score: {:.3f}'.format(r2_score(test_targets, mlp_reg_predictions)))

# %%
# ## 2. Binary Classification Experiment

# %%
from sklearn.datasets import make_classification

clf_features, clf_targets = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42
)
clf_targets = clf_targets.reshape(-1, 1)

clf_train_x, clf_test_x, clf_train_y, clf_test_y = train_test_split(
    clf_features, clf_targets, random_state=42
)

clf_scaler = StandardScaler()
clf_train_x = clf_scaler.fit_transform(clf_train_x)
clf_test_x = clf_scaler.transform(clf_test_x)

# %%
clf_nn = NeuralNetwork(
    clf_train_x, clf_train_y, hidden_units=128, out_activation='sigmoid', loss_fn='cross_entropy'
)
clf_weights_ih, clf_biases_h, clf_weights_ho, clf_bias_o = clf_nn.train(
    clf_train_x, clf_train_y, lr_rate=0.001, num_epochs=5000
)

# %%
# Weights and biases
print('Input-hidden weights:\n', clf_weights_ih)
print('Hidden-output weights:\n', clf_weights_ho)
print('Hidden biases:\n', clf_biases_h)
print('Output bias:\n', clf_bias_o)

# %%
from sklearn.metrics import accuracy_score

clf_predictions = clf_nn.predict(clf_test_x)
print('Accuracy: {:.3f}'.format(accuracy_score(clf_test_y, clf_predictions)))

# %%
# Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
clf_train_y = clf_train_y.reshape(-1)
log_reg.fit(clf_train_x, clf_train_y)
log_predictions = log_reg.predict(clf_test_x)
print('Accuracy: {:.3f}'.format(accuracy_score(clf_test_y, log_predictions)))

# %%
# MLPClassifier
from sklearn.neural_network import MLPClassifier

sklearn_mlp_clf = MLPClassifier(
    hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=5000, random_state=42
)
sklearn_mlp_clf.fit(clf_train_x, clf_train_y)
mlp_clf_predictions = sklearn_mlp_clf.predict(clf_test_x)
print('Accuracy: {:.3f}'.format(accuracy_score(clf_test_y, mlp_clf_predictions)))