import torch
import torch.nn as nn
import torch.optim as optim
import pickle
#import tqdm
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

class Regressor(BaseEstimator):

    def __init__(self, x = None, nb_epoch=1000, batch_size=500, layer1 = 24, layer2 = 12, layer3 = 6, prob=0.1,
                 learning_rate=0.0001, loss_func=nn.MSELoss()):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Training data set
        self.x = x
        self.label_binarizer = [None] * len(self.x.columns)
        self.normalizer = [MinMaxScaler(), MinMaxScaler()]
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.prob = prob
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.loss_by_epoch = [0 for i in range(self.nb_epoch)]
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.layer1),
            nn.Dropout(p=self.prob),
            nn.ReLU(),
            nn.Linear(self.layer1, self.layer2),
            nn.Dropout(p=self.prob),
            nn.ReLU(),
            nn.Linear(self.layer2, self.layer3),
            nn.ReLU(),
            nn.Linear(self.layer3, self.output_size),
            nn.ReLU()
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        x.reset_index(inplace=True, drop=True)
        # One-Hot Encoding for 'textual data'
        cnt = 0  # record the count of textual data column
        for col in x.columns:
            if x[col].dtypes == 'O':
                if training:
                    self.label_binarizer[cnt] = LabelBinarizer()
                    onehot_col = self.label_binarizer[cnt].fit_transform(x[col].values)
                else:
                    onehot_col = self.label_binarizer[cnt].transform(x[col].values)
                x = pd.concat([x, pd.DataFrame(onehot_col)], axis=1)
                x.drop([col], axis=1, inplace=True)
            cnt += 1
        x.columns = x.columns.astype(str)
        # Handle the missing value
        x = x.fillna(x.mean())
        if y is not None:
            y.reset_index(inplace=True, drop=True)
            y = y.fillna(y.mean())


        # Normalize data
        if (training):
            normalized_x = self.normalizer[0].fit_transform(x)
            if y is not None:
                normalized_y = self.normalizer[1].fit_transform(y)
        else:
            normalized_x = self.normalizer[0].transform(x)
            if y is not None:
                normalized_y = self.normalizer[1].transform(y)

        return torch.Tensor(normalized_x), (torch.Tensor(normalized_y) if y is not None else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # preprocess the training data
        X_train, Y_train = self._preprocessor(x, y=y, training=True)
        # shuffle the training data
        batch_start = torch.arange(0, len(X_train), self.batch_size)
        # for each epoch
        for epoch in range(self.nb_epoch):
            # set to training mode
            self.model.train()
                # for each batch
            for start in batch_start:
                X_batch = X_train[start: start + self.batch_size]
                Y_batch = Y_train[start: start + self.batch_size]
                # forward pass
                Y_pred = self.model(X_batch)
                # compute the loss
                self.loss = self.loss_func(Y_pred, Y_batch)
                self.loss_by_epoch[epoch] = self.loss_func(Y_pred, Y_batch).item()
                # backward pass
                self.optimizer.zero_grad()
                self.loss.backward()
                # update parameters
                self.optimizer.step()
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.model.eval()
        X,_ = self._preprocessor(x, training= False)
        Y_pred = self.model(X).detach().numpy().reshape(-1,1)
        y_pred = self.normalizer[1].inverse_transform(Y_pred)

        return y_pred.reshape(-1,1)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # preprocess the data
        X, Y = self._preprocessor(x, y, training=False)
        Y_pred = self.model(X)
        mse = self.loss_func(Y_pred, Y)

        return mse.item()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x_train, y_train, x_validate, y_validate, nb_epoch=[500], batch_size=[1000],
                                  layer1=[36], layer2=[24], layer3=[12], prob=[0.1], lr=[0.005]):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        - model {regressor} -- an regressor object to be used as the nn model
        - x_train {pd.DataFrame} Raw input array of shape (train_data_size, input_size).
        - y_train {pd.DataFrame} -- Raw output array of shape (train_data_size, 1).
        - x_validate {pd.DataFrame} Raw input array of shape (validate_data_size, input_size).
        - y_validate {pd.DataFrame} -- Raw output array of shape (validate_data_size, 1).
        - nb_epoch {list} -- list of possible epoch numbers
        - batch_size {size} -- list of possible batch size
        - layer1 {list} -- list of possible number of neurons in hidden layer 1;
        - layer2 {list} -- list of possible number of neurons in hidden layer 2;
        - layer3 {list} -- list of possible number of neurons in hidden layer 3;
        - prob {list} -- list of possible dropout probability
        - lr {list} -- list of possible learning rate used in gradient descent
        
    Returns:
        {list} -- list of optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    #tuning_data = pd.DataFrame()
    grid_search_result = []
    grid_vals = [nb_epoch, batch_size, layer1, layer2, layer3, prob, lr]
    products = list(itertools.product(*grid_vals))

    for (nb_epoch, batch_size, layer1, layer2, layer3, prob, lr) in products:
        regressor = Regressor(x_train, nb_epoch = nb_epoch, batch_size = batch_size, layer1 = layer1, layer2 = layer2,
                              layer3 = layer3, prob = prob, learning_rate = lr)
        regressor.fit(x_train, y_train)
        #save_regressor(regressor)
        # Y_predict = regressor.predict(x_validate)
        score = regressor.score(x_validate, y_validate)
        grid_search_result.append({"params":[nb_epoch, batch_size, layer1, layer2, layer3, prob, lr], "model": regressor, "mse": score})
        print(("params: ", nb_epoch, batch_size, layer1, layer2, layer3, prob, lr, " mse:", score))
    #     tuning_data = tuning_data.append({'nb_epoch': nb_epoch, 'batch_size': batch_size,
    #                                       'layer1': layer1, "layer2": layer2, "layer3": layer3,
    #                                       "prob": prob, "lr": lr, "mse": score}, ignore_index=True)
    # tuning_data.to_excel("tuning data.xlsx", index=False, sheet_name='1')
    optimal = min(grid_search_result, key=lambda x: x["mse"])
    print("Optimal mse:", optimal["mse"])
    optimalRegressor = optimal["model"]
    save_regressor(optimalRegressor)
    print("Optimal regressor saved to pickle.")

    return optimal["params"]

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    # Splitting train / validate / test data
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.3, random_state=22)
    x_validate, x_test, y_validate, y_test = train_test_split(
        x_test, y_test, test_size=0.3333, random_state=22)
    print("Train size: ", x_train.shape, y_train.shape)
    print("Validate size: ", x_validate.shape, y_validate.shape)
    print("Test size: ", x_test.shape, y_test.shape)
    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting

    # regressor = Regressor(x_train)
    # regressor.fit(x_train, y_train):
    # save_regressor(regressor)
    #
    # # Error
    # train_error = regressor.score(x_train, y_train)
    # print("\nTraining Regressor error: {}\n".format(train_error))
    # test_error = regressor.score(x_test, y_test)
    # print("\nTesting Regressor error: {}\n".format(test_error))

    prob = [0.1, 0.2, 0.3]
    nb_epoch = [100, 250, 500]
    learning_rate = [0.0001, 0.001, 0.005, 0.01]
    batch_size = [500, 1000, 2000]
    layer1 = [12, 24, 36]
    layer2 = [6, 12, 24]
    layer3 = [3, 6, 12]

    best_params = RegressorHyperParameterSearch(x_train, y_train, x_validate, y_validate, nb_epoch, batch_size,
                                                layer1, layer2, layer3, prob, learning_rate)
    # best_params = RegressorHyperParameterSearch(x_train, y_train, x_validate, y_validate)
    print("\nThe best hyperparameter values are: {}\n".format(best_params))
    best_model = load_regressor()
    test_error = best_model.score(x_test, y_test)
    print("\nTesting Regressor error: {}\n".format(test_error))


if __name__ == "__main__":
    example_main()

