import torch
from part1_nn_lib import Preprocessor
from sklearn.preprocessing import LabelBinarizer
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Regressor(torch.nn.Module):

    def __init__(self, x, nb_epoch = 100, hidden_size=128, activation=torch.nn.Sigmoid(), num_hidden_layers=1, lr=0.01):
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

        # Replace this code with your own
        super().__init__()

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.hidden_size = hidden_size
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.lr = lr
        self.num_hidden_layers = num_hidden_layers
 
        self.input_layer = torch.nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = torch.nn.ModuleList()
        
        for layer in range(self.num_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))

        self.output_layer = torch.nn.Linear(self.hidden_size, self.output_size)
        self.activation_layer = activation
        
        self.train_loss_per_epoch = []
        self.val_loss_per_epoch = []

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
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

        # One hot encoding
        if training:
            self.lb = LabelBinarizer()
            x = x.join(pd.DataFrame(self.lb.fit_transform(x["ocean_proximity"]),
                        columns=self.lb.classes_, 
                        index=x.index))
            x = x.drop(columns=["ocean_proximity"])
        else:
            x = x.join(pd.DataFrame(self.lb.transform(x["ocean_proximity"]),
                        columns=self.lb.classes_, 
                        index=x.index))
            x = x.drop(columns=["ocean_proximity"])

        for col in x.head():
            x[col] = x[col].fillna(x[col].median())

        # Must be numpy array for Preprocesser()
        x = x.to_numpy()

        if y is not None:
            y = y.to_numpy()

        # Normalises data
        if training:
            self.prep_input = Preprocessor(x)
            x = self.prep_input.apply(x)
            if y is not None:
                self.prep_output = Preprocessor(y)
                y = self.prep_output.apply(y)
        else:
            x = self.prep_input.apply(x)
            
        # Corrects type before returning
        x = torch.tensor(x, dtype=torch.float)
        if y is not None:
            y = torch.tensor(y, dtype=torch.float)
        return x, (y if isinstance(y, torch.Tensor) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        x = self.activation_layer(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_layer(layer(x))
        x = self.activation_layer(self.output_layer(x))
        return x

    
    def fit_and_compute_train_val_loss(self, x_train, y_train, x_val, y_val):
        """ Train the model on training data, then compute and store the loss on 
            the training and validation data for each epoch.
            Used for plotting training and testing loss against parameter.
        """
        self.loss_function = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.train_loss_per_epoch = []
        self.val_loss_per_epoch = []

        X, Y = self._preprocessor(x_train, y = y_train, training = True) # Do not forget
        
        for epoch in range(self.nb_epoch):            
            self.optimiser.zero_grad()
            Y_pred = self.forward(X)
            loss = self.loss_function(Y_pred, Y)
            loss.backward()
            self.optimiser.step()
            # Unscale training data
            Y_pred_unscaled = self.prep_output.revert(Y_pred.detach().numpy())
            Y_train_unscaled = self.prep_output.revert(Y.numpy())
            train_loss = torch.sqrt(torch.mean((torch.tensor(Y_pred_unscaled) - torch.tensor(Y_train_unscaled))**2))
            self.train_loss_per_epoch.append(train_loss.item())

            
            val_loss = self.score(x_val, y_val)
            self.val_loss_per_epoch.append(val_loss)
            print(f"Epoch: {epoch+1}/{self.nb_epoch}, Loss: {loss}")

        print("Training Completed", end='\r')
        
        return self
    

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

        self.loss_function = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        for epoch in range(self.nb_epoch):            
            self.optimiser.zero_grad()
            Y_pred = self.forward(X)
            loss = self.loss_function(Y_pred, Y)
            loss.backward()
            self.optimiser.step()
            loss = torch.sqrt(loss)
            print(f"Epoch: {epoch+1}/{self.nb_epoch}, Loss: {loss}")

        print("Training Completed", end='\r')
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

        X, _ = self._preprocessor(x, training = False) # Do not forget

        with torch.inference_mode():
            Y_pred = self.forward(X)

        return self.prep_output.revert(Y_pred.numpy())
    

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

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        with torch.inference_mode():
            Y_pred = self.forward(X)
        Y_pred = self.prep_output.revert(Y_pred.numpy())

        MSE = torch.nn.MSELoss()
        loss = MSE(torch.tensor(Y_pred), torch.tensor(Y))
        
        return torch.sqrt(loss).item()
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

def shuffle(x, y, split = 0.8):
    """ Shuffles x and y into training and test splits

    Args:
        - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
        - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
        - split {int} - Training split ratio

    Returns:
        (pd.dataframe, pd.dataframe, pd.dataframe, pd.dataframe) -- training and test pandas dataframes for x and y.
    """
    split_idx = int(split * len(x))

    indices = np.random.permutation(len(x))
    x_shuffled = x.iloc[indices] 
    y_shuffled = y.iloc[indices]
    x_train = x_shuffled[:split_idx]
    y_train = y_shuffled[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    return x_train, y_train, x_test, y_test


def perform_hyperparameter_search(x, y, hyperparams): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
        - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
        - hyperparams (dict[list]) -- Dictionary containing hyperparameters with each key having an associated value
                                      of a list of parameters to evaluate over.
        
    Returns:
        dict : {activation layer, neurons in hidden layer, number of hidden layers, learning rate}
                Dictionary containing parameters of best model.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    
    # Shuffle the data
    x_train, y_train, x_val, y_val = shuffle(x, y, 0.8)

    least_loss = float("inf")
    
    for nb_epoch in hyperparams["nb_epoch"]:
        for learning_rate in hyperparams["learning_rate"]:
            for num_hidden_layers in hyperparams["num_hidden_layers"]:
                for activation in hyperparams["activation_layer"]:
                    for hidden_size in hyperparams["hidden_size"]:
                        model = Regressor(x_train, 
                                        hidden_size=hidden_size, 
                                        activation=activation, 
                                        num_hidden_layers=num_hidden_layers,
                                        lr=learning_rate,
                                        nb_epoch=nb_epoch)
                        
                        model.fit(x_train, y_train)
                        loss = model.score(x_val, y_val)
                        if loss < least_loss:
                            least_loss = loss
                            best_hyperparam = {"activation_layer": activation, 
                                            "hidden_size": hidden_size, 
                                            "num_hidden_layers": num_hidden_layers,
                                            "learning_rate": learning_rate,
                                            "nb_epoch": nb_epoch}
                            best_model = model

    #retrain the model on the full dataset using the best_model hyperparameters           
    best_model = Regressor(x, 
                        hidden_size=best_hyperparam["hidden_size"], 
                        activation=best_hyperparam["activation_layer"], 
                        num_hidden_layers=best_hyperparam["num_hidden_layers"],
                        lr=best_hyperparam["learning_rate"],
                        nb_epoch=best_hyperparam["nb_epoch"])
    best_model.fit(x, y)
    print(f"Loss for the best model is {least_loss}.") #after retaining the model should be better than this
    print(f"Optimised hyperparameters are {best_hyperparam}.")

    save_regressor(best_model)

    return  best_hyperparam # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def generate_graph_loss_against_nb_epoch(x, y, best_hyperparam):

    x_train, y_train, x_val, y_val = shuffle(x, y, 0.8)
    train_loss_values = []
    val_loss_values = []
    nb_epoch_max = 300
    model = Regressor(x_train, 
                        hidden_size=best_hyperparam["hidden_size"], 
                        activation=torch.nn.ReLU(), 
                        num_hidden_layers=best_hyperparam["num_hidden_layers"],
                        lr=best_hyperparam["learning_rate"],
                        nb_epoch=nb_epoch_max)
    
    
    model.fit_and_compute_train_val_loss(x_train, y_train, x_val, y_val)
        
    train_loss_values = model.train_loss_per_epoch

    val_loss_values = model.val_loss_per_epoch

    plt.plot(list(range(1,nb_epoch_max+1)), train_loss_values, color="b", label="Training loss")
    plt.plot(list(range(1,nb_epoch_max+1)), val_loss_values, color="r", label="Validation loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Training and validation loss against number of epochs")
    plt.legend()
    plt.show()

def generate_graph_loss_against_activation_layer(x, y, best_hyperparam):

    x_train, y_train, x_val, y_val = shuffle(x, y, 0.8)
    train_loss_values = []
    val_loss_values = []
    nb_epoch_max = 300
    model = Regressor(x_train, 
                        hidden_size=best_hyperparam["hidden_size"], 
                        activation=best_hyperparam["actication"], 
                        num_hidden_layers=best_hyperparam["num_hidden_layers"],
                        lr=best_hyperparam["learning_rate"],
                        nb_epoch=best_hyperparam["nb_epoch"])
    
    
    model.fit_and_compute_train_val_loss(x_train, y_train, x_val, y_val)
        
    train_loss_values = model.train_loss_per_epoch

    val_loss_values = model.val_loss_per_epoch

    plt.plot(list(range(1,nb_epoch_max+1)), train_loss_values, color="b", label="Training loss")
    plt.plot(list(range(1,nb_epoch_max+1)), val_loss_values, color="r", label="Validation loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Training and validation loss against number of epochs")
    plt.legend()
    plt.show()       




def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 
    

    # Splitting input and output

    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    x_train, y_train, x_test, y_test = shuffle(x, y, 0.8)


    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print(f"\nRegressor error: {error}\n")

    hyperparams = {"activation_layer": [torch.nn.Sigmoid(), torch.nn.ReLU()],
                   "hidden_size": [64, 128, 256],
                   "num_hidden_layers": [1, 2, 3],
                   "learning_rate": [0.005, 0.01],
                   "nb_epoch": [50, 100, 200]}
    
    best_hyperparam = perform_hyperparameter_search(x_train, y_train, hyperparams)
    # best_hyperparam = {'activation_layer': torch.nn.ReLU(), 'hidden_size': 128, 'num_hidden_layers': 1, 'learning_rate': 0.005, 'nb_epoch': 200}
    generate_graph_loss_against_nb_epoch(x, y, best_hyperparam)
    print(best_hyperparam)

    


def test_pickle():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output


    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    x_train, y_train, x_test, y_test = shuffle(x, y, 0.8)

    trained_model = load_regressor()

    error = trained_model.score(x_test, y_test)
    print(f"\nRegressor error: {error}\n")


if __name__ == "__main__":
    example_main()
    test_pickle()