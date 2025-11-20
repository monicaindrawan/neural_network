# Neural Network Coursework


## Running Part 1

1. Change directory to the root of the project, containing the python source files
2. Run the part 1 python file: `python -u part1_nn_lib`


## Running Part 2

1. Change directory to the root of the project, containing the python source files
2. Run the part 2 python file: `python -u part2_house_value_regression.py`

This will allow all functions to be performed (ie predict, score, etc) on the trained_model.

### To run individual functionalities in part 2

to shuffle the data, run
```python
shuffle(x, y, split=0.8)
``` 
where split is a number between 0 to 1 containing training set split%. If not specified default is 0.8.

The hyperparameters can be adjusted when creating the class, to train the model yourself:
```python
regressor = Regressor(x_train, nb_epoch = 100, hidden_size=128, activation=torch.nn.Sigmoid(), num_hidden_layers=1, lr=0.01)
regressor.fit(x_train, y_train)
```

To evaluate and predict using the model after fitting, run
```python
regressor.predict(x_test)
regressor.score(x_test, y_test)
```

To find a model with the optimal hyperparameter settings, run the function:
``` python
perform_hyperparameter_search(x, y, hyperparams)
```

Where x, y is the training data and hyperparams is a dictionary of hyperparameters to grid search over with keys: {activation layer, neurons in hidden layer, number of hidden layers, learning rate, number of epochs, dropout rate}. This function will then save a pickle file containing the best model. To load the pickle model, run 
```python
trained_model = load_regressor()
```
