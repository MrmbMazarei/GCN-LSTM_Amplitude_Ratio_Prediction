import os
import sys
import urllib.request

import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import GRU, SimpleRNN, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
import optuna
from skopt import BayesSearchCV

# Set random seed for Python's built-in random module
import random
random.seed(42)

import stellargraph as sg
import sys
#if 'google.colab' in sys.modules:
#    %pip install -q stellargraph[demos]==1.2.1

try:
    sg.utils.validate_notebook_version("1.2.1")
except AttributeError:
    raise ValueError(
        f"This notebook requires StellarGraph version 1.2.1, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
    ) from None

# Set random seed for TensorFlow
tf.random.set_seed(42)
# Set the random seed for reproducibility
np.random.seed(42)


"""## Data

We apply the GCN-LSTM model to the **Los-loop** data. This traffic dataset
contains traffic information collected from loop detectors in the highway of Los Angeles County (Jagadish
et al., 2014).  There are several processed versions of this dataset used by the research community working in Traffic forecasting space.

This demo is based on the preprocessed version of the dataset used by the TGCN paper. It can be directly accessed from there [github repo](https://github.com/lehaifeng/T-GCN/tree/master/data).

This dataset  contains traffic speeds from Mar.1 to Mar.7, 2012 of 207 sensors, recorded every 5 minutes.

In order to use the model, we need:

* A N by N adjacency matrix, which describes the distance relationship between the N sensors,
* A N by T feature matrix, which describes the (f_1, .., f_T) speed records over T timesteps for the N sensors.

A couple of other references for the same data albeit different time length are as follows:

* [DIFFUSION CONVOLUTIONAL RECURRENT NEURAL NETWORK: DATA-DRIVEN TRAFFIC FORECASTING](https://github.com/liyaguang/DCRNN/tree/master/data): This dataset consists of 207 sensors and collect 4 months of data ranging from Mar 1st 2012 to Jun 30th 2012 for the experiment. It has some missing values.
* [ST-MetaNet: Urban Traffic Prediction from Spatio-Temporal Data Using Deep Meta Learning](https://github.com/panzheyi/ST-MetaNet/tree/master/traffic-prediction). This work uses the DCRNN preprocessed data.

## Loading and preprocessing the data
"""
"""This demo is based on the preprocessed version of the dataset used by the TGCN paper."""





# Number of data points
num_points = 486
Num_Dat = 4
Num_St = Num_Dat*51+6
#Pred_line = 0
epc_number=400
train_rate = 0.8
seq_len = 7
pre_len = 1
corr_threshold = 0.0001 #Thresold for correlation consideration
distance_threshold = 2.344840598 # Adjust this value based on your specific dataset
# Set the window size for the moving average
window_size = 4

# Read the Excel file
file_path1 = r'C:\UConn\USGS\GCN_LSTM\37.Mean_Sand-Different_Lags\Input _to _model.xlsx'
dataaa = pd.read_excel(file_path1)

# Calculate the moving average
data_MA = dataaa.T.rolling(window=7, axis=0, min_periods=1).mean().T

# Display the data
print(data_MA)
Input = np.array(data_MA)

num_nodes, time_len = Input.shape
print("No. of Stations:", num_nodes, "\nNo of timesteps:", time_len)

def train_test_split(data, train_portion):
    time_len = data.shape[1]
    train_size = int(time_len * train_portion)
    train_data = data[:, :train_size]
    test_data = data[:, train_size:]
    return train_data, test_data



train_data, test_data = train_test_split(Input, train_rate)
print("Train data: ", train_data.shape)
print("Test data: ", test_data.shape)

x = np.linspace(0, train_data.shape[1], train_data.shape[1])  # Example x values
y = train_data  # Example y values

# Plot the NumPy array, to show the objective you need to active the next 5 lines:
#plt.plot(x, y[0])
#plt.xlabel('day')
#plt.ylabel('AR')
#plt.title('Plot of Amplitude Ratio')
#plt.show()

def scale_data(train_data, test_data):
    max_speed = train_data.max(axis=1, keepdims=True)
    min_speed = train_data.min(axis=1, keepdims=True)
    train_scaled = (train_data - min_speed) / (max_speed - min_speed)
    test_scaled = (test_data - min_speed) / (max_speed - min_speed)
    return train_scaled, test_scaled, max_speed, min_speed

train_scaled, test_scaled, max_speed, min_speed = scale_data(train_data, test_data)

print(train_scaled.shape)
print(test_scaled.shape)

"""### Sequence data preparation for LSTM

We first need to prepare the data to be fed into an LSTM.
The LSTM model learns a function that maps a sequence of past observations as input to an output observation. As such, the sequence of observations must be transformed into multiple examples from which the LSTM can learn.

To make it concrete in terms of the speed prediction problem, we choose to use 50 minutes of historical speed observations to predict the speed in future, lets say, 1 hour ahead. Hence, we would first  reshape the timeseries data into windows of 10 historical observations for each segment as the input and the speed 60 minutes later is the label we are interested in predicting. We use the sliding window approach to prepare the data. This is how it works:  

* Starting from the beginning of the timeseries, we take the first 10 speed records as the 10 input features and the speed 12 timesteps head (60 minutes) as the speed we want to predict.
* Shift the timeseries by one timestep and take the 10 observations from the current point as the input features and the speed one hour ahead as the output to predict.
* Keep shifting by 1 timestep and picking the 10 timestep window from the current time as input feature and the speed one hour ahead of the 10th timestep as the output to predict, for the entire data.
* The above steps are done for each sensor.

The function below returns the above transformed timeseries data for the model to train on. The parameter `seq_len` is the size of the past window of information. The `pre_len` is how far in the future does the model need to learn to predict.

For this demo:

* Each training observation are seq_len = 10 historical speeds (`seq_len`).
* Each training prediction is the speed pre_len timestep later (`pre_len`).
"""



def sequence_data_preparation(seq_len, pre_len, train_data, test_data):
    trainX, trainY, testX, testY = [], [], [], []

    for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
        a = train_data[:, i : i + seq_len + pre_len]
        trainX.append(a[:, :seq_len])
        trainY.append(a[:, -1])

    for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
        b = test_data[:, i : i + seq_len + pre_len]
        testX.append(b[:, :seq_len])
        testY.append(b[:, -1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, testX, testY

trainX, trainY, testX, testY = sequence_data_preparation(
    seq_len, pre_len, train_scaled, test_scaled
)
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

import torch
from scipy.spatial.distance import cdist
# Set a random seed for reproducibility
torch.manual_seed(42)

#we need to calculate the correlation matrix and set the adjacancy matrix based on the correlation:
Correlation_Matrix = np.corrcoef(dataaa.T, rowvar=False)
np.fill_diagonal(Correlation_Matrix, 0)

# Find the index of the maximum correlation in each row
max_correlation_index = np.argmax(Correlation_Matrix, axis=1)

print("Index of Maximum Correlation in Each Row:")
print(max_correlation_index)

#in this part we determin the correlation bigger that treshould for adjacency matrix.            
# Read the Excel file for pheysical adjacency. we will add these parameters adjacency to correlation adjacency.
file_path2 = r'C:\UConn\USGS\GCN_LSTM\37.Mean_Sand-Different_Lags\Pheysical_Carachter_Model.xlsx'
spatial_coordinates = pd.read_excel(file_path2)

# Print the spatial coordinates
print(spatial_coordinates)
print(spatial_coordinates.shape)

# Specify the column you want to use for distance calculation
column_name = 'mean_sand'

# Convert the column data into arrays
column_data = spatial_coordinates[column_name].values.reshape(-1, 1)

# Calculate the distance between different lines in the specified column
distances = cdist(column_data, column_data, metric='euclidean')

# Create the adjacency matrix based on physical parameter threshold
#adj_matrix = torch.tensor(distances <= distance_threshold, dtype=torch.float)

# Create a boolean tensor for the first condition
condition1 = Correlation_Matrix >= corr_threshold

# Create a boolean tensor for the second condition
condition2 = distances <= distance_threshold

# Create the adjacency matrix using the logical OR operation
adj_matrix = torch.tensor((condition1) | (condition2), dtype=torch.float)

#we need to be sure there is atleast one connection for each station. so we choose the maxcorrelation for each
#station just in case all of them are lower than the treshould.
for i in range(51):
    adj_matrix[i,max_correlation_index[i]] = 1
    adj_matrix[max_correlation_index[i],i] = 1

#Now we are going to add a condition to do not use AR from other station for forecasting AR:
for i in range(51):
    for j in range(51):
        adj_matrix[(i-1)*Num_Dat,(j-1)*Num_Dat] = 0

# Set the diagonal elements to zero (assuming self-loops are not allowed)
adj_matrix.fill_diagonal_(0)

# Print the adjacency matrix
print(adj_matrix.shape)
print(adj_matrix[0,:])


Symb_adj_matrix = adj_matrix[:51, :51]


import networkx as nx
import matplotlib.pyplot as plt

# Convert the adjacency matrix to a NetworkX graph
graph = nx.from_numpy_array(adj_matrix.numpy())
symb_graph = nx.from_numpy_array(Symb_adj_matrix.numpy())

# Plot the graph, To show the Graph you need to active the next two lines:
#nx.draw(symb_graph, with_labels=True)
#plt.show()

########   ******* Hyper parameters tunning: *******************************
from stellargraph.layer import GCN_LSTM
import optuna
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Define your GCN_LSTM class here or import it

def create_model(trial):
    # Suggest the number of layers for GCN and LSTM
    num_gc_layers = trial.suggest_int("num_gc_layers", 1, 5)  # Adjust range as needed
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 5)  # Adjust range as needed

    # Suggest sizes for each GCN layer
    gc_layer_sizes = [trial.suggest_int(f"gc_layer_{i}", 8, 64) for i in range(num_gc_layers)]
    
    # Suggest sizes for each LSTM layer
    lstm_layer_sizes = [trial.suggest_int(f"lstm_layer_{i}", 100, 300) for i in range(num_lstm_layers)]

    # Define possible activation functions
    activation_functions = ["relu", "softsign"]

    # Suggest activation functions for GCN layers
    gc_activations = [trial.suggest_categorical(f"gc_activation_{i}", activation_functions) for i in range(num_gc_layers)]
    
    # Suggest activation functions for LSTM layers
    lstm_activations = [trial.suggest_categorical(f"lstm_activation_{i}", activation_functions) for i in range(num_lstm_layers)]

    # Suggest learning rate
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-1)

    # Define possible loss functions to choose from
    loss_functions = ["mse", "mae", "huber_loss"]  # You can add more loss functions if needed
    loss_function = trial.suggest_categorical("loss_function", loss_functions)

    # Define possible optimizers to choose from
    optimizers = ["adam", "sgd", "rmsprop"]  # You can add other optimizers like Adagrad, Adamax, etc.
    optimizer_name = trial.suggest_categorical("optimizer", optimizers)

    # Configure optimizer with associated parameters
    if optimizer_name == "adam":
        beta_1 = trial.suggest_float("adam_beta_1", 0.8, 0.999)
        beta_2 = trial.suggest_float("adam_beta_2", 0.9, 0.999)
        optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    elif optimizer_name == "sgd":
        momentum = trial.suggest_float("sgd_momentum", 0.0, 0.9)
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_name == "rmsprop":
        rho = trial.suggest_float("rmsprop_rho", 0.85, 0.99)
        optimizer = RMSprop(learning_rate=learning_rate, rho=rho)

    # Suggest batch size
    batch_size = trial.suggest_int("batch_size", 16, 128)  # Define the range for batch size

    # Suggest dropout rate
    dropout = trial.suggest_float("dropout", 0.1, 0.5)  # Adjust range for dropout rate


    # Initialize GCN_LSTM with trial parameters
    gcn_lstm = GCN_LSTM(
        seq_len=seq_len,
        adj=adj_matrix,
        gc_layer_sizes=gc_layer_sizes,
        gc_activations=gc_activations,
        lstm_layer_sizes=lstm_layer_sizes,
        lstm_activations=lstm_activations,
        dropout=dropout,  # Change this to your desired dropout value
    )

    x_input, x_output = gcn_lstm.in_out_tensors()
    model = Model(inputs=x_input, outputs=x_output)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=["mse"])

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train your model
    history = model.fit(
        trainX,
        trainY,
        epochs=50,
        batch_size=batch_size, # Use the tuned batch size
        shuffle=False,
        verbose=1,
        validation_data=(testX, testY),
        callbacks=[early_stopping]
    )

    # Return the best validation loss
    return min(history.history['val_loss'])

def optimize():
    global study  # Use global to modify the study variable
    study = optuna.create_study(direction="minimize")
    study.optimize(create_model, n_trials=100)  # Adjust the number of trials as needed

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    # Optimize and get best trial object
    optimize()
    # After optimization, use best parameters to create final model
    best_trial = study.best_trial

    # Save best hyperparameters to file
    with open("Tuned_Hype_Params.txt", "w") as f:
        f.write(f"Best trial value: {best_trial.value}\n")
        for key, value in best_trial.params.items():
            f.write(f"{key}: {value}\n")


# Continue using gc_layer_sizes in your code as needed
for ic in range(1):

    gcn_lstm = GCN_LSTM(
    seq_len=seq_len,
    adj=adj_matrix,
    gc_layer_sizes=[best_trial.params[f"gc_layer_{i}"] for i in range(best_trial.params["num_gc_layers"])],
    gc_activations=[best_trial.params[f"gc_activation_{i}"] for i in range(best_trial.params["num_gc_layers"])],
    lstm_layer_sizes=[best_trial.params[f"lstm_layer_{i}"] for i in range(best_trial.params["num_lstm_layers"])],
    lstm_activations=[best_trial.params[f"lstm_activation_{i}"] for i in range(best_trial.params["num_lstm_layers"])],
    dropout=best_trial.params["dropout"],  # Use the best-tuned dropout value
    )

    x_input, x_output = gcn_lstm.in_out_tensors()
    print(x_output)

    model = Model(inputs=x_input, outputs=x_output)

    # Define the optimizer again using the best parameters
    if best_trial.params["optimizer"] == "adam":
        optimizer = Adam(
            learning_rate=best_trial.params["lr"],
            beta_1=best_trial.params["adam_beta_1"],
            beta_2=best_trial.params["adam_beta_2"]
        )
    elif best_trial.params["optimizer"] == "sgd":
        optimizer = SGD(
            learning_rate=best_trial.params["lr"],
            momentum=best_trial.params["sgd_momentum"]
        )
    elif best_trial.params["optimizer"] == "rmsprop":
        optimizer = RMSprop(
            learning_rate=best_trial.params["lr"],
            rho=best_trial.params["rmsprop_rho"]
        )


    model.compile(optimizer=optimizer, loss=best_trial.params["loss_function"], metrics=["mse"])

    # Define the EarlyStopping callback
    from keras.callbacks import EarlyStopping

    # Define EarlyStopping callback
    #early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    history = model.fit(
       trainX,
       trainY,
       epochs=epc_number,
       batch_size=43,
       shuffle=False,
       verbose=1,
       validation_data=[testX, testY],
       #callbacks=[early_stopping]  # Add the EarlyStopping callback
    )
    model.summary()

    print(
    "Train loss: ",
    history.history["loss"][-1],
    "\nTest loss:",
    history.history["val_loss"][-1],
    )

    sg.utils.plot_history(history)

    ythat = model.predict(trainX)
    yhat = model.predict(testX)


    """## Rescale values

    Rescale the predicted values to the original value range of the timeseries.
    """

    max_speed = max_speed.T
    min_speed = min_speed.T


    def scale_data(train_scaled, test_scaled, max_speed, min_speed):

        # Rescale normalized data to the original type
        train_rescaled = train_scaled * (max_speed - min_speed) + min_speed
        test_rescaled = test_scaled * (max_speed - min_speed) + min_speed

        return train_rescaled, test_rescaled

    train_rescref, test_rescref = scale_data(trainY, testY, max_speed, min_speed)
    train_rescpred, test_rescpred = scale_data(ythat, yhat, max_speed, min_speed)


    #Shift one step to back
    #train_rescref = train_rescref[:-1]
    #test_rescref = test_rescref[:-1]
    #train_rescpred = train_rescpred[1:]
    #test_rescpred = test_rescpred[1:]
    #testX = testX[:-1]



    """## Measuring the performance of the model

    To understand how well the model is performing, we compare it against a naive benchmark.

    1. Naive prediction: using the most recently **observed** value as the predicted value. Note, that albeit being **naive** this is a very strong baseline to beat. Especially, when speeds are recorded at a 5 minutes granularity,  one does not expect many drastic changes within such a short period of time. Hence, for short-term predictions naive is a reasonable good guess.

    ### Naive prediction benchmark (using latest observed value)
    """

    ## Naive prediction benchmark (using previous observed value)

    testnpred = np.array(testX)[
    :, :, -1
    ]  # picking the last speed of the 10 sequence for each segment in each sample
    testnpredc = (testnpred) * (max_speed - min_speed) + min_speed

    ## Performance measures

    seg_mael = []
    seg_masel = []
    seg_nmael = []

    for j in range(testX.shape[-1]):

        seg_mael.append(
            np.mean(np.abs(test_rescref.T[j] - test_rescpred.T[j]))
        )  # Mean Absolute Error for NN
        seg_nmael.append(
            np.mean(np.abs(test_rescref.T[j] - testnpredc.T[j]))
        )  # Mean Absolute Error for naive prediction
        if seg_nmael[-1] != 0:
            seg_masel.append(
                seg_mael[-1] / seg_nmael[-1]
            )  # Ratio of the two: Mean Absolute Scaled Error
        else:
            seg_masel.append(np.NaN)
    print(yhat.shape)

    print("Total (ave) MAE for NN: " + str(np.mean(np.array(seg_mael))))
    print("Total (ave) MAE for naive prediction: " + str(np.mean(np.array(seg_nmael))))
    print(
        "Total (ave) MASE for per-segment NN/naive MAE: "
        + str(np.nanmean(np.array(seg_masel)))
    )
    print(
        "...note that MASE<1 (for a given segment) means that the NN prediction is better than the naive prediction."
    )

    # plot violin plot of MAE for naive and NN predictions
    fig, ax = plt.subplots()
    # xl = minsl

    ax.violinplot(
        list(seg_mael), showmeans=True, showmedians=False, showextrema=False, widths=1.0
    )

    ax.violinplot(
        list(seg_nmael), showmeans=True, showmedians=False, showextrema=False, widths=1.0
    )

    line1 = mlines.Line2D([], [], label="NN")
    line2 = mlines.Line2D([], [], color="C1", label="Instantaneous")

    ax.set_xlabel("Scaled distribution amplitude (after Gaussian convolution)")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Distribution over segments: NN pred (blue) and naive pred (orange)")
    plt.legend(handles=(line1, line2), title="Prediction Model", loc=2)
    plt.show()
    """#### Plot of actual and predicted speeds on a sample sensor"""

    np.savetxt('All_pred.txt', test_rescpred)
    np.savetxt('All_true.txt', test_rescref) 

    # Save a_pred and a_true to a text file
    a_pred = test_rescpred[:, 7*Num_Dat]
    a_true = test_rescref[:, 7*Num_Dat]
    np.savetxt('a_pred.txt', a_pred)
    np.savetxt('a_true.txt', a_true)

    print(test_rescref.shape)
 



    ###in this part we are going to get the results for all stations.
    output_file = f'output_{ic}.txt'
    #output_file = 'output.txt'
    with open(output_file, 'w') as file:
         for C in range(51):
             Pred_line = C*Num_Dat

             a_pred = test_rescpred[:, Pred_line]
             a_true = test_rescref[:, Pred_line]
             b_pred = train_rescpred[:, Pred_line]
             b_true = train_rescref[:, Pred_line]


             import statistics       
             file.write(f"Station: {C}\n")
             # Calculate and write the variance and mean of the true set
             true_variance = statistics.variance(a_true)
             true_mean = statistics.mean(a_true)
             file.write(f"Variance of true set is: {true_variance}\n")
             file.write(f"Mean of true set is: {true_mean}\n")

             a=np.array(a_pred)

             # Calculate and write the variance and mean of the pred set
             pred_variance = np.var(a)
             pred_mean = np.mean(a)
             file.write(f"Variance of pred set is: {pred_variance}\n")
             file.write(f"Mean of pred set is: {pred_mean}\n")

             #Moving average calculation Test:
             import pandas as pd
             import matplotlib.pyplot as plt

             # Create example time series data
             observational_data = a_true
             forecasting_data = a_pred

             # Create pandas Series objects for the time series data
             observational_series = pd.Series(observational_data)
             forecasting_series = pd.Series(forecasting_data)
             # Calculate the moving average for the observational time series
             a_observational_ma = observational_series.rolling(window_size).mean()
             # Calculate the moving average for the forecasting time series
             a_forecasting_ma = forecasting_series.rolling(window_size).mean()

             # Plot the original and moving average time series
             plt.figure(figsize=(16, 8))
             plt.plot(observational_series, label='Observational')
             plt.plot(a_observational_ma, label='Observational Moving Average')
             plt.plot(forecasting_series, label='Forecasting')
             plt.plot(a_forecasting_ma, label='Forecasting Moving Average')
             plt.legend(loc='upper right', prop={'size': 10})
             # Add labels and legend to the plot
             plt.xlabel('Day')
             plt.ylabel('Weekly MA AR')

             # Save the plot to a file
             plt.savefig(f'prediction_MA_plot_Test_{C}.png')

             # Close the plot to release memory
             plt.close()


             # Calculate the correlation between the moving averages
             correlation = a_observational_ma.corr(a_forecasting_ma)
             # Print the correlation value
             file.write(f"Correlation between moving averages Test: {correlation}\n")



             #Moving average calculation Train
             observational_data = b_true
             forecasting_data = b_pred
             # Create pandas Series objects for the time series data
             observational_series = pd.Series(observational_data)
             forecasting_series = pd.Series(forecasting_data)
             # Calculate the moving average for the observational time series
             b_observational_ma = observational_series.rolling(window_size).mean()
             # Calculate the moving average for the forecasting time series
             b_forecasting_ma = forecasting_series.rolling(window_size).mean()

             # Plot the original and moving average time series
             plt.figure(figsize=(16, 8))
             plt.plot(observational_series, label='Observational')
             plt.plot(b_observational_ma, label='Observational Moving Average')
             plt.plot(forecasting_series, label='Forecasting')
             plt.plot(b_forecasting_ma, label='Forecasting Moving Average')
             plt.legend(loc='upper right', prop={'size': 10})
             # Add labels and legend to the plot
             plt.xlabel('Day')
             plt.ylabel('Weekly MA AR')

             # Save the plot to a file
             plt.savefig(f'prediction_MA_plot_Train_{C}.png')

             # Close the plot to release memory
             plt.close()
 
             # Calculate the correlation between the moving averages
             correlation = b_observational_ma.corr(b_forecasting_ma)
             # Print the correlation value
             file.write(f"Correlation between moving averages Train: {correlation}\n")





             # Calculate Metrics for Test
             #print("FOR TEST PERIOD WE HAVE:")
             from sklearn.metrics import mean_squared_error
             MSE = mean_squared_error(a_true, a_pred)
             file.write(f"MSE Test: {MSE}\n")


             from sklearn.metrics import mean_absolute_error
             MAE = mean_absolute_error(a_true, a_pred)
             file.write(f"MAE Test: {MAE}\n")

 
             corr_matrix = numpy.corrcoef(a_true, a_pred)
             corr = corr_matrix[0,1]
             R_sq = corr**2
             file.write(f"R2 Test: {R_sq}\n")


             def calculate_nse(y_true, y_pred):
                # Calculate the mean of the true values
                y_true_mean = np.mean(y_true)

                # Calculate the numerator and denominator of the NSE formula
                numerator = np.sum((y_true - y_pred) ** 2)
                denominator = np.sum((y_true - y_true_mean) ** 2)

                # Calculate NSE
                nse = 1 - (numerator / denominator)

                return nse
             nse = calculate_nse(a_true, a_pred)
             file.write(f"NSE Test: {nse}\n")


             def calculate_pbias(y_true, y_pred):
                # Calculate the numerator and denominator of the PBIAS formula
                numerator = np.sum(y_pred - y_true)
                denominator = np.sum(y_true)

                # Calculate PBIAS
                pbias = (numerator / denominator) * 100

                return pbias
             pbias = calculate_pbias(a_true, a_pred)
             file.write(f"PBIAS Test: {pbias}\n")


             def calculate_mape(actual, predicted):
                return np.mean(np.abs((actual - predicted) / actual)) * 100

             mape = calculate_mape(a_true, a_pred)
             file.write(f"MAPE Test: {mape}\n")


             def calculate_smape(actual, predicted):
                numerator = np.abs(actual - predicted)
                denominator = (np.abs(actual) + np.abs(predicted)) / 2
                smape = np.mean(numerator / denominator) * 100
                return smape

             smape = calculate_smape(a_true, a_pred)
             file.write(f"SMAPE Test: {smape}\n")





             # Calculate Metrics for TRAIN

             #print("FOR TRAIN PERIOD WE HAVE")
             from sklearn.metrics import mean_squared_error
             MSE = mean_squared_error(b_true, b_pred)
             file.write(f"MSE Train: {MSE}\n")


             from sklearn.metrics import mean_absolute_error
             MAE = mean_absolute_error(b_true, b_pred)
             file.write(f"MAE Train: {MAE}\n")


             corr_matrix = numpy.corrcoef(b_true, b_pred)
             corr = corr_matrix[0,1]
             R_sq = corr**2
             file.write(f"R2 Train: {R_sq}\n")


             def calculate_nse(y_true, y_pred):
                # Calculate the mean of the true values
                y_true_mean = np.mean(y_true)

                # Calculate the numerator and denominator of the NSE formula
                numerator = np.sum((y_true - y_pred) ** 2)
                denominator = np.sum((y_true - y_true_mean) ** 2)

                # Calculate NSE
                nse = 1 - (numerator / denominator)

                return nse
             nse = calculate_nse(b_true, b_pred)
             file.write(f"NSE Train: {nse}\n")


             def calculate_pbias(y_true, y_pred):
                # Calculate the numerator and denominator of the PBIAS formula
                numerator = np.sum(y_pred - y_true)
                denominator = np.sum(y_true)

                # Calculate PBIAS
                pbias = (numerator / denominator) * 100

                return pbias
             pbias = calculate_pbias(b_true, b_pred)
             file.write(f"PBIAS Train: {pbias}\n")


             def calculate_mape(actual, predicted):
                return np.mean(np.abs((actual - predicted) / actual)) * 100

             mape = calculate_mape(b_true, b_pred)
             file.write(f"MAPE Train: {mape}\n")


             def calculate_smape(actual, predicted):
                numerator = np.abs(actual - predicted)
                denominator = (np.abs(actual) + np.abs(predicted)) / 2
                smape = np.mean(numerator / denominator) * 100
                return smape

             smape = calculate_smape(b_true, b_pred)
             file.write(f"SMAPE Train: {smape}\n")

  