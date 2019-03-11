import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from core.data_processor3 import DataLoader
from core.model import Model
from core.eval_metrics import EvalMetrics
#import pdb

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.ylabel('Close (GBPUSD)')
    plt.xlabel('Time (Days)')
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def denorm_transform(p0_vec, n_pred, n_true):
    p_pred = np.multiply(p0_vec, (n_pred+1))
    p_true = np.multiply(p0_vec, (n_true+1))
    return p_pred,p_true

def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('../data', configs['data']['filename']), 
        os.path.join('../data', configs['data']['VIMfile']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    '''
	# in-memory training
	model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)

    '''
    # Out-of memory generative training
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )

    x_test, y_test, p0_vec = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    #predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    predictions = model.predict_point_by_point(x_test)
    pred        = predictions.reshape((predictions.size,1))

    #plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    #plot_results(pred, y_test) #normalised predictions

    # De-normalise & plot 
    p_pred, p_true = denorm_transform(p0_vec, pred, y_test)
    plot_results(p_pred, p_true) #de-normalised, i.e., original fex units

    # Compute evaluation metrics
    assess = EvalMetrics(p_true, p_pred)
    MAE    = assess.get_MAE()
    RMSE   = assess.get_RMSE()
    print("MAE on validation set is: %f" % MAE)
    print("RMSE on validation set is: %f" % RMSE)

if __name__ == '__main__':
    main()
