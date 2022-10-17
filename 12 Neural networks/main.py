# =============================================================================
# Neural networks
# Author: Miha Pompe
# =============================================================================

from fileinput import filename
import tensorflow as tf
# import random
import numpy as np
#from collab_v1.bdt_catboost_higgs import BATCH_SIZE
import data_higgs as dh
import plotting
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
# import scipy.stats as sc
import time
# import pandas as pd
# from scipy import optimize
# np.random.seed(int(time.time()))
import sklearn as sk

# =============================================================================
# Core functions
# =============================================================================

def mnist_nn():
    mnist = tf.keras.datasets.mnist     # Import dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()    # Extract test and train data
    x_train, x_test = x_train / 255.0, x_test / 255.0   # Normalize input data

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
        ])

    # Loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1)

    loss, acc = model.evaluate(x_test,  y_test, verbose=2)  #return [loss, accuracy]

    print(model.summary())
    return


def split_xy(rawdata):
    #split features and labels from data 
    #prepare the data => normalizations !   

    # split 
    data_y=rawdata['hlabel'] # labels only: 0.=bkg, 1.=sig
    data_x=rawdata.drop(['hlabel'], axis=1) # features only
    
    #now prepare the data
    mu = data_x.mean()
    s = data_x.std()
    dmax = data_x.max()
    dmin = data_x.min()

    # normal/standard rescaling 
    #data_x = (data_x - mu)/s

    # scaling to [-1,1] range
    #data_x = -1. + 2.*(data_x - dmin)/(dmax-dmin)

    # scaling to [0,1] range
    data_x = (data_x - dmin)/(dmax-dmin)

    return data_x,data_y


def prepare_data(BATCH_SIZE = 1000, remove = None):
    hdata=dh.load_data()
    data_fnames=hdata['feature_names'].to_numpy()[1:] #drop labels
    n_dims=data_fnames.shape[0]
    # print ("Entries read {} with feature names {}".format(n_dims,data_fnames))

    x_trn,y_trn=split_xy(hdata['train']) # training sample, should split a fraction for testing
    p = 0.9
    N = len(x_trn)
    x_train, x_test,y_train, y_test =  x_trn[:int(p*N)], x_trn[int(p*N):], y_trn[:int(p*N)], y_trn[int(p*N):] #train_test_split(x_trn,y_trn,test_size=0.1) # 10% split
    x_val,y_val=split_xy(hdata['valid']) # independent cross-valid sample

    if remove is not None:    
        x_train.drop(data_fnames[remove], axis=1, inplace=True)
        x_test.drop(data_fnames[remove], axis=1, inplace=True)
        x_val.drop(data_fnames[remove], axis=1, inplace=True)
        n_dims -= 1


    # print("Shapes train:{} and test:{}".format(x_train.shape,x_test.shape))

    #plot distributions
    # plotting.plot_sig_bkg_from_np_arrays(x_train.to_numpy(),y_train.to_numpy(),data_fnames,logy=False)

    # ready the data for TF

    ds_train = tf.data.Dataset.from_tensor_slices((x_train.to_numpy(),y_train.to_numpy()))
    ds_train = ds_train.repeat()
    
    ds_train = ds_train.batch(BATCH_SIZE,drop_remainder=True)

    ds_test = tf.data.Dataset.from_tensor_slices((x_test.to_numpy(),y_test.to_numpy()))
    ds_test = ds_test.repeat()
    ds_test = ds_test.batch(BATCH_SIZE,drop_remainder=True)

    train_steps=int(x_train.shape[0]/BATCH_SIZE)
    test_steps=int(x_test.shape[0]/BATCH_SIZE)
    # print("Steps train:{} and test:{}".format(train_steps,test_steps))
    return n_dims, x_val, y_val, x_train, y_train, x_test, y_test, ds_test, ds_train, train_steps, test_steps


def higgs_dnn(data, filename, increment, epochs=10, BATCH_SIZE = 1000, num_nodes = 50, num_layers = 2):
    n_dims, x_val, y_val, x_train, y_train, x_test, y_test, ds_test, ds_train, train_steps, test_steps = data

    # build a model - a DNN in TF 2.0 
    dnn = tf.keras.models.Sequential()
    for i in range(num_layers):
        dnn.add(tf.keras.layers.Dense(num_nodes, input_dim=n_dims, activation='relu'))
    #dnn.add(tf.keras.layers.Dense(50, input_dim=n_dims, activation='relu'))
    dnn.add(tf.keras.layers.Dense(1, activation='sigmoid')) # output layer/value
    #plot_model(dnn, to_file='dnn_model.png', show_shapes=True)

    dnn.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='binary_crossentropy',
                metrics=['accuracy', 'AUC', 'binary_crossentropy'])

    dnn.summary()

    #optional early stopping
    eval_metric = 'AUC'
    earlystop_callback = EarlyStopping(
            mode='max',
            monitor='val_' + eval_metric,
            patience=5,
            min_delta=0.00001,
            verbose=1
        )
    
    t1 = time.time()
    #run the training
    dnn_model_history = dnn.fit(ds_train,
            epochs=epochs,
            steps_per_epoch=train_steps,
            #callbacks=[earlystop_callback],
            validation_data=ds_test,
            validation_steps=test_steps
        )
    dt = time.time()-t1

    eval = dnn.evaluate(x_val.to_numpy(),  y_val.to_numpy())
    eval_training = dnn.evaluate(x_test.to_numpy(),y_test.to_numpy())

    # loss, acc = dnn.evaluate(x_val.to_numpy(),  y_val.to_numpy())
    # loss_training, acc_training = dnn.evaluate(x_train.to_numpy(),y_train.to_numpy())

    #plot training history
    # print("history values",dnn_model_history.history.keys())
    # plotting.plot_history([('DNN model', dnn_model_history),],key='binary_crossentropy')
    # plotting.plot_history([('DNN model', dnn_model_history),],key='auc')
    # plotting.plot_history([('DNN model', dnn_model_history),],key='accuracy')

    #plot & print results like ROC and score distribution etc...
    y_score=dnn.predict(x_val.to_numpy())[:,0]

    # plotting.plot_roc(y_val,y_score)
    # plotting.plot_score(y_val,y_score)
    # print()
    auc=roc_auc_score(y_val,y_score)
    # print("AUC score: {}".format(auc))

    out = [eval, eval_training, dt, epochs, BATCH_SIZE, num_layers, num_nodes, [y_val, y_score], auc]
    np.save("results/"+filename+"_"+str(increment)+".npy", out)

    return out

def higgs_bdt(data, filename = "bdt", increment = 0, n_est = 200):
    n_dims, x_val, y_val, x_train, y_train, x_test, y_test, ds_test, ds_train, train_steps, test_steps = data
    print(n_est)
    clf = AdaBoostClassifier(
        sk.tree.DecisionTreeClassifier(max_depth=1), 
        n_estimators=n_est, 
        algorithm="SAMME.R", 
        learning_rate=0.5)
    t1 = time.time()
    clf = clf.fit(x_train.to_numpy(), y_train.to_numpy())
    dt = time.time()-t1

    y_score = clf.predict(x_val.to_numpy())
    auc=roc_auc_score(y_val,y_score)
    score = clf.score(x_val.to_numpy(),  y_val.to_numpy())
    score_training = clf.score(x_test.to_numpy(),y_test.to_numpy())
    out = [score, score_training, dt, auc]
    print(out)
    print(f"Training time {dt}")
    np.save(f"results/{filename}_{increment}.npy", out)
    return out


# TODO
# + loss/accuracy vs batch size for training and validaiton set
# + loss/accuracy vs epoch for training and validaiton set
# + ROC for best configuration
# + bar plot: accuracy vs removing one input variable
# + 3d plot: accuracy vs number of nodes vs number of layers
# - accuracy for weirder configurations
# + time


if __name__ == "__main__":
    # mnist_nn()
    BATCH_SIZE = 1000
    data = prepare_data(BATCH_SIZE)
    
    # best nn
    out = higgs_dnn(data, "optimal_dnn", "0", 100, 500, 60, 7)
    print(out)

    # for epochs in [40, 50, 60, 70, 80, 90, 100, 200, 500]:
    #     out = higgs_dnn(data, "epochs", epochs, epochs=epochs)
    #     print(out[2]/epochs)

    # for batch in [ 50, 100, 500, 1000, 5000, 10000]:
    #     data_ = prepare_data(batch)
    #     out = higgs_dnn(data_, "batch", batch, BATCH_SIZE=batch)
    #     print(out[2]/5)

    # for n_layers in range(1, 10):
    #     for n_nodes in range(10, 100, 10):
    #         out = higgs_dnn(data, "layers_nodes", str(n_layers)+"_"+str(n_nodes), num_layers=n_layers, num_nodes=n_nodes)

    # for i in range(1,18):
    #     data_ = prepare_data(remove = i)
    #     out = higgs_dnn(data_, "remove", i)

    # for n in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,75,100,150,200,500]:
    #     higgs_bdt(data, increment=n, n_est=n)
    
