# This code is part of the paper "Synthetic Time-Series Load Data via Conditional Generative Adversarial Networks" 
# by Andrea Pinceti, Dr. Lalitha Sankar, and Dr. Oliver Kosut
# submitted to the IEEE PES General Meeting 2021

# This sample code shows how to use the trained GENERATOR model to generate syntehtic, week-long load profiles
# The user can select the number of profiles, the season and the type of load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf # Version 1.13

def sample_Z(m, n): 
	return np.random.normal(loc=0,scale=0.2,size=[m, n])

if __name__ == "__main__":

    ## User defined parameters ##
    # Use the following parameters to specify how many profiles to generate, for which season and which type load
    nl = 10 # Total number of individual load profiles to generate
    season = 1 # 0 = winter, 1 = spring, 2 = summer, 3 = fall ; Select for which season you want to generate profiles
    ltype = 1 # 1 = mainly residential, 0 = mainly industrial ; Select for which type of load you want to generate profiles
    plot = 1 # If set to 1, it will plot each generated profile and save it to a .png file


    ## Fixed parameters ##
    # Do not change these values
    n_noise = 25
    n_types = 6
    g_learning_rate = 0.00005
    scaling_factor = 5.606 # This was the max value by which all of trX was scaled before training GAN. Generated profiles need to be scaled back by this factor


    ## Generator
    # load json and create model
    json_file = open('g_model.json', 'r')
    loaded_g_model_json = json_file.read()
    json_file.close()
    loaded_g_model = tf.keras.models.model_from_json(loaded_g_model_json)
    loaded_g_model.load_weights("g_model.h5")
    loaded_g_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=g_learning_rate, decay=g_learning_rate/10))
    print("Loaded gen model from disk")


    ## Generate data 
    z_test = sample_Z(nl, n_noise)
    z_test_ones = np.zeros((nl, n_types))
    z_test_ones[:, season] = 1
    z_test_ones[:,4+ltype] = 1
    t = np.random.randint(2, size=nl)

    x_pred = loaded_g_model.predict([z_test, z_test_ones])
    x_pred = x_pred *scaling_factor*0.5

    np.savetxt('generatedData.csv', np.transpose(x_pred), delimiter=',')

    if plot == 1:
        for i in range(nl):
            fig = plt.figure()
            fig.set_size_inches(7, 3.6)
            plt.plot(x_pred[i,:])
            plt.savefig('generatedProfile'+str(i)+'.png')
            plt.close()


 
