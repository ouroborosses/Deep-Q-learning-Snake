import random as rd
import copy as cp
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

 
class QNetwork:
    def __init__(self, state_size, action_size): 
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.create_model()

    def create_model(self):
        # Créez un modèle de réseau de neurones avec les couches appropriées
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
    
    def predict(self, state):
        state_array = np.array(state)  
        state_array = state_array.reshape(1, -1)  # Reshape to (1, 26)
        return self.model.predict(state_array)
 
    def update(self, state, action, reward, next_state):
        # Mettez à jour les poids du réseau de neurones
        target = reward + 0.99 * np.max(self.predict(next_state))
        target_f = self.predict(state)
        target_f[0, action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

class QLearning:
    #Epsilon = exploration rate
    def __init__(self, q_network, epsilon=0.1):
        self.q_network = q_network
        self.epsilon = epsilon

    def choose_action(self, state):
        # Choisissez une action en fonction de la valeur Q prédite
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.q_network.action_size)
            #print("Au hasard : ")
            #print(action)
            return action
        else:
            #print("Q-table : ")
            #print(self.q_network.predict(state))
            #print("Choisi dans la q-table : ")
            #print(np.argmax(self.q_network.predict(state)))
            return np.argmax(self.q_network.predict(state))
        
    def train(self, env, episodes):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                if done:
                    break

    def learn(self, state, action, reward, next_state):
        # Mettez à jour la fonction de valeur Q en fonction de l'expérience
        self.q_network.update(state, action, reward, next_state)