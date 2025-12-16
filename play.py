import gymnasium as gym
import ale_py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import time

# Enregistrement des environnements
gym.register_envs(ale_py)

# --- CONFIGURATION ---
ENV_NAME = 'ALE/DonkeyKong-v5'
weights_path = "donkey_kong_final.weights.h5"  # Le fichier que tu as téléchargé
SEQ_LEN = 4

# --- 1. LES CLASS IDENTIQUES A L'ENTRAINEMENT ---
class AtariPreprocessing(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(84, 84, 1), dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process(obs), info

    def _process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        frame = frame / 255.0
        return np.expand_dims(frame, axis=-1)

class DuelingDARQN(models.Model):
    def __init__(self, action_size):
        super(DuelingDARQN, self).__init__()
        self.conv1 = layers.TimeDistributed(layers.Conv2D(32, 8, strides=4, activation='relu'))
        self.conv2 = layers.TimeDistributed(layers.Conv2D(64, 4, strides=2, activation='relu'))
        self.conv3 = layers.TimeDistributed(layers.Conv2D(64, 3, strides=1, activation='relu'))
        self.flatten = layers.TimeDistributed(layers.Flatten())
        self.lstm = layers.LSTM(512, return_sequences=True)
        self.attention = layers.MultiHeadAttention(num_heads=2, key_dim=64)
        self.layer_norm = layers.LayerNormalization()
        self.v_dense = layers.Dense(512, activation='relu')
        self.v_out = layers.Dense(1)
        self.a_dense = layers.Dense(512, activation='relu')
        self.a_out = layers.Dense(action_size)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        lstm_out = self.lstm(x)
        attn_out = self.attention(query=lstm_out, value=lstm_out, key=lstm_out)
        context = self.layer_norm(lstm_out + attn_out)
        last_context = context[:, -1, :]
        v = self.v_dense(last_context)
        v = self.v_out(v)
        a = self.a_dense(last_context)
        a = self.a_out(a)
        q = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
        return q

# --- 2. FONCTION POUR JOUER ---
def run_game():
    # render_mode='human' OUVRE LA FENÊTRE DE JEU
    env = gym.make(ENV_NAME, render_mode='human')
    env = AtariPreprocessing(env)
    
    action_size = env.action_space.n
    model = DuelingDARQN(action_size)
    
    # Initialisation dummy pour construire le graphe TensorFlow
    print("Construction du modèle...")
    dummy_input = tf.zeros((1, SEQ_LEN, 84, 84, 1))
    model(dummy_input)
    
    # Chargement des poids
    try:
        model.load_weights(weights_path)
        print("Poids chargés avec succès !")
    except Exception as e:
        print(f"Erreur lors du chargement des poids : {e}")
        return

    # Boucle de jeu
    episodes = 5
    for e in range(episodes):
        obs, _ = env.reset()
        # Initialiser la séquence
        state_seq = np.stack([obs] * SEQ_LEN, axis=0)
        
        done = False
        score = 0
        
        while not done:
            # Prédictions
            # Ajout dim Batch -> (1, Seq, 84, 84, 1)
            state_tensor = tf.convert_to_tensor([state_seq], dtype=tf.float32)
            q_values = model(state_tensor)
            
            # Choix action (Argmax pur, pas d'aléatoire ici)
            action = np.argmax(q_values.numpy()[0])
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Mise à jour séquence pour la prochaine frame
            next_state_seq = np.roll(state_seq, -1, axis=0)
            next_state_seq[-1] = next_obs
            state_seq = next_state_seq
            
            score += reward
            
            # Ralentir un peu pour que ce soit visible à l'œil humain (optionnel)
            time.sleep(0.01)
            
        print(f"Episode {e+1} terminé. Score: {score}")

    env.close()

if __name__ == "__main__":
    run_game()