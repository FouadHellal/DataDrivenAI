import numpy as np
'''-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_Initialisation des parametres-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_'''
env_L = 10
env_C = 10
epsilon=float(0.8)
Episodes = 10000
actions = [0, 1, 2, 3]
q_values = np.zeros((env_L, env_C, len(actions)))

rec = np.ones((10,10))
# Récompenses :
rec[:, :] = -1  
rec[:,9]=rec[0:6,0]=rec[2,4:9]=rec[5,1:5]=rec[3:5,4]=-10
rec[0,5]=5
rec[9,5]=100

'''-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_Les fcts utilisées-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_'''

def final(i, j):
    if rec[i, j] == -1.:
        return False
    else:
        return True

def start():
    i = np.random.randint(env_L)
    j = np.random.randint(env_C)
    while final(i, j)==True:
        i = np.random.randint(env_L)
        j = np.random.randint(env_C)
    return i, j

def new_action(i, j, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[i, j]) #exploitation
    else:
        return np.random.randint(4) #exploration

def new_location(i, j , indice_action):

    # 0: droite, 1: gauche, 2: bas, 3: haut

    if indice_action == 0 and j < env_C - 1:
        # Action droite (0) : incrémente la colonne si elle n'est pas la dernière
        j += 1
    elif indice_action == 1 and j > 0:
        # Action gauche (1) : décrémente la colonne si elle n'est pas la première
        j -= 1
    elif indice_action == 2 and i < env_L - 1:
        # Action bas (2) : incrémente la ligne si elle n'est pas la dernière
        i += 1
    elif indice_action == 3 and i > 0:
        # Action haut (3) : décrémente la ligne si elle n'est pas la première
        i -= 1

    # Renvoie le nouvel état
    return i, j


def shortest_path(start_L, start_C):
    if final(start_L, start_C):
        return []
    else:
        shortest = []
        shortest.append((start_L, start_C))
        
        while not final(start_L, start_C):
            action_index = new_action(start_L, start_C, 1.)
            
            start_L, start_C = new_location(start_L, start_C, action_index)
            shortest.append((start_L, start_C))
        return shortest
'''-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_Training-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_'''

epsilon = 0.8
DF = 0.9
learning_rate = 0.9

for episode in range(10000):
    i, j = start()

    while not final(i, j):
        action_index = new_action(i, j, epsilon)
        old_i, old_j = i, j
        i, j = new_location(i, j, action_index)

        reward = rec[i, j]
        old_value = q_values[old_i, old_j, action_index]
        temporal_difference = reward + (DF * np.max(q_values[i, j])) - old_value

        new_q_value = old_value + (learning_rate * temporal_difference)
        q_values[old_i, old_j, action_index] = new_q_value

print("Training done")

print(shortest_path(0,5))



# Initialisation des paramètres
epsilon = 0.2  # Taux d'exploration initial
epsilon_increment = 0.0008  # Incrément du taux d'exploration à chaque épisode
DF = 0.9  # Facteur d'actualisation
learning_rate = 0.9  # Taux d'apprentissage

# Boucle d'apprentissage sur un certain nombre d'épisodes
for episode in range(10000):
    epsilon += epsilon_increment  

    i, j = start()

    # Boucle principale d'apprentissage jusqu'à ce que l'état final soit atteint
    while not final(i, j):
        # Choix d'une action en utilisant la politique epsilon-Greedy
        action_index = new_action(i, j, epsilon)

        # Sauvegarde de l'état actuel
        old_i, old_j = i, j

        # Mise à jour de l'état en fonction de l'action choisie
        i, j = new_location(i, j, action_index)

        # Récupération de la récompense pour l'état actuel
        reward = rec[i, j]

        # Récupération de la valeur Q de l'état précédent et calcul de la différence temporelle
        old_value = q_values[old_i, old_j, action_index]
        temporal_difference = reward + (DF * np.max(q_values[i, j])) - old_value

        # Mise à jour de la valeur Q de l'état précédent
        new_q_value = old_value + (learning_rate * temporal_difference)
        q_values[old_i, old_j, action_index] = new_q_value

print("Training complete")
print(shortest_path(1, 1))
