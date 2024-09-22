import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial
device = torch.device('cuda') if (USE_GPU and torch.cuda.is_available()) else torch.device('cpu')

df = pd.read_csv('X_train_hYV2vs5.csv')
employee_embeddings = df["employee embedding"].apply(json.loads).tolist()
company_embeddings = df["company embedding"].apply(json.loads).tolist()

fused_embeddings = [emp + comp for emp, comp in zip(employee_embeddings, company_embeddings)]

standard_scaler = StandardScaler()
normalized_embeddings = standard_scaler.fit_transform(np.array(fused_embeddings))

dl = pd.read_csv('y_train_Ga8ie3n.csv')
positions = dl["position"]

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(positions)
    
one_hot_labels_poor = np.array(F.one_hot(torch.tensor(labels_encoded, dtype=torch.long, device=device), num_classes=4))

one_hot_labels = np.zeros(one_hot_labels_poor.shape)
labels_order = [0, 2, 3, 1]
for i in range(4):
    one_hot_labels[:, i] = one_hot_labels_poor[:, labels_order[i]]

right_ordered_labels = torch.tensor(np.argmax(one_hot_labels, axis=1), dtype=torch.long, device=device)

pca = PCA(n_components=50)
features_tensor = torch.tensor(pca.fit_transform(normalized_embeddings), dtype=dtype, device=device)

# features_tensor = torch.tensor(fused_embeddings, dtype=dtype, device=device)  # Convert embeddings to a tensor

train_features, val_features, train_labels, val_labels = train_test_split(
            features_tensor, right_ordered_labels, test_size=0.2, shuffle=True # , stratify=right_ordered_labels
        )

import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(QNetwork, self).__init__()
        
        # Combine input size, hidden sizes, and output size
        all_sizes = [input_size] + hidden_sizes + [num_classes]
        
        # Define the layers
        layers = []
        for hidden_idx in range(len(all_sizes) - 1):
            linear_layer = nn.Linear(all_sizes[hidden_idx], all_sizes[hidden_idx + 1])
            nn.init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
            layers.append(linear_layer)
            if hidden_idx < len(all_sizes) - 2:  # Add ReLU for all layers except the last
                #layers.append(nn.BatchNorm1d(all_sizes[hidden_idx + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
        
        # Use Sequential to create the model
        self.model = nn.Sequential(*layers)
        
        # Define the output layer with Softmax
        #self.softmax = nn.Softmax(dim=1)
        #self.softmax = nn.Sigmoid()
        
    def forward(self, x):
        x = self.model(x)  # Pass through the sequential layers
        #x = self.softmax(x)  # Apply Softmax to the final output
        return x
    
def run_val(val_features, val_labels, model):
    loss = 0.
    with torch.no_grad():
            x = torch.tensor(np.hstack((np.zeros((val_features.shape[0], 1)), val_features))).to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = val_labels.to(device=device, dtype=torch.long)
            scores = model.forward(x)
            print(scores)
            # loss += metric(torch.argmax(scores, dim=1), torch.argmax(y, dim=1)).item()
            loss = f1_score(y, scores, average='macro')

    return loss

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer0 = deque(maxlen=buffer_size)
        self.buffer1 = deque(maxlen=buffer_size)
        self.buffer2 = deque(maxlen=buffer_size)
        self.buffer3 = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience, target_class):
        if(target_class==0):
            self.buffer0.append(experience)
        elif(target_class==1):
            self.buffer1.append(experience)
        elif(target_class==2):
            self.buffer2.append(experience)
        elif(target_class==3):
            self.buffer3.append(experience)

    def sample(self):
        experiences0 = random.sample(self.buffer0, k=self.batch_size//4)
        experiences1 = random.sample(self.buffer1, k=2*self.batch_size//4)
        experiences2 = random.sample(self.buffer2, k=self.batch_size//8)
        experiences3 = random.sample(self.buffer3, k=self.batch_size//8)
        experiences = experiences0 + experiences1 + experiences2 + experiences3

        # Convert list of tuples to a NumPy array
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convert lists to NumPy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Convert NumPy arrays to PyTorch tensors
        return (torch.tensor(states, dtype=torch.float),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float),
                torch.tensor(next_states, dtype=torch.float),
                torch.tensor(dones, dtype=torch.float))

    def size(self):
        return min(len(self.buffer0),len(self.buffer1),len(self.buffer2),len(self.buffer3))


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=[512, 512, 512], gamma=0.9, learning_rate=0.001,
                 buffer_size=40000, batch_size=32, target_update=30, epsilon_start=1.0,
                 epsilon_end=0.1, epsilon_decay=0.99, moving_average=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.moving_average = moving_average

        self.qnetwork = QNetwork(state_size, hidden_size, action_size)
        self.target_network = QNetwork(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            self.qnetwork.eval()
            with torch.no_grad():
                return torch.argmax(self.qnetwork(state)).item()
            
    def forward(self, state_set):
        self.qnetwork.eval()
        with torch.no_grad():
            for _ in range(3):
                actions = torch.argmax(self.qnetwork(state_set), dim=1)
                state_set[:, 0] += actions
                state_set[:, 0] = np.minimum(state_set[:, 0], 3)
        return state_set[:, 0]
    
    def inc_steps(self):
        self.steps +=1

    def train(self):
        if self.replay_buffer.size() < self.batch_size//2:
            return
        self.qnetwork.train()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Compute Q targets
        q_targets_next = self.target_network(next_states).max(1)[0].detach()
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute Q expected
        q_expected = self.qnetwork(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss
        loss = nn.MSELoss()(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        #if self.steps % self.target_update == 0:
        #    self.target_network.load_state_dict(self.qnetwork.state_dict())
        for target_param, q_param in zip(self.target_network.parameters(), self.qnetwork.parameters()):
            target_param.data.copy_(self.moving_average * target_param.data + (1 - self.moving_average) * q_param.data)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)


# Example environment setup (you should adapt this to your specific problem)
class SimpleEnvironment:
    def __init__(self, state_size, features, label):
        self.state_size = state_size
        self.current_class = 0
        self.features = features  # Example feature vector
        self.target_class = label  # The correct class for this instance

    def get_state(self):
        return np.concatenate(([self.current_class], self.features))

    def step(self, action):
        if action == 1:  # Move to next class
            self.current_class += 1

        # Reward logic based on the target class
        if self.current_class == self.target_class:
            if(action==1): # High reward for reaching the target class
                reward = 1
            else:
                reward = 1 # Reward for staying in target class
            if(self.current_class == 3):
                done = True
            else:
                done = False
        elif self.current_class > self.target_class:
            reward = -1  # Penalty for exceeding the target class
            done = True
        else:
            done = False
            if(action == 1):
                reward = 1  # Small reward for every move towards the target
            else :
                reward = -1 # Small penalty for every move that isn't the target

        next_state = np.concatenate(([self.current_class], self.features))
        return next_state, reward, done


state_size = 1 + train_features.shape[1]  # Example state size (1 for class + 4 features)
action_size = 2  # Stay or Move
batch_size= 64
agent = DQNAgent(state_size, action_size, batch_size=batch_size)

num_episodes = train_labels.shape[0]  # Number of episodes matches the number of instances
num_epochs = 20

for episode in range(num_episodes):
    env = SimpleEnvironment(state_size, train_features[episode], train_labels[episode])  # Initialize environment
    state = env.get_state()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.replay_buffer.add((state, action, reward, next_state, done), train_labels[episode])
        state = next_state
        total_reward += reward
    if episode % 64 == 0:
        # print(f"Reward = {total_reward}")
        agent.inc_steps()
        agent.train()
    if episode % 1000 == 0:
        print(f"{episode} episodes seen")
        print(f"Validation score : {run_val(val_features, val_labels, agent)}")

print("End of construction of ReplayBuffer")
max_val = run_val(val_features, val_labels, agent)
print(f"Epoch 0 | Validation score : {max_val}")
current_val = 0
max_dict = agent.qnetwork.state_dict()
for epoch in range(num_epochs):
    for _ in range(int(num_episodes/batch_size)):
        agent.inc_steps()
        agent.train()  # Train using mini-batches from replay buffer
    current_val = run_val(val_features, val_labels, agent)
    if(current_val > max_val):
        max_val = current_val
        max_dict = agent.qnetwork.state_dict()
    print(f"Epoch {epoch+1} | Validation score : {current_val}")

agent.qnetwork.load_state_dict(max_dict)


df = pd.read_csv('X_test_jhAIQq9.csv')
employee_embeddings = df["employee embedding"].apply(json.loads).tolist()
company_embeddings = df["company embedding"].apply(json.loads).tolist()
ids = df["id"]

fused_embeddings = [emp + comp for emp, comp in zip(employee_embeddings, company_embeddings)]


features_tensor = torch.tensor(pca.transform(standard_scaler.transform(np.array(fused_embeddings))), dtype=dtype, device=device)
# features_tensor = torch.tensor(fused_embeddings, dtype=dtype, device=device)  # Convert embeddings to a tensor
features_tensor = torch.tensor(np.hstack((np.zeros((features_tensor.shape[0], 1)), features_tensor))).to(device=device, dtype=dtype)
test_labels = agent.forward(features_tensor)

class_names = np.array(["Assistant", "Executive", "Manager", "Director"])
class_names_labels = class_names[test_labels.numpy().astype(np.int32)]

# Create a DataFrame
data = {
    'id': ids,
    'position': class_names_labels
}

output_df = pd.DataFrame(data)

# Add an index column
output_df.reset_index(inplace=True)

# Rename the columns
output_df.columns = ['', 'id', 'position']

# Save to CSV
output_df.to_csv('y_test_dq_network.csv', index=False)

