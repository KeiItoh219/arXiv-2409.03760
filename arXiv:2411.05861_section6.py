# Copyright (c) 2024 Kei Itoh
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import defaultdict

# GPUが利用可能かどうかを確認し、利用可能であればGPUを使用 Check if GPU is available and if so, use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 浮動小数点の精度を倍精度に設定 Set floating point precision to double precision
torch.set_default_dtype(torch.float64)

# 全データの結果保存のオンオフ切り替え On/off toggle data results storage
save_all_data_results = False  # Trueで全データの結果を保存、Falseでスキップ True to save results for all data, false to skip

# シード値の設定関数 Seed value setting function
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# シード値の設定 Seed value setting
seed_value = 42
set_seed(seed_value)

# データの正規化関数 Data Normalization Functions
def normalize_data(tensor):
    norm = torch.norm(tensor.view(-1))
    return tensor / norm if norm != 0 else tensor

def get_transform(normalize=True):
    transform_list = [transforms.ToTensor(), transforms.ConvertImageDtype(torch.float64)]
    if normalize:
        transform_list.append(transforms.Lambda(lambda x: normalize_data(x)))
    return transforms.Compose(transform_list)

# データセットをラベルごとに制限する関数 Function to restrict the data set by label
def limit_dataset_by_label(dataset, min_count):
    label_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        if len(label_indices[label]) < min_count:
            label_indices[label].append(idx)
    limited_indices = [idx for indices in label_indices.values() for idx in indices]
    return Subset(dataset, limited_indices)

# MNISTデータの読み込みと前処理 MNIST data loading and preprocessing
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=get_transform(normalize=True))
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=get_transform(normalize=True))
train_dataset = limit_dataset_by_label(train_dataset, min_count=5420)

# データローダーを作成（学習時に各データ毎にヘブ学習させるためバッチサイズを1に設定）Create a data loader (set batch size to 1 in order to hebtrain each data at training time)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ステップ関数の定義 Step function definition
def step_function(x):
    return torch.where(x > 0, torch.tensor(1.0, dtype=torch.float64, device=x.device), torch.tensor(0.0, dtype=torch.float64, device=x.device))

# ニューラルネットワークの定義(隠れ層の数num_hidden_layersを可変的に変更可能) Definition of Neural Network (the number of hidden layers; num_hidden_layers can be changed)
class FlexibleNetwork(torch.nn.Module):
    def __init__(self, input_size=784, hidden_size=784, output_size=784, num_hidden_layers=4, activation_function='relu'):
        super(FlexibleNetwork, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.activation_function = activation_function
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size, dtype=torch.float64) for i in range(num_hidden_layers)]) #全結合層を用いた順伝播構造の定義 Definition of forward propagation structure with all coupling layers
        self.output_layer = torch.nn.Linear(hidden_size, output_size, dtype=torch.float64) #出力層の定義 Output Layer Definition
        self._initialize_weights()
    #重み初期化 weight initialization
    def _initialize_weights(self):
        for layer in self.hidden_layers:
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='relu')
    #活性化関数の追加（reluかstep選択可） Add activation function (relu or step selectable)
    def forward(self, x):
        activations = [x.view(x.size(0), -1)]
        for layer in self.hidden_layers:
            if self.activation_function == 'relu':
                activations.append(torch.relu(layer(activations[-1])))
            elif self.activation_function == 'step':
                activations.append(step_function(layer(activations[-1])))
            else:
                raise ValueError("Invalid activation function. Choose 'relu' or 'step'.")
        output = self.output_layer(activations[-1])
        return activations, output
    #重み更新のためのヘブ学習則の定義 Definition of Hebbian learning rule for weight update
    def update_weights(self, activations, learning_rate=0.000001):
        for i in range(1, self.num_hidden_layers): # 隠れ層間の重みを更新 Update weights between hidden layers
            prev_activation = activations[i].view(-1) # i層の活性化状態 Activation state of the i layer
            next_activation = activations[i + 1].view(-1) # i+1層の活性化状態 Activation state of the i+1 layer

            if prev_activation.dim() != 1 or next_activation.dim() != 1:
                raise RuntimeError("Activations must be 1-D vectors.")
            delta_w = learning_rate * torch.ger(next_activation, prev_activation) #活性化状態同士の行列積 Matrix product of activation states
            self.hidden_layers[i].weight.data += delta_w #重み更新 weight update
            #重みの無制限強化を防ぐ補正 Correction to prevent unlimited enhancement of weights
            total_weight_update = delta_w.abs().sum().item()
            layer_weight_count = self.hidden_layers[i].weight.numel()
            average_weight_update = total_weight_update / layer_weight_count
            self.hidden_layers[i].weight.data -= average_weight_update

# モデルを訓練する関数 Function to train the model
def train_model(loader, model, learning_rate=0.000001):
    model.train()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            activations, _ = model(data)
            model.update_weights(activations, learning_rate)

# 結果を保存する関数 Function to save the result
def save_results(test_loader, models_per_label, hidden_layers, learning_rate, overall_accuracy, accuracy_per_label):
    global overall_total, overall_correct  # グローバル変数を使用
    folder_name = f"E:\\results\\hidden_layers_{hidden_layers}_lr_{learning_rate:.0e}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with open(os.path.join(folder_name, "accuracy_summary.txt"), "w") as summary_file:
        summary_file.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
        for label, accuracy in accuracy_per_label.items():
            summary_file.write(f"Accuracy for Label {label}: {accuracy:.2f}%\n")

    with torch.no_grad():
        for idx, (data, targets) in enumerate(tqdm(test_loader, desc="Processing test data results")):
            data = data.to(device)
            true_label = targets.item()
            norms = {}
            outputs = {}
            valid_data = True  # 有効なデータかどうかのフラグ Flag if data is valid or not

            for label in range(10):
                model = models_per_label[label]
                _, output = model(data)

                # 出力ベクトルとノルムにinfやnanが含まれていないかを確認 Check to see if the force vector and norm contain inf or nan
                if not torch.isfinite(output).all():
                    valid_data = False
                    break

                norm = torch.norm(output, dim=1).item()
                if not math.isfinite(norm):
                    valid_data = False
                    break

                norms[label] = norm
                outputs[label] = output

            total_counts[true_label] += 1
            overall_total += 1

            if not valid_data:
                continue

            # 予測ラベルは最大のノルムを持つラベル Predicted labels are labels with the largest norm
            predicted_label = max(norms, key=norms.get)
            predicted_norm = norms[predicted_label]

            if predicted_label == true_label:
                correct_counts[true_label] += 1
                overall_correct += 1

            # 結果の全保存 Save all results
            if save_all_data_results:
                file_path = os.path.join(folder_name, f"test_data_{idx}_label_{true_label}.txt")
                with open(file_path, "w") as file:
                    file.write(f"True Label: {true_label}\n")
                    file.write(f"Predicted Label: {predicted_label} (Norm: {predicted_norm})\n\n")
                    
                    file.write("Norm Comparisons:\n")
                    for label, norm in norms.items():
                        comparison = "Higher" if label == predicted_label else "Lower"
                        file.write(f"Label {label} Model Norm: {norm} - {comparison}\n")

                    file.write("\nOutput Vectors:\n")
                    for label, output in outputs.items():
                        file.write(f"Label {label} Model Output: {output}\n")

# 各種パラメータ設定 Various parameter settings
hidden_layer_start = 2 #繰り返す最初の隠れ層数 First number of hidden layers
hidden_layer_increment = 1 #隠れ層数の増加刻み Incremental ticks in the number of hidden layers
num_hidden_layer_variants = 14 #隠れ層数を増やす数 Number to increase the hidden layers
learning_rates = [10**(-i) for i in range(1, 9)] #学習率(range(1, 10)で10^-1 ~ 10^-8まで) #Learning rate (range(1, 10) from 10^-1 ~ 10^-8)
hidden_layer_counts = [hidden_layer_start + i * hidden_layer_increment for i in range(num_hidden_layer_variants)]

results = []
#複数の隠れ層数と複数の学習率で繰り返し計算 Iterations with multiple hidden layer counts and multiple learning rates
for num_hidden_layers in hidden_layer_counts:
    for learning_rate in learning_rates:
        print(f"Training models with {num_hidden_layers} hidden layers and learning rate {learning_rate}...")
        models_per_label = {}
         #各ラベルの個別学習NNs Individual training NNs for each label
        for label in range(10):
            set_seed(seed_value)
            model = FlexibleNetwork(num_hidden_layers=num_hidden_layers, activation_function='relu').to(device)
            label_data_loader = DataLoader([data for data in train_dataset if data[1] == label], batch_size=1, shuffle=True)
            train_model(label_data_loader, model, learning_rate=learning_rate)
            models_per_label[label] = model

        correct_counts = {i: 0 for i in range(10)}
        total_counts = {i: 0 for i in range(10)}
        overall_correct = 0
        overall_total = 0
         #全データに対して予測値計算と正答判断 Calculate predictions and determine correct answers for all data
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc=f"Evaluating models with {num_hidden_layers} hidden layers and learning rate {learning_rate}"):
                data = data.to(device)
                targets = targets.to(device)

                magnitudes = []
                valid_data = True
                #各ラベルの出力ベクトルノルム計算 Output vector norm calculation for each label
                for label in range(10):
                    model = models_per_label[label]
                    _, outputs_label = model(data)
                    if not torch.isfinite(outputs_label).all():
                        valid_data = False
                        break
                    magnitudes.append(torch.norm(outputs_label, dim=1).item()) #ノルム計算

                total_counts[targets.item()] += 1
                overall_total += 1
                if not valid_data:
                    continue

                predicted_label = np.argmax(magnitudes) #予測値はノルムが最も大きいラベル Prediction is label with the largest norm

                if predicted_label == targets.item():
                    correct_counts[targets.item()] += 1
                    overall_correct += 1

        accuracy_per_label = {label: (correct_counts[label] / total_counts[label]) * 100 if total_counts[label] > 0 else 0 for label in range(10)}
        overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0

        results.append({
            'hidden_layers': num_hidden_layers,
            'learning_rate': learning_rate,
            'overall_accuracy': overall_accuracy,
            'accuracy_per_label': accuracy_per_label
        })

        save_results(test_loader, models_per_label, num_hidden_layers, learning_rate, overall_accuracy, accuracy_per_label)

graph_folder = 'results\\accuracy_graphs'
os.makedirs(graph_folder, exist_ok=True)

# 正答率をCSVファイルに保存 Save the percentage of correct answers to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(graph_folder, 'accuracy_results.csv'), index=False)

# 全体およびラベル別正答率のグラフ生成 Graph generation of overall and label-specific percentages of correct answers
hidden_layers_array = np.array(hidden_layer_counts)
learning_rates_array = np.array(learning_rates)
overall_accuracy_matrix = np.zeros((len(learning_rates), len(hidden_layer_counts)))

for i, lr in enumerate(learning_rates):
    for j, hl in enumerate(hidden_layer_counts):
        for result in results:
            if result['hidden_layers'] == hl and result['learning_rate'] == lr:
                overall_accuracy_matrix[i, j] = result['overall_accuracy']

plt.figure(figsize=(8, 6))
plt.contourf(hidden_layers_array, learning_rates_array, overall_accuracy_matrix, levels=20, cmap='viridis', vmin=0, vmax=100)
cbar = plt.colorbar()
cbar.set_label('Overall Accuracy (%)', fontsize=12)
plt.xlabel('Number of Hidden Layers', fontsize=14)
plt.ylabel('Learning Rate', fontsize=14)
plt.yscale('log')
plt.savefig(os.path.join(graph_folder, 'overall_accuracy_contour.svg'), format='svg')
plt.close()

for label in range(10):
    label_accuracy_matrix = np.zeros((len(learning_rates), len(hidden_layer_counts)))
    for i, lr in enumerate(learning_rates):
        for j, hl in enumerate(hidden_layer_counts):
            for result in results:
                if result['hidden_layers'] == hl and result['learning_rate'] == lr:
                    label_accuracy_matrix[i, j] = result['accuracy_per_label'][label]

    plt.figure(figsize=(8, 6))
    plt.contourf(hidden_layers_array, learning_rates_array, label_accuracy_matrix, levels=20, cmap='viridis', vmin=0, vmax=100)
    cbar = plt.colorbar()
    cbar.set_label(f'Accuracy for Label {label} (%)', fontsize=12)
    plt.xlabel('Number of Hidden Layers', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.yscale('log')
    plt.savefig(os.path.join(graph_folder, f'label_{label}_accuracy_contour.svg'), format='svg')
    plt.close()
