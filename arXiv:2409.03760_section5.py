import torch  
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm 
import random
import matplotlib.pyplot as plt
import winsound 
import os

# GPUが利用可能かどうかを確認し、利用可能であればGPUを使用 Check if GPU is available and if so, use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 浮動小数点の精度を倍精度に設定 Set floating point precision to double precision
torch.set_default_dtype(torch.float64)

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
    label_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        if len(label_indices[label]) < min_count:
            label_indices[label].append(idx)
    limited_indices = [idx for indices in label_indices.values() for idx in indices]
    return Subset(dataset, limited_indices)

# MNISTデータの読み込みと前処理 MNIST data loading and preprocessing
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=get_transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=get_transform)
train_dataset = limit_dataset_by_label(train_dataset, min_count=5420)

# データローダーを作成（学習時に各データ毎にヘブ学習させるためバッチサイズを1に設定）Create a data loader (set batch size to 1 in order to hebtrain each data at training time)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 各ラベルごとにデータセットをフィルタリング Filter dataset by each label
label_datasets = {i: [] for i in range(10)}
for data, target in train_dataset:
    label_datasets[target].append((data, target))

# ステップ関数の定義 Step function definition
def step_function(x):
    return torch.where(x > 0, torch.tensor(1.0, dtype=torch.float64, device=x.device), torch.tensor(0.0, dtype=torch.float64, device=x.device))

# ニューラルネットワークの定義(隠れ層の数num_hidden_layersを可変的に変更可能) Definition of Neural Network (the number of hidden layers; num_hidden_layers can be changed)
class FlexibleNetwork(torch.nn.Module):
    def __init__(self, input_size=784, hidden_size=784, output_size=784, num_hidden_layers=3, activation_function='relu'):
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
    def update_weights(self, activations, learning_rate=0.0000001):
        for i in range(1, self.num_hidden_layers):  # 隠れ層間の重みを更新 Update weights between hidden layers
            prev_activation = activations[i].view(-1)  # i層の活性化状態 Activation state of the i layer
            next_activation = activations[i + 1].view(-1)  # i+1層の活性化状態 Activation state of the i+1 layer

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
def train_model(loader, model):
    model.train()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            activations, _ = model(data)
            model.update_weights(activations)

# 出力ベクトルのノルムの大きさを比較し、テキストファイルに保存 Compare norm magnitudes of output vectors and save to text file
def compare_output_magnitude(test_loader, model_all_data, model_untrained, model_uniform_data, label_models):
    comparison_results = {
        'all_vs_label': {i: 0 for i in range(10)},
        'untrained_vs_label': {i: 0 for i in range(10)},
        'uniform_vs_label': {i: 0 for i in range(10)},
        'label_vs_others': {i: {j: 0 for j in range(10) if i != j} for i in range(10)}
    }
    total_counts = {i: 0 for i in range(10)}

    model_all_data.eval()
    model_untrained.eval()
    model_uniform_data.eval()
    for model in label_models.values():
        model.eval()

    # 保存ディレクトリの設定 Save directory settings
    save_dir = "output_comparison_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for idx, (data, targets) in enumerate(tqdm(test_loader, desc="Evaluating test data")):
            data = data.to(device)
            label = targets.item()
            _, outputs_all_data = model_all_data(data)
            _, outputs_untrained = model_untrained(data)
            _, outputs_uniform_data = model_uniform_data(data)
            magnitudes_all_data = torch.norm(outputs_all_data, dim=1) #ノルム化 Norming all_data model
            magnitudes_untrained = torch.norm(outputs_untrained, dim=1) #Norming untrained model
            magnitudes_uniform_data = torch.norm(outputs_uniform_data, dim=1) #Norming uniform model

            model = label_models[label]
            _, outputs_label = model(data)
            magnitudes_label = torch.norm(outputs_label, dim=1)

            total_counts[label] += 1

            comparison_results['all_vs_label'][label] += (magnitudes_label > magnitudes_all_data).sum().item() #ノルムの大きさ比較　入力したデータのノルムが大きければカウントアップ Norm magnitude comparison If the norm of the input data is large, it is counted up.
            comparison_results['untrained_vs_label'][label] += (magnitudes_label > magnitudes_untrained).sum().item()
            comparison_results['uniform_vs_label'][label] += (magnitudes_label > magnitudes_uniform_data).sum().item()

            # 他ラベルモデルとの比較 Comparison with other label models
            other_outputs = {}
            other_magnitudes = {}
            for other_label in range(10):
                if other_label != label:
                    model_other = label_models[other_label]
                    _, outputs_other_label = model_other(data)
                    magnitudes_other_label = torch.norm(outputs_other_label, dim=1)
                    other_outputs[other_label] = outputs_other_label.cpu().numpy()
                    other_magnitudes[other_label] = magnitudes_other_label.item()
                    comparison_results['label_vs_others'][label][other_label] += (magnitudes_label > magnitudes_other_label).sum().item()

            # 結果をテキストファイルに出力 Output results to a text file
            file_path = os.path.join(save_dir, f"test_data_{idx}_label_{label}.txt")
            with open(file_path, "w") as file:
                #比較結果 Comparison Results
                file.write(f"Test data label: {label}\n")
                file.write(f"Base Norm (Label {label} Model): {magnitudes_label.item()}\n")
                file.write(f"All Data Model Norm: {magnitudes_all_data.item()}\n")
                file.write(f"Untrained Model Norm: {magnitudes_untrained.item()}\n")
                file.write(f"Uniform Data Model Norm: {magnitudes_uniform_data.item()}\n")
                for other_label, magnitude in other_magnitudes.items():
                    file.write(f"Label {other_label} Model Norm: {magnitude}\n")

                # 出力ベクトル Output vector
                file.write("\nOutput Vectors:\n")
                file.write(f"Base Model (Label {label}) Output: {outputs_label.cpu().numpy()}\n")
                file.write(f"All Data Model Output: {outputs_all_data.cpu().numpy()}\n")
                file.write(f"Untrained Model Output: {outputs_untrained.cpu().numpy()}\n")
                file.write(f"Uniform Data Model Output: {outputs_uniform_data.cpu().numpy()}\n")
                file.write("Other Label Models Outputs:\n")
                for other_label, other_output in other_outputs.items():
                    file.write(f"  Label {other_label} Model Output: {other_output}\n")

    comparison_ratios = {
        key: {i: (comparison_results[key][i] / total_counts[i]) * 100 if total_counts[i] > 0 else 0 for i in range(10)}
        for key in ['all_vs_label', 'untrained_vs_label', 'uniform_vs_label']
    }

    label_vs_others_ratios = {
        label: {other_label: (comparison_results['label_vs_others'][label][other_label] / total_counts[label]) * 100 if total_counts[label] > 0 else 0 for other_label in comparison_results['label_vs_others'][label]}
        for label in comparison_results['label_vs_others']
    }

    return comparison_ratios, label_vs_others_ratios

# グラフ描画関数 Graph Drawing Functions
def plot_comparison_results(comparison_ratios, label_vs_others_ratios, save_dir="E:\\output_comparison_results\comparison_results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for label in range(10):
        fig, ax = plt.subplots(figsize=(10, 6))

        all_vs_label = comparison_ratios['all_vs_label'][label]
        untrained_vs_label = comparison_ratios['untrained_vs_label'][label]
        uniform_vs_label = comparison_ratios['uniform_vs_label'][label]

        others_vs_label = [label_vs_others_ratios[label][other_label] for other_label in label_vs_others_ratios[label]]

        model_labels = ['All Data', 'Untrained', 'Uniform'] + [f'Label {other}' for other in range(10) if other != label]
        values = [all_vs_label, untrained_vs_label, uniform_vs_label] + others_vs_label

        ax.bar(model_labels, values, color='b', alpha=0.7)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Vector Norm Magnitude Comparison Ratio (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'comparison_label_{label}.svg')
        plt.savefig(save_path, format='svg')
        plt.close()

# 各モデルの作成と学習 Creation and study of each model
set_seed(seed_value)
model_untrained = FlexibleNetwork(activation_function='relu').to(device)

set_seed(seed_value)
model_all_data = FlexibleNetwork(activation_function='relu').to(device)
train_model(train_loader, model_all_data)

set_seed(seed_value)
model_uniform_data = FlexibleNetwork(activation_function='relu').to(device)
uniform_data = []
for label in range(10):
    label_data = label_datasets[label]
    sample_size = len(label_data) // 10
    uniform_data.extend(random.sample(label_data, sample_size))
uniform_loader = DataLoader(uniform_data, batch_size=1, shuffle=True)
train_model(uniform_loader, model_uniform_data)

models_per_label = {}
for label in range(10):
    set_seed(seed_value)
    model = FlexibleNetwork(activation_function='relu').to(device)
    label_data_loader = DataLoader([data for data in train_dataset if data[1] == label], batch_size=1, shuffle=True)
    train_model(label_data_loader, model)
    models_per_label[label] = model

comparison_ratios, label_vs_others_ratios = compare_output_magnitude(test_loader, model_all_data, model_untrained, model_uniform_data, models_per_label)
plot_comparison_results(comparison_ratios, label_vs_others_ratios)

winsound.Beep(1000, 1000)  # 周波数1000Hz、1秒間ビープ音を鳴らす
