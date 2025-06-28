import torch
import pickle
import os
import threading
from flask import Flask, request, send_file
import io
from model import *
import torchvision.transforms as transforms #
from torchvision.datasets import CIFAR10 #
from torch.utils.data import DataLoader #

# --- Configuração do Servidor Flask ---
app = Flask(__name__)

# --- Variáveis Globais para gerenciar o estado do treinamento ---
num_clients = 3
rounds = 30
current_round = 0
received_params_this_round = []
lock = threading.Lock()

# Inicializa o modelo global
global_net = FederatedNet()
global_parameters = global_net.get_parameters()

# --- Bloco de Avaliação --- #
def load_test_data():
    """Carrega o conjunto de dados de teste do CIFAR-10."""
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128)
    return test_loader

def evaluate_model(parameters, test_loader):
    """Avalia o modelo global atual no conjunto de teste."""
    net = FederatedNet()
    net.apply_parameters(parameters)
    net.eval()  # Coloca o modelo em modo de avaliação

    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():  # Desabilita o cálculo de gradientes para economizar memória e tempo
        for images, labels in test_loader:
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy

# Carrega os dados de teste uma vez quando o servidor inicia
test_loader = load_test_data()
# --- Fim do Bloco de Avaliação ---

@app.route('/get_model', methods=['GET'])
def get_model():
    print("Servidor: Enviando modelo global para um cliente.")
    serialized_params = pickle.dumps(global_parameters)
    return send_file(
        io.BytesIO(serialized_params),
        mimetype='application/octet-stream'
    )

@app.route('/submit_parameters', methods=['POST'])
def submit_parameters():
    global received_params_this_round, global_parameters, current_round

    client_parameters = pickle.loads(request.data)

    with lock:
        print(f"Servidor: Recebeu parâmetros. Total nesta rodada: {len(received_params_this_round) + 1}/{num_clients}")
        received_params_this_round.append(client_parameters)

        if len(received_params_this_round) == num_clients:
            print(f"Servidor: Todos os clientes responderam. Agregando parâmetros para a rodada {current_round + 1}...")

            # Lógica de agregação (FedAVG)
            new_parameters = {name: {'weight': torch.zeros_like(param['weight']), 'bias': torch.zeros_like(param['bias'])} for name, param in global_parameters.items()}
            for client_params in received_params_this_round:
                for name in client_params:
                    new_parameters[name]['weight'] += client_params[name]['weight'] / num_clients
                    new_parameters[name]['bias'] += client_params[name]['bias'] / num_clients
            
            global_parameters = new_parameters
            global_net.apply_parameters(global_parameters)

            #: Avalia o modelo após a agregação
            loss, accuracy = evaluate_model(global_parameters, test_loader)
            print("="*30)
            print(f"Servidor: Rodada {current_round + 1} CONCLUÍDA")
            print(f"Desempenho do Modelo Global: Perda (Loss) = {loss:.4f} | Acurácia (Accuracy) = {accuracy:.2f}%")
            print("="*30)
            
            received_params_this_round = []
            current_round += 1

            if current_round >= rounds:
                print("Servidor: Treinamento global concluído!")
                with open("global_parameters.pkl", 'wb') as f:
                    pickle.dump(global_parameters, f)

    return "Parâmetros recebidos com sucesso!", 200

if __name__ == "__main__":
    # Avalia o modelo inicial (aleatório) para ter uma linha de base
    print("Avaliando o modelo inicial (antes do treinamento)...")
    initial_loss, initial_accuracy = evaluate_model(global_parameters, test_loader)
    print(f"Desempenho Inicial: Perda = {initial_loss:.4f} | Acurácia = {initial_accuracy:.2f}%")
    
    print("\nServidor de Aprendizado Federado iniciado em http://0.0.0.0:5000")
    app.run(host='25.34.238.74', port=5000, debug=False)