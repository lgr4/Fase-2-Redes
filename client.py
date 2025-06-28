import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import sys
import pickle
import requests 
import time
import psutil 
from model import *

class Client:
    def __init__(self, client_id):
        self.client_id = client_id
        self.dataset = self.load_data()

    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        # Assumindo 3 clientes para a divisão por 3. Isso pode ser tornado mais flexível.
        examples_per_client = len(full_dataset) // 3
        start_idx = self.client_id * examples_per_client
        end_idx = start_idx + examples_per_client
        indices = list(range(start_idx, end_idx))
        client_dataset = torch.utils.data.Subset(full_dataset, indices)
        return client_dataset

    def train(self, parameters):
        net = FederatedNet()
        net.apply_parameters(parameters)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
        dataloader = DataLoader(self.dataset, batch_size=128, shuffle=True)

        for epoch in range(3):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

        return net.get_parameters()

# --- Novas Funções de Comunicação ---
SERVER_URL = "http://25.34.238.74:5000" 

def get_global_model():
    """Busca o modelo global mais recente do servidor."""
    try:
        start_time = time.time()
        response = requests.get(f"{SERVER_URL}/get_model")
        response.raise_for_status()
        download_duration = time.time() - start_time # Medição correta do tempo de download
        
        payload_size = len(response.content) # Medição correta do tamanho do payload
        print(f"Cliente: Tamanho do modelo global recebido: {payload_size / 1024:.2f} KB")
        print(f"Cliente: Tempo para baixar modelo global: {download_duration:.4f}s")
        
        parameters = pickle.loads(response.content)
        return parameters
    except requests.exceptions.RequestException as e:
        print(f"Erro ao conectar com o servidor: {e}")
        return None

def send_trained_parameters(parameters):
    """Envia os parâmetros treinados para o servidor."""
    serialized_params = pickle.dumps(parameters)
    # Medição mais precisa do tamanho do payload
    payload_size = len(serialized_params)
    print(f"Cliente: Tamanho dos parâmetros enviados: {payload_size / 1024:.2f} KB")

    headers = {'Content-Type': 'application/octet-stream'}
    
    # Medir o tempo de serialização E envio
    start_time = time.time()
    response = requests.post(f"{SERVER_URL}/submit_parameters", data=serialized_params, headers=headers)
    upload_duration = time.time() - start_time
    print(f"Cliente: Tempo para enviar parâmetros: {upload_duration:.4f}s")
    
    return response

# --- Lógica Principal do Cliente ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python client.py <client_id>")
        sys.exit(1)

    client_id = int(sys.argv[-1])
    client = Client(client_id)

    num_rounds = 10

    for round_num in range(num_rounds):
        print(f"\n--- Rodada {round_num + 1}/{num_rounds} ---")

        # 1. Obter modelo global
        print(f"Cliente {client_id}: Solicitando modelo global do servidor...")
        global_params = get_global_model()

        if global_params is None:
            print(f"Cliente {client_id}: Falha ao obter modelo global. Abortando rodada.")
            time.sleep(5)
            continue
            
        # 2. Treinamento local com medições
        print(f"Cliente {client_id}: Iniciando treinamento local...")
        
        # Medir tempo de processamento e requisitos computacionais
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024) # Memória em MB
        cpu_before = process.cpu_percent(interval=None)
        
        train_start_time = time.time()
        updated_parameters = client.train(global_params)
        train_duration = time.time() - train_start_time
        
        cpu_after = process.cpu_percent(interval=None)
        mem_after = process.memory_info().rss / (1024 * 1024) # Memória em MB

        print(f"Cliente: Tempo de treinamento local: {train_duration:.4f}s")
        print(f"Cliente: Uso de CPU durante treinamento: {cpu_after - cpu_before:.2f}%")
        print(f"Cliente: Uso de Memória RAM para treinamento: {mem_after - mem_before:.2f} MB")

        # 3. Enviar parâmetros atualizados
        print(f"Cliente {client_id}: Enviando parâmetros atualizados para o servidor...")
        response = send_trained_parameters(updated_parameters)

        if response and response.status_code == 200:
            print(f"Cliente {client_id}: Parâmetros enviados com sucesso.")
        else:
            status = response.status_code if response else "N/A"
            print(f"Cliente {client_id}: Falha ao enviar parâmetros. Status: {status}")

        # Aguarda um tempo para a próxima rodada
        print("Aguardando próxima rodada...")
        time.sleep(5)