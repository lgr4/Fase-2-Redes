import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import sys
import pickle
import requests 
import time
import psutil 
import pandas as pd
import os
from datetime import datetime
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

# Classe para rastrear todas as métricas do processo de aprendizado federado
class MetricsTracker:
    def __init__(self, client_id):
        self.client_id = client_id
        self.data = {
            'round': [],
            'timestamp': [],
            # Métricas de dados transmitidos
            'model_download_size_kb': [],
            'parameters_upload_size_kb': [],
            'total_data_transferred_kb': [],
            # Métricas de tempo
            'model_download_time_s': [],
            'training_time_s': [],
            'parameters_upload_time_s': [],
            'total_round_time_s': [],
            # Métricas de recursos computacionais
            'cpu_usage_percent': [],
            'memory_usage_mb': [],
            'peak_memory_mb': []
        }
        
    def add_round_data(self, round_num, download_size, download_time, 
                       upload_size, upload_time, training_time,
                       cpu_usage, memory_usage, peak_memory):
        self.data['round'].append(round_num)
        self.data['timestamp'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Dados transmitidos
        self.data['model_download_size_kb'].append(download_size)
        self.data['parameters_upload_size_kb'].append(upload_size)
        self.data['total_data_transferred_kb'].append(download_size + upload_size)
        
        # Métricas de tempo
        self.data['model_download_time_s'].append(download_time)
        self.data['training_time_s'].append(training_time)
        self.data['parameters_upload_time_s'].append(upload_time)
        self.data['total_round_time_s'].append(download_time + training_time + upload_time)
        
        # Recursos computacionais
        self.data['cpu_usage_percent'].append(cpu_usage)
        self.data['memory_usage_mb'].append(memory_usage)
        self.data['peak_memory_mb'].append(peak_memory)
        
    def save_to_excel(self):
        df = pd.DataFrame(self.data)
        
        # Criar diretório para métricas se não existir
        os.makedirs('metrics', exist_ok=True)
        
        # Nome do arquivo inclui ID do cliente e timestamp
        filename = f'metrics/client_{self.client_id}_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        
        # Salvar para Excel
        df.to_excel(filename, index=False)
        print(f"\nMétricas salvas em: {filename}")
        
        # Também salvar um resumo estatístico
        summary = pd.DataFrame({
            'Métrica': [
                'Dados baixados (média KB)', 
                'Dados enviados (média KB)',
                'Total de dados transferidos (KB)',
                'Tempo de download (média s)',
                'Tempo de treinamento (média s)',
                'Tempo de upload (média s)',
                'Tempo total de rodadas (média s)',
                'Uso de CPU (média %)',
                'Uso de memória (média MB)',
                'Pico de memória (MB)'
            ],
            'Valor': [
                df['model_download_size_kb'].mean(),
                df['parameters_upload_size_kb'].mean(),
                df['total_data_transferred_kb'].sum(),
                df['model_download_time_s'].mean(),
                df['training_time_s'].mean(),
                df['parameters_upload_time_s'].mean(),
                df['total_round_time_s'].mean(),
                df['cpu_usage_percent'].mean(),
                df['memory_usage_mb'].mean(),
                df['peak_memory_mb'].max()
            ]
        })
        
        summary_filename = f'metrics/client_{self.client_id}_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        summary.to_excel(summary_filename, index=False)
        print(f"Resumo estatístico salvo em: {summary_filename}")
        
        return filename

def get_global_model():
    """Busca o modelo global mais recente do servidor."""
    try:
        start_time = time.time()
        response = requests.get(f"{SERVER_URL}/get_model")
        response.raise_for_status()
        download_duration = time.time() - start_time # Medição correta do tempo de download
        
        payload_size = len(response.content) / 1024 # KB
        print(f"Cliente: Tamanho do modelo global recebido: {payload_size:.2f} KB")
        print(f"Cliente: Tempo para baixar modelo global: {download_duration:.4f}s")
        
        parameters = pickle.loads(response.content)
        return parameters, payload_size, download_duration
    except requests.exceptions.RequestException as e:
        print(f"Erro ao conectar com o servidor: {e}")
        return None, 0, 0

def send_trained_parameters(parameters):
    """Envia os parâmetros treinados para o servidor."""
    serialized_params = pickle.dumps(parameters)
    # Medição mais precisa do tamanho do payload
    payload_size = len(serialized_params) / 1024  # KB
    print(f"Cliente: Tamanho dos parâmetros enviados: {payload_size:.2f} KB")

    headers = {'Content-Type': 'application/octet-stream'}
    
    # Medir o tempo de serialização E envio
    start_time = time.time()
    response = requests.post(f"{SERVER_URL}/submit_parameters", data=serialized_params, headers=headers)
    upload_duration = time.time() - start_time
    print(f"Cliente: Tempo para enviar parâmetros: {upload_duration:.4f}s")
    
    return response, payload_size, upload_duration

# --- Lógica Principal do Cliente ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python client.py <client_id>")
        sys.exit(1)

    client_id = int(sys.argv[-1])
    client = Client(client_id)
    
    # Inicializar o rastreador de métricas
    metrics = MetricsTracker(client_id)
    
    # Para rastrear o pico de memória
    peak_memory_usage = 0

    num_rounds = 30

    for round_num in range(num_rounds):
        print(f"\n--- Rodada {round_num + 1}/{num_rounds} ---")
        round_start_time = time.time()

        # 1. Obter modelo global
        print(f"Cliente {client_id}: Solicitando modelo global do servidor...")
        global_params, download_size, download_time = get_global_model()

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
        
        cpu_after = process.cpu_percent(interval=1)  # Medir CPU com intervalo de 1s para mais precisão
        mem_after = process.memory_info().rss / (1024 * 1024) # Memória em MB
        
        cpu_usage = cpu_after - cpu_before
        memory_usage = mem_after - mem_before
        
        # Atualizar pico de memória se necessário
        peak_memory_usage = max(peak_memory_usage, mem_after)

        print(f"Cliente: Tempo de treinamento local: {train_duration:.4f}s")
        print(f"Cliente: Uso de CPU durante treinamento: {cpu_usage:.2f}%")
        print(f"Cliente: Uso de Memória RAM para treinamento: {memory_usage:.2f} MB")

        # 3. Enviar parâmetros atualizados
        print(f"Cliente {client_id}: Enviando parâmetros atualizados para o servidor...")
        response, upload_size, upload_time = send_trained_parameters(updated_parameters)

        if response and response.status_code == 200:
            print(f"Cliente {client_id}: Parâmetros enviados com sucesso.")
        else:
            status = response.status_code if response else "N/A"
            print(f"Cliente {client_id}: Falha ao enviar parâmetros. Status: {status}")
        
        # Adicionar dados da rodada ao rastreador de métricas
        metrics.add_round_data(
            round_num=round_num + 1,
            download_size=download_size,
            download_time=download_time,
            upload_size=upload_size,
            upload_time=upload_time,
            training_time=train_duration,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            peak_memory=peak_memory_usage
        )

        # Aguarda um tempo para a próxima rodada
        print("Aguardando próxima rodada...")
        time.sleep(5)
    
    # Ao final de todas as rodadas salva as métricas em um Excel
    excel_file = metrics.save_to_excel()
    print(f"\nProcesso de treinamento federado concluído. {num_rounds} rodadas completadas.")
    print(f"Análise detalhada disponível em: {excel_file}")