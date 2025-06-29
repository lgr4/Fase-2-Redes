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

def get_global_model(client_id):
    """Busca o modelo global mais recente do servidor."""
    try:
        start_time = time.time()
        response = requests.get(f"{SERVER_URL}/get_model?client_id={client_id}")
        response.raise_for_status()
        download_duration = time.time() - start_time # Medição correta do tempo de download
        
        # Verificar se a resposta é um erro JSON
        if response.headers.get('content-type') == 'application/json':
            data = response.json()
            if data.get('status') == 'error':
                print(f"Cliente {client_id}: {data.get('message', 'Erro ao obter modelo')}")
                return None, 0, 0
        
        payload_size = len(response.content) / 1024 # KB
        print(f"Cliente {client_id}: Tamanho do modelo global recebido: {payload_size:.2f} KB")
        print(f"Cliente {client_id}: Tempo para baixar modelo global: {download_duration:.4f}s")
        
        parameters = pickle.loads(response.content)
        return parameters, payload_size, download_duration
    except requests.exceptions.RequestException as e:
        print(f"Cliente {client_id}: Erro ao conectar com o servidor: {e}")
        return None, 0, 0

def send_trained_parameters(client_id, parameters):
    """Envia os parâmetros treinados para o servidor."""
    serialized_params = pickle.dumps(parameters)
    # Medição mais precisa do tamanho do payload
    payload_size = len(serialized_params) / 1024  # KB
    print(f"Cliente {client_id}: Tamanho dos parâmetros enviados: {payload_size:.2f} KB")

    headers = {'Content-Type': 'application/octet-stream'}
    
    # Medir o tempo de serialização E envio
    start_time = time.time()
    response = requests.post(f"{SERVER_URL}/submit_parameters?client_id={client_id}", 
                            data=serialized_params, headers=headers)
    upload_duration = time.time() - start_time
    print(f"Cliente {client_id}: Tempo para enviar parâmetros: {upload_duration:.4f}s")
    
    return response, payload_size, upload_duration

def register_client(client_id):
    """Registra o cliente no servidor."""
    try:
        response = requests.post(f"{SERVER_URL}/register", json={"client_id": client_id})
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'success':
            clients_connected = data.get('clients_connected', 0)
            clients_expected = data.get('clients_expected', 0)
            training_started = data.get('training_started', False)
            
            print(f"Cliente {client_id}: Registrado com sucesso. "
                  f"Clientes conectados: {clients_connected}/{clients_expected}")
            
            return training_started
        else:
            print(f"Cliente {client_id}: Falha no registro. Mensagem: {data.get('message', 'Desconhecida')}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Cliente {client_id}: Erro ao conectar com o servidor para registro: {e}")
        return False

def send_heartbeat(client_id):
    """Envia sinal de heartbeat para o servidor."""
    try:
        response = requests.post(f"{SERVER_URL}/heartbeat", json={"client_id": client_id})
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'success':
            return data.get('training_started', False)
        else:
            print(f"Cliente {client_id}: Falha no heartbeat. Mensagem: {data.get('message', 'Desconhecida')}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Cliente {client_id}: Erro ao enviar heartbeat: {e}")
        return False

def check_training_status():
    """Verifica o status atual do treinamento no servidor."""
    try:
        response = requests.get(f"{SERVER_URL}/status")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro ao verificar status do treinamento: {e}")
        return None

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
    
    # Número de rodadas de treinamento
    num_rounds = 30
    heartbeat_interval = 10  # segundos
    last_heartbeat_time = 0

    print(f"\n{'='*30}")
    print(f"CLIENTE DE APRENDIZADO FEDERADO - ID: {client_id}")
    print(f"{'='*30}")
    
    # 1. Registrar cliente no servidor
    print(f"Cliente {client_id}: Registrando no servidor...")
    training_started = False
    
    while not training_started:
        training_started = register_client(client_id)
        if not training_started:
            status = check_training_status()
            if status and status.get('clients_connected') > 0:
                print(f"Cliente {client_id}: Aguardando início do treinamento... "
                      f"({status.get('clients_connected')}/{status.get('clients_expected')} clientes conectados)")
            else:
                print(f"Cliente {client_id}: Aguardando conexão com o servidor...")
            time.sleep(5)
    
    print(f"Cliente {client_id}: Treinamento iniciado! Começando rodadas de treinamento.")

    for round_num in range(num_rounds):
        print(f"\n--- Rodada {round_num + 1}/{num_rounds} ---")
        round_start_time = time.time()
        
        # Enviar heartbeat periodicamente
        current_time = time.time()
        if current_time - last_heartbeat_time > heartbeat_interval:
            send_heartbeat(client_id)
            last_heartbeat_time = current_time

        # 1. Obter modelo global
        print(f"Cliente {client_id}: Solicitando modelo global do servidor...")
        global_params, download_size, download_time = get_global_model(client_id)

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
        response, upload_size, upload_time = send_trained_parameters(client_id, updated_parameters)

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

        # Verificar status do treinamento no servidor
        status_data = check_training_status()
        if status_data:
            server_round = status_data.get('current_round', 0)
            total_rounds = status_data.get('total_rounds', num_rounds)
            clients_active = status_data.get('clients_connected', 0)
            
            print(f"Cliente {client_id}: Status do servidor - "
                  f"Rodada: {server_round}/{total_rounds}, "
                  f"Clientes ativos: {clients_active}")
            
            # Se o servidor já completou todas as rodadas, podemos parar
            if server_round >= total_rounds:
                print(f"Cliente {client_id}: Servidor concluiu todas as rodadas. Finalizando cliente.")
                break

        # Aguarda um tempo antes da próxima rodada e envia heartbeat
        print(f"Cliente {client_id}: Aguardando próxima rodada...")
        send_heartbeat(client_id)
        last_heartbeat_time = time.time()
        time.sleep(5)
    
    # Ao final de todas as rodadas salva as métricas em um Excel
    excel_file = metrics.save_to_excel()
    print(f"\n{'='*30}")
    print(f"Cliente {client_id}: Processo de treinamento federado concluído.")
    print(f"Análise detalhada disponível em: {excel_file}")
    print(f"{'='*30}")