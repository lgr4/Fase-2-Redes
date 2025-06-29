import torch
import pickle
import os
import threading
import time
from flask import Flask, request, send_file, jsonify
import io
from model import *
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from datetime import datetime

# --- Configuração do Servidor Flask ---
app = Flask(__name__)

# --- Variáveis Globais para gerenciar o estado do treinamento ---
num_clients_expected = 3  # Número esperado de clientes
rounds = 30
current_round = 0
received_params_this_round = {}  # Dicionário para mapear client_id -> parâmetros
active_clients = {}  # Dicionário para rastrear clientes ativos - {client_id: last_active_timestamp}
client_timeout_seconds = 180  # Tempo em segundos para considerar um cliente como inativo
training_started = False  # Flag para indicar se o treinamento começou
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
    """Envia o modelo global para o cliente."""
    global training_started
    
    # Obter ID do cliente (se disponível)
    client_id = request.args.get('client_id')
    
    # Verificar se o treinamento já começou
    if not training_started:
        return jsonify({'status': 'error', 'message': 'Aguardando todos os clientes se conectarem'}), 400
        
    # Se o cliente forneceu um ID, atualize sua atividade
    if client_id:
        update_client_activity(int(client_id))
        print(f"Servidor: Enviando modelo global para o cliente {client_id}.")
    else:
        print("Servidor: Enviando modelo global para um cliente não identificado.")
    
    serialized_params = pickle.dumps(global_parameters)
    return send_file(
        io.BytesIO(serialized_params),
        mimetype='application/octet-stream'
    )

@app.route('/submit_parameters', methods=['POST'])
def submit_parameters():
    """Recebe os parâmetros atualizados de um cliente."""
    global received_params_this_round, global_parameters, current_round, active_clients
    
    # Verificar se há clientes que excederam o timeout
    timed_out_clients = check_clients_timeout()
    if timed_out_clients:
        print(f"Servidor: {len(timed_out_clients)} clientes removidos por timeout: {timed_out_clients}")
    
    try:
        # Obter ID do cliente do parâmetro da requisição
        client_id = request.args.get('client_id')
        if client_id is None:
            return jsonify({'status': 'error', 'message': 'ID do cliente não fornecido'}), 400
            
        client_id = int(client_id)
        client_parameters = pickle.loads(request.data)
        
        # Atualizar atividade do cliente
        update_client_activity(client_id)

        with lock:
            # Adicionar parâmetros do cliente ao dicionário
            received_params_this_round[client_id] = client_parameters
            num_received = len(received_params_this_round)
            num_active = len(active_clients)
            
            print(f"Servidor: Recebeu parâmetros do cliente {client_id}. "
                  f"Total nesta rodada: {num_received}/{num_active} clientes ativos")
            
            # Verificar se todos os clientes ativos enviaram seus parâmetros
            all_params_received = num_received >= num_active
            
            if all_params_received:
                print(f"Servidor: Todos os {num_active} clientes ativos responderam. "
                      f"Agregando parâmetros para a rodada {current_round + 1}...")

                # Lógica de agregação (FedAVG)
                new_parameters = {name: {'weight': torch.zeros_like(param['weight']), 
                                         'bias': torch.zeros_like(param['bias'])} 
                                 for name, param in global_parameters.items()}
                
                # Calcular a média dos parâmetros recebidos
                num_clients_this_round = len(received_params_this_round)
                for client_id, client_params in received_params_this_round.items():
                    for name in client_params:
                        new_parameters[name]['weight'] += client_params[name]['weight'] / num_clients_this_round
                        new_parameters[name]['bias'] += client_params[name]['bias'] / num_clients_this_round
                
                global_parameters = new_parameters
                global_net.apply_parameters(global_parameters)

                # Avalia o modelo após a agregação
                loss, accuracy = evaluate_model(global_parameters, test_loader)
                print("="*30)
                print(f"Servidor: Rodada {current_round + 1} CONCLUÍDA com {num_clients_this_round} clientes")
                print(f"Desempenho do Modelo Global: Perda = {loss:.4f} | Acurácia = {accuracy:.2f}%")
                print("="*30)
                
                # Limpa os parâmetros recebidos para a próxima rodada
                received_params_this_round.clear()
                current_round += 1

                if current_round >= rounds:
                    print("="*30)
                    print("Servidor: Treinamento global CONCLUÍDO!")
                    print("="*30)
                    
                    # Salva o modelo final com timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_path = f"global_parameters_{timestamp}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(global_parameters, f)
                    print(f"Modelo global salvo em: {model_path}")

        return jsonify({'status': 'success', 'message': 'Parâmetros recebidos com sucesso'}), 200
        
    except Exception as e:
        import traceback
        print(f"Erro ao processar parâmetros: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- Funções para gerenciamento de clientes ---
def check_clients_timeout():
    """Verifica se algum cliente excedeu o tempo limite de inatividade."""
    global active_clients, received_params_this_round
    current_time = time.time()
    timed_out_clients = []
    
    with lock:
        for client_id, last_active in list(active_clients.items()):
            if current_time - last_active > client_timeout_seconds:
                print(f"Servidor: Cliente {client_id} excedeu o tempo limite e será removido.")
                timed_out_clients.append(client_id)
                if client_id in received_params_this_round:
                    del received_params_this_round[client_id]
                del active_clients[client_id]
    
    return timed_out_clients

def update_client_activity(client_id):
    """Atualiza o timestamp de atividade do cliente."""
    with lock:
        active_clients[client_id] = time.time()

def all_clients_ready():
    """Verifica se todos os clientes esperados estão registrados e ativos."""
    with lock:
        return len(active_clients) >= num_clients_expected

# --- Rotas para gerenciamento de clientes ---
@app.route('/register', methods=['POST'])
def register_client():
    """Registra um novo cliente no sistema."""
    global active_clients, training_started
    
    try:
        data = request.json
        client_id = data.get('client_id')
        
        if client_id is None:
            return jsonify({'status': 'error', 'message': 'ID do cliente não fornecido'}), 400
            
        with lock:
            active_clients[client_id] = time.time()
            num_active = len(active_clients)
            is_ready = num_active >= num_clients_expected
            
        print(f"Servidor: Cliente {client_id} registrado. Total de clientes ativos: {num_active}/{num_clients_expected}")
        
        # Verificar se podemos iniciar o treinamento
        if is_ready and not training_started:
            with lock:
                training_started = True
            print(f"Servidor: Todos os {num_clients_expected} clientes estão conectados! Treinamento iniciado.")
        
        return jsonify({
            'status': 'success', 
            'message': f'Cliente {client_id} registrado com sucesso',
            'clients_connected': num_active,
            'clients_expected': num_clients_expected,
            'training_started': training_started
        })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Retorna o status atual do treinamento."""
    with lock:
        num_active = len(active_clients)
        current_round_local = current_round
        received_count = len(received_params_this_round)
        
    return jsonify({
        'status': 'active',
        'clients_connected': num_active,
        'clients_expected': num_clients_expected,
        'training_started': training_started,
        'current_round': current_round_local,
        'total_rounds': rounds,
        'received_parameters': received_count
    })

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    """Endpoint para os clientes sinalizarem que estão ativos."""
    try:
        data = request.json
        client_id = data.get('client_id')
        
        if client_id is None:
            return jsonify({'status': 'error', 'message': 'ID do cliente não fornecido'}), 400
            
        update_client_activity(client_id)
        return jsonify({'status': 'success', 'training_started': training_started})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    # Avalia o modelo inicial (aleatório) para ter uma linha de base
    print("\n" + "="*30)
    print("SERVIDOR DE APRENDIZADO FEDERADO")
    print("="*30)
    print("Configurações:")
    print(f"- Número esperado de clientes: {num_clients_expected}")
    print(f"- Timeout de cliente: {client_timeout_seconds} segundos")
    print(f"- Rodadas de treinamento: {rounds}")
    print("="*30)
    
    print("\nAvaliando o modelo inicial (antes do treinamento)...")
    initial_loss, initial_accuracy = evaluate_model(global_parameters, test_loader)
    print(f"Desempenho Inicial: Perda = {initial_loss:.4f} | Acurácia = {initial_accuracy:.2f}%")
    print("\nAguardando a conexão de todos os clientes para iniciar o treinamento...")
    
    # Iniciar em uma thread separada para não bloquear o Flask
    print(f"\nServidor iniciado em http://25.34.238.74:5000 às {datetime.now().strftime('%H:%M:%S')}")
    app.run(host='25.34.238.74', port=5000, debug=False, threaded=True)