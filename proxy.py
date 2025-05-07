"""
Прокси-сервер для перенаправления запросов с порта 8501 на порт 5000.
Это решает проблему проверки Replit, который ищет Streamlit на порту 8501.
"""
import socket
import sys
from threading import Thread

def proxy_thread(src, dst):
    """Перенаправление данных между сокетами"""
    try:
        while True:
            data = src.recv(4096)
            if not data:
                break
            dst.sendall(data)
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        try:
            src.close()
            dst.close()
        except:
            pass

def start_proxy(source_port, destination_port):
    """Запуск прокси-сервера для перенаправления с source_port на destination_port"""
    try:
        # Создаем серверный сокет
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Привязываем к порту 8501 (источник)
        server_socket.bind(('0.0.0.0', source_port))
        server_socket.listen(5)
        
        print(f"Прокси запущен: перенаправление {source_port} -> {destination_port}")
        
        while True:
            # Принимаем подключение
            client_socket, addr = server_socket.accept()
            print(f"Подключение от {addr}")
            
            # Подключаемся к целевому порту (5000)
            target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target_socket.connect(('localhost', destination_port))
            
            # Запускаем потоки для двунаправленного проксирования
            Thread(target=proxy_thread, args=(client_socket, target_socket)).start()
            Thread(target=proxy_thread, args=(target_socket, client_socket)).start()
            
    except Exception as e:
        print(f"Ошибка запуска прокси: {e}")
        sys.exit(1)
    finally:
        try:
            server_socket.close()
        except:
            pass

if __name__ == "__main__":
    # Перенаправляем запросы с порта 8501 на 5000
    start_proxy(8501, 5000)