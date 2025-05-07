"""
Healthcheck endpoint для Streamlit Cloud
Этот файл позволяет проверить доступность порта 8501
"""
import socket
import sys
import os
import time

def check_port(host='localhost', port=8501, retries=5, delay=2):
    """Проверяет доступность порта с повторными попытками"""
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt+1}/{retries} checking {host}:{port}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # 2 second timeout
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"Port {port} is open on {host}!")
                return True
            else:
                print(f"Port {port} is not available on {host} (code: {result})")
                if attempt < retries - 1:
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
        except Exception as e:
            print(f"Error checking port: {e}")
            if attempt < retries - 1:
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
    
    return False

def main():
    """Основная логика здравпроверки"""
    # Проверка порта на localhost
    localhost_ok = check_port(host='localhost', port=8501)
    
    # Проверка порта на 0.0.0.0
    all_interfaces_ok = check_port(host='0.0.0.0', port=8501)
    
    # Проверка порта на 127.0.0.1
    loopback_ok = check_port(host='127.0.0.1', port=8501)
    
    # Вывод системной информации
    print("\nSystem Information:")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Environment: {dict(os.environ)}")
    
    # Вывод итоговых результатов
    print("\nHealthcheck Results:")
    print(f"Port 8501 on localhost: {'OK' if localhost_ok else 'FAILED'}")
    print(f"Port 8501 on all interfaces: {'OK' if all_interfaces_ok else 'FAILED'}")
    print(f"Port 8501 on loopback: {'OK' if loopback_ok else 'FAILED'}")
    
    # Возвращаем код ошибки, если все проверки не удались
    if not (localhost_ok or all_interfaces_ok or loopback_ok):
        print("All healthchecks failed!")
        sys.exit(1)
    else:
        print("At least one healthcheck succeeded. Service is considered available.")
        sys.exit(0)

if __name__ == "__main__":
    main()