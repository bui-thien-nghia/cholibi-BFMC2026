import socket
import struct
import subprocess
import shutil
import time

# CONFIGURATION - MUST MATCH LAPTOP IP
LAPTOP_IP = '192.168.50.2' 
PORT = 8000

if shutil.which("rpicam-still"):
    CMD_BASE = ["rpicam-still"]

CMD_ARGS = ["-t", "1", "-o", "-", "-n", "--width", "1280", "--height", "720"]

client_socket = socket.socket()

try:
    print(f"Connecting to {LAPTOP_IP}...")
    client_socket.connect((LAPTOP_IP, PORT))
    connection = client_socket.makefile('wb')

    while True:
        result = subprocess.run(
            CMD_BASE + CMD_ARGS,
            stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL
        )

        data = result.stdout

        if len(data) == 0:
            print("Error")
            continue

        connection.write(struct.pack('<L', len(data)))
        connection.flush()
        connection.write(data)


except Exception as e:
    print("Connection error: {e}")

finally:
    connection.close()
    client_socket.close()