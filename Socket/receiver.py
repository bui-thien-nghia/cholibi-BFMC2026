import socket
import struct
import os
import glob
import time

# CONFIGURATION
HOST = '0.0.0.0'
PORT = 8000
SAVE_DIR = os.path.expanduser("C:/anhtest")
MAX_IMAGES = 30

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
server.bind((HOST, PORT))
server.listen(0)

def maintain_limits(directory, limit):
    files = glob.glob(os.path.join(directory, "*.jpg"))
    if len(files) >= limit:
        time.sleep(10)
        for f in files:
            os.remove(f)



print(f"--- SERVER RUNNING ---")



while True:
    try:
        print("\nWaiting for connect")
        client, addr = server.accept()
        connection = client.makefile('rb')
        count = 1
        
        while True:
            data = connection.read(struct.calcsize('<L'))

            image_len = struct.unpack('<L',data)[0]
            
            if image_len:
                image_stream = connection.read(image_len)

                total_files = len(os.listdir(SAVE_DIR))
                filename = os.path.join(SAVE_DIR, f"image_{total_files + 1}.jpg")

                with open(filename, 'wb') as f:
                    f.write(image_stream)
                print(f"Received image: {filename}")

                maintain_limits(SAVE_DIR, MAX_IMAGES)

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        try:
            client.close()
        except:
            pass
