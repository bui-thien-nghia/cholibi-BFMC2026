instructions:
1. connect your lap to bfmcdemocar wifi
2. ssh to pi
3. find your laptop ip
  ping "laptop name".local
4. go to sender.py on raspberry and replace LAPTOP_IP with your lap's IP
5. change SAVE_DIR in receiver.py if wanted
6. run sender.py on the raspberry (ssh) and receiver.py on your lap simultaneously
  python3 ....py
    sender: pi
    receiver: lap
7. wait for connection.
