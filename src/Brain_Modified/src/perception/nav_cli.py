import sys
import os
import time

# Add root to path so we can import project modules
sys.path.insert(0, os.getcwd())

from multiprocessing import Queue
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import NavGoal

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 src/perception/nav_cli.py <start_node> <end_node>")
        print("Example: python3 src/perception/nav_cli.py 342 98")
        return

    start_node = sys.argv[1]
    end_node = sys.argv[2]
    
    # In this project architecture, queues are localized to the main process.
    # To talk to a RUNNING main.py, we would ideally use a Socket.
    # However, since we want a quick tool, let's explain the logic:
    
    print(f"[CLI] Planning path from {start_node} to {end_node}...")
    
    # NOTE: This script won't work while main.py is running because 
    # multiprocessing.Queues cannot be easily shared between unrelated processes
    # without a Manager or a socket.
    
    print("\n[ERROR] Direct Queue access from a separate process is not possible.")
    print("To send this command while main.py is running, please use the Dashboard.")
    print("Or, I can add a Socket Listener to threadVIO.py so you can use 'nc' (netcat).")

if __name__ == "__main__":
    main()
