import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/kyrgios/Documents/bfmc_ws/install/bfmc_perception'
