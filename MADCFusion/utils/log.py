import time
import logging
import os

def initialize_log(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = 'model_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    FORMAT = '%(levelname)s %(filename)s ==  %(message)s'
    log_file = os.path.join(log_path, log_file)
    logging.basicConfig(level=logging.INFO, format=FORMAT, filename=log_file)
    logging.root.addHandler(logging.StreamHandler())