import os
import psutil
import time
import logging


def find_pid(process_name_):
    pid_ = None
    for proc in psutil.process_iter():
        if process_name_ in proc.name():
            pid_ = proc.pid

    return pid_


if __name__ == '__main__':
    process_name = "gmsh"
    logging.basicConfig(level=logging.INFO)

    while True:
        pid = find_pid(process_name)
        if pid is None:
            logging.info(f'No {process_name} found, sleeping... (My pid: {os.getpid()})')
            time.sleep(300)
        else:
            logging.info(f'{process_name} found! Killing pid {pid}')
            os.system("kill {0}".format(pid))
            time.sleep(30)



