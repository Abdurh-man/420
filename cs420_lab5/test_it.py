import logging
import threading
import time

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    x1 = threading.Thread(target=thread_function, args=(1,))
    x2 = threading.Thread(target=thread_function, args=(2,))

    x1.start()
    x2.start()
    # x.join()