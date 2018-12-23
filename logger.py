import sys
import time
import datetime

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout

        timestamp = int(time.time())
        date = datetime.datetime.fromtimestamp(timestamp)
        timestamp = date.strftime('%Y-%m-%d_%H:%M:%S')

        self.log = open('logs/file_'+str(timestamp), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass   