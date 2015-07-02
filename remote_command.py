import sys
import subprocess
import time
import threading
import os
import signal
import datetime
import glob

fin = open(sys.argv[1], "r")
for line in fin:
    lineList = line.split()
    host = lineList[0] + "@" + lineList[1]
    command = 'ssh ' + host + ' "' + sys.argv[2] + '"'
    print command
    sys.stdout.flush()
    p = subprocess.Popen(command, shell=True)
    code = p.wait()
    if code > 0:
        print "Host " + host + " Failed!"
fin.close()
