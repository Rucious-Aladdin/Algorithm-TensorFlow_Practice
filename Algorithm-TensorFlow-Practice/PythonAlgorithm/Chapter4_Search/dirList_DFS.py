"""
import os
print(os.listdir("/"))

for i in os.listdir("/"):
    print(i + " : " + str(os.path.isfile("/" + i)))
    print(i + " : " + str(os.path.isdir("/" + i)))
"""

import os 

def search(dir, name):
    for i in os.listdir(dir):
        if i == name:
            print(dir + i)
        if os.path.isdir(dir + i):
            if os.access(dir + i, os.R_OK):
                search(dir + i + "/", name)
                
search("C:/book/", "book")    