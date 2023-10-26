import socket
import struct
import time
from collections import namedtuple

# Define the host and port to listen on
HOST = 'localhost'
PORT = 6788

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Define the struct format to match the XML configuration
# The format includes double values for all the specified chunks
struct_format = '<d'  # For 20 double values

dataStr = namedtuple('dataStr',['aileron','elev','rudder','throttle'])

t = time.time()

while time.time()-t < 45:

    aileron = 0.0
    elev = 0.0
    rudder = 0.0
    if(time.time()-t < 10):
        throttle = 0.0
    else:
        print("I made it")
        throttle = 0.5
        print("throttle changed")

    data_bytes = struct.pack('>dddd', aileron, elev, rudder, throttle)
    sock.sendto(data_bytes, (HOST, PORT))
    print("data sent")

# Close the socket when done (usually, you'd use some exit condition to stop the loop)
sock.close()