import socket
import struct
import time
from collections import namedtuple
import pandas as pd

# Define the host and port to listen on
HOST = '127.0.0.1'
PORT = 6789

#Create a log
data_log = []

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

# Define the struct format to match the XML configuration
# The format includes double values for all the specified chunks
struct_format = '<ddddd'  # For 20 double values

dataStr = namedtuple('dataStr',['elapsedTime', 'altitude','xaccel','yaccel','zaccel'])

t = time.time()
try:
    while True:
        # Receive data from the socket
        data, addr = sock.recvfrom(300)  # Adjust the size according to your struct format
        print(f"Received {len(data)} bytes")
        for i in range(0, len(data)):
            print(hex(data[i]))
        print("\n")

        try:
            tm = struct.unpack('>ddddd', data)
        except struct.error as e:
            print(f"Error unpacking data: {e}")
            continue  # Skip the rest of this loop iteration

        # print(tm[0])
        instance = dataStr(*tm)
        print(instance.elapsedTime)
        print(instance.altitude)
        print(instance.xaccel)
        print(instance.yaccel)
        print(instance.zaccel)

        data_log.append({
            'elapsedTime': instance.elapsedTime,
            'altitude': instance.altitude,
            'xaccel': instance.xaccel,
            'yaccel': instance.yaccel,
            'zaccel': instance.zaccel
        })

        elapsed = time.time() - t
        #print(elapsed)
finally:
    df = pd.DataFrame(data_log)
    df.to_csv('data.csv', index=False)
    sock.close()

# Close the socket when done (usually, you'd use some exit condition to stop the loop)