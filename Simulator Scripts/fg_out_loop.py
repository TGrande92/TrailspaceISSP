import socket
import struct
import time
import subprocess
import os, signal
from collections import namedtuple
import pandas as pd
import telnetlib

# Define the host and port to listen on
HOST = '127.0.0.1'
PORT = 6789

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

# Define the struct format to match the XML configuration
# The format includes double values for all the specified chunks
struct_format = '<ddddd'  # For 20 double values

dataStr = namedtuple('dataStr',['elapsedTime', 'altitude','xaccel','yaccel','zaccel'])

def get_output_filename():
    i = 0
    while True:
        filename = f"data{i}.csv"
        if not os.path.exists(filename):
            return filename
        i += 1

def exit_fgfs():
    try:
        tn = telnetlib.Telnet("localhost", 5401, timeout=1)
        tn.write(b"run fgcommand(\"exit\")\n")
        tn.read_all()  # Wait for the command to execute
    except Exception as e:
        print(f"Failed to exit FlightGear via telnet: {e}")

def run_fgfs(filename):
    with open(filename, 'r') as file:
        fg_command = file.read().strip()  # Read and strip any trailing whitespace
    return subprocess.Popen(fg_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


# def exit_fgfs(fgfs_process):
#     try:
#         fgfs_process.send_signal(signal.SIGINT)  # Send an interrupt signal (Ctrl+C) to gracefully exit FlightGear
#         print("Flightgear exited!")
#         fgfs_process.wait()
#           # Wait for FlightGear to exit
#     except Exception as e:
#         print(f"Failed to exit FlightGear: {e}")

def kill_fgfs():
    subprocess.run(["bash", "kill.sh"])


def main():
    while True:
        data_log = []
        fgfs_process = run_fgfs('runfg.sh')  # Assuming the script is named FG_run.py and is in the same directory
        print("Booting up FlightGear")
        time.sleep(10)  # give FlightGear some time to start up
        print('Continuing with data collection')
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

                if instance.altitude <= 153:
                    break

        finally:
            kill_fgfs()
            # exit_fgfs()  # Gracefully exit FlightGear
            fgfs_process.wait()
            output, _ = fgfs_process.communicate()
            print(output.decode())  # print any output or errors from FlightGear
            df = pd.DataFrame(data_log)
            df.to_csv(get_output_filename(), index=False)

        print("Restarting FlightGear for the next run...")
        time.sleep(5)  # Optional: give some time before restarting FlightGear

if __name__ == "__main__":
    main()