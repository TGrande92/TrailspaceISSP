# Import the FgClient class
from fgclient import FgClient
import subprocess
import time
import csv, os

def run_fgfs(filename):
    with open(filename, 'r') as file:
        fg_command = file.read().strip()  # Read and strip any trailing whitespace
    return subprocess.Popen(fg_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def kill_fgfs():
    subprocess.run(["bash", "kill.sh"])

def get_run_number():
    # Check the number of runs already logged in the CSV file
    if os.path.exists('flight_duration_log.csv'):
        with open('flight_duration_log.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            run_number = sum(1 for _ in reader)  # Subtract 1 for the header row
    else:
        run_number = 0

    return run_number

#Individual input setting

def throttle_input(throttle):
    #Set up the client then set the throttle
    client = FgClient()
    client.set_throttle(throttle)

def aileron_input(aileron):
    #Set up the client then set the aileron
    client = FgClient()
    client.set_aileron(aileron)

def elevator_input(elevator):
    #Set up the client then set the elevator
    client = FgClient()
    client.set_elevator(elevator)

def rudder_input(rudder):
    #Set up the client then set the rudder
    client = FgClient()
    client.set_rudder(rudder)

def main():
    run_number = get_run_number()
    while True:
        try:
            print(f"Starting Flightgear for Run No {run_number}!")
            run_fgfs('run_fg_in.sh')
            time.sleep(10)
            run_start_time = time.time()  # Record the start time for the current run
            print("You can now control the plane!")
            client = FgClient()

            # Define the property name and the new value you want to set
            while client.altitude_ft() > 180:
                throttle = input("Enter Throttle Value between -1.0 and 1.0: ")
                client.set_throttle(throttle)
                if client.altitude_ft() < 180:
                    break
                aileron = input("Enter Aileron Value between -1.0 and 1.0: ")
                client.set_aileron(aileron)
                if client.altitude_ft() < 180:
                    break
                elevator = input("Enter Elevator Value between -1.0 and 1.0: ")
                client.set_elevator(elevator)
                if client.altitude_ft() < 180:
                    break
                rudder = input("Enter Rudder Value between -1.0 and 1.0: ")
                client.set_rudder(rudder)
            # Set the property value

        finally:
            run_end_time = time.time()  # Record the end time for the current run
            kill_fgfs()
            duration = run_end_time - run_start_time  # Calculate the flight duration
            print(f"Flight duration for Run No {run_number}: {duration} seconds")

            # Log the duration to a CSV file
            with open('flight_duration_log.csv', mode='a', newline='') as csvfile:
                fieldnames = ['Run No', 'Duration (sec)']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if run_number == 0 or os.path.getsize('flight_duration_log.csv') == 0:
                    writer.writeheader()  # Write header in the first run
                

                # Write the data to the CSV file
                writer.writerow({'Run No': run_number, 'Duration (sec)': duration})
                run_number += 1
            print(f"Restarting FlightGear for the next run...")
            time.sleep(5)

if __name__ == "__main__":
    main()
