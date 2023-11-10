# Import the FgClient class
from fgclient import FgClient
import subprocess
import time
import csv, os
import asyncio

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

def data_bg(client):
    data_log = []
    data_log.append({
        client.get_xaccel(),
        client.get_yaccel(),
        client.get_zaccel()
    })
    print(data_log)
    return data_log

def main():
    run_number = get_run_number()
    data = []
    while True:
        try:
            print(f"Starting Flightgear for Run No {run_number}!")
            run_fgfs('run_fg_in.sh')
            time.sleep(10)
            print("You can now control the plane!")
            client = FgClient()
            print("Current Wind Speed in knots: ", client.get_windspeed())
            # client.set_windspeed(input("Enter wind speed in knots: "))
            
            # print("Updated Wind Speed:", client.get_windspeed())
            print("Current Wind Direction in Deg: ", client.get_wind_direction())
            # client.set_wind_direction(input("Enter wind direction in degrees: "))
            # client.set_wind_direction(220) #Setting the default wind direction to 220 degrees
            # print("Updated Wind Direction in Deg: ", client.get_wind_direction())            
            # Define the property name and the new value you want to set
            while client.altitude_ft() > 180:
                client.set_wind_direction(100.00) #Setting the default wind direction to 220 degrees
                client.set_windspeed(8.0)

                # throttle = input("Enter Throttle Value between -1.0 and 1.0: ")
                client.set_throttle(0)
                if client.altitude_ft() < 180:
                    break
                # aileron = input("Enter Aileron Value between -1.0 and 1.0: ")
                client.set_aileron(0)
                if client.altitude_ft() < 180:
                    break
                # elevator = input("Enter Elevator Value between -1.0 and 1.0: ")
                client.set_elevator(0)
                if client.altitude_ft() < 180:
                    break

                # rudder = input("Enter Rudder Value between -1.0 and 1.0: ")
                client.set_rudder(0)
                start_time = time.time()  # Get the current time in seconds
                while True:
                         #Setting default wind speed to 20 knots
                    
                    print(  
                            client.get_wind_direction(),
                            client.get_windspeed(),
                            client.get_aileron(),
                            client.get_elevator(),
                            client.get_rudder(),
                            client.get_xaccel(),
                            client.get_yaccel(),
                            client.get_zaccel()
                            )
                    current_time = time.time()  # Get the current time in seconds
                    elapsed_time = current_time - start_time

                    if elapsed_time >= 5:
                        break
                # client.set_windspeed(input("Enter wind speed in knots: "))
            # Set the property value

        finally:
            # print(data)
            print(client.elapsed_time())
            duration = client.elapsed_time()
            kill_fgfs()
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