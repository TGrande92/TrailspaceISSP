import csv
import os
import matplotlib.pyplot as plt

def extract_flight_duration(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'Flight Duration':
                return float(row[4])

def plot_flight_durations(runlogs_folder):
    flight_durations = []
    run_numbers = []

    # Loop through all files in the runlogs folder
    for filename in os.listdir(runlogs_folder):
        if filename.endswith('.csv'):
            run_number = int(filename.split('run')[1].split('.csv')[0])
            file_path = os.path.join(runlogs_folder, filename)
            duration = extract_flight_duration(file_path)
            
            if duration is not None:
                flight_durations.append(duration)
                run_numbers.append(run_number)

    # Sort the runs by run number
    sorted_runs = sorted(zip(run_numbers, flight_durations))
    run_numbers, flight_durations = zip(*sorted_runs)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(run_numbers, flight_durations, marker='o')
    plt.title('Flight Duration for Each Simulation Run')
    plt.xlabel('Run Number')
    plt.ylabel('Flight Duration (Seconds)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    runlogs_folder = 'runlogs'  # Set this to your runlogs folder path
    plot_flight_durations(runlogs_folder)
