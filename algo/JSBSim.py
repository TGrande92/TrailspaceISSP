import jsbsim

class JsbsimInterface:
    def __init__(self):
        # Initialize JSBSim
        self.previous_altitude = None
        self.exec = jsbsim.FGFDMExec(None)
        self.exec.set_debug_level(0)
        self.exec.load_model('Rascal110-JSBSim')
        # Set initial conditions
        self.exec['ic/h-sl-ft'] = 1000   # Initial altitude (feet)
        # self.exec['ic/vc-kts'] = 100      # Initial airspeed (knots)

    def get_sim_time(self):
        # Retrieve the internal simulation time
        return self.exec.get_sim_time()

    def start(self):
        # Run the initial conditions
        self.exec.run_ic()
        self.exec['fcs/throttle-cmd-norm'] = 0

    def get_state(self):
        # Retrieve state information
        altitude = self.exec.get_property_value('position/h-sl-ft')
        if self.previous_altitude is None:
            self.previous_altitude = altitude  
        x_accel = self.exec.get_property_value('accelerations/udot-ft_sec2')
        y_accel = self.exec.get_property_value('accelerations/vdot-ft_sec2')
        z_accel = self.exec.get_property_value('accelerations/wdot-ft_sec2')
        return [altitude, x_accel, y_accel, z_accel]

    def get_altitude_change(self):
        # Calculate altitude change since last state retrieval
        current_altitude = self.exec.get_property_value('position/h-sl-ft')
        altitude_change = self.previous_altitude - current_altitude
        self.previous_altitude = current_altitude  # Update previous altitude
        return altitude_change

    def set_controls(self, elevator, aileron, rudder):
        # Set control surface positions
        self.exec.set_property_value('fcs/elevator-cmd-norm', elevator)
        self.exec.set_property_value('fcs/aileron-cmd-norm', aileron)
        self.exec.set_property_value('fcs/rudder-cmd-norm', rudder)

    def run(self):
        # Execute one simulation step
        return self.exec.run()
        
    def stop(self):
        # Reset the JSBSim execution to initial conditions
        self.exec.reset_to_initial_conditions(0)

    def gear_contact(self):
        # Check if any gear has made contact with the ground
        # Return True if any gear is in contact with the ground
        number_of_gears = 3
        return any(self.exec.get_property_value(f'gear/unit[{i}]/WOW') for i in range(number_of_gears))


