import jsbsim

class JsbsimInterface:
    def __init__(self):
        # Initialize JSBSim
        self.exec = jsbsim.FGFDMExec(None)
        self.exec.load_model('Rascal110-JSBSim')
        # Set initial conditions (modify as necessary)
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
        x_accel = self.exec.get_property_value('accelerations/udot-ft_sec2')
        y_accel = self.exec.get_property_value('accelerations/vdot-ft_sec2')
        z_accel = self.exec.get_property_value('accelerations/wdot-ft_sec2')
        return [altitude, x_accel, y_accel, z_accel]

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
        self.exec.reset_to_initial_conditions(1)
        self.exec.run_ic()  # Re-run initial conditions

        # Reset control surfaces and other variables
        self.exec['fcs/throttle-cmd-norm'] = 0  # Throttle
        self.exec['fcs/elevator-cmd-norm'] = 0  # Elevator
        self.exec['fcs/aileron-cmd-norm'] = 0   # Aileron
        self.exec['fcs/rudder-cmd-norm'] = 0    # Rudder

