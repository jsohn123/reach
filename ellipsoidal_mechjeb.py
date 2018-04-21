import math
import time
import krpc
import threading
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import time
import pandas as pd
import matplotlib.animation as animation


conn = krpc.connect(name='Launch into orbit')
vessel = conn.space_center.active_vessel
KCKF = vessel.orbit.body.reference_frame

# Set up streams for telemetry
ut = conn.add_stream(getattr, conn.space_center, 'ut')
altitude = conn.add_stream(getattr, vessel.flight(), 'mean_altitude')
apoapsis = conn.add_stream(getattr, vessel.orbit, 'apoapsis_altitude')
stage_2_resources = vessel.resources_in_decouple_stage(stage=2, cumulative=False)
srb_fuel = conn.add_stream(stage_2_resources.amount, 'SolidFuel')
latitude = conn.add_stream(getattr, vessel.flight(KCKF), "latitude")
longitude = conn.add_stream(getattr, vessel.flight(KCKF), "longitude")

planet_radius = vessel.orbit.body.equatorial_radius

ksc_coord = np.array((-0.09720771, -74.55767342))


class flight_data():
    def __init__(self, vessel):
        self.launch_vehicle = vessel
        self.srbs_separated = False
        self.turn_angle = 0
        self.trajectory =self.launch_vehicle.position(vessel.orbit.body.reference_frame)
        self.turn_start_altitude = 250
        self.turn_end_altitude = 45000
        self.target_altitude = 150000
        # Main ascent loop
        #srbs_separated = False
        #turn_angle = 0
    def update_trajectory(self, lv_pos):
        self.trajectory = np.vstack((self.trajectory, lv_pos))




def launch_init_sequence(flight_data):
    # Countdown...
    print('3...')
    time.sleep(1)
    print('2...')
    time.sleep(1)
    print('1...')
    time.sleep(1)
    print('Launch!')

    # Activate the first stage
    flight_data.launch_vehicle.control.activate_next_stage()
    flight_data.launch_vehicle.auto_pilot.engage()
    flight_data.launch_vehicle.auto_pilot.target_pitch_and_heading(90, 90)

def main_ascent_loop(flight_data):

    vessel = flight_data.launch_vehicle
    while True:
        lv_pos = vessel.position(vessel.orbit.body.reference_frame)
        print('(%.1f, %.1f, %.1f)' % lv_pos)
        flight_data.update_trajectory(lv_pos)

        #ax.scatter3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c=trajectory[:, 2], cmap='Greens');

        #plt.pause(0.0001)  # Note this correction
        time.sleep(0.05)
        # Gravity turn
        if altitude() > flight_data.turn_start_altitude and altitude() < flight_data.turn_end_altitude:
            frac = ((altitude() - flight_data.turn_start_altitude) /
                    (flight_data.turn_end_altitude - flight_data.turn_start_altitude))
            new_turn_angle = frac * 90
            if abs(new_turn_angle - flight_data.turn_angle) > 0.5:
                flight_data.turn_angle = new_turn_angle
                vessel.auto_pilot.target_pitch_and_heading(90 - flight_data.turn_angle, 90)

        # Separate SRBs when finished
        if not flight_data.srbs_separated:
            if srb_fuel() < 0.1:
                vessel.control.activate_next_stage()
                flight_data.srbs_separated = True
                print('SRBs separated')

        # Decrease throttle when approaching target apoapsis
        if apoapsis() > flight_data.target_altitude * 0.9:
            print('Approaching target apoapsis')
            break

    # Disable engines when target apoapsis is reached
    vessel.control.throttle = 0.25
    while apoapsis() < flight_data.target_altitude:
        flight_data.update_trajectory(lv_pos)
        time.sleep(1)
    print('Target apoapsis reached')
    vessel.control.throttle = 0.0

    # Wait until out of atmosphere
    print('Coasting out of atmosphere')
    while altitude() < 70500:
        flight_data.update_trajectory(lv_pos)
        time.sleep(1)

def circularization(flight_data):
    # Plan circularization burn (using vis-viva equation)

    vessel = flight_data.launch_vehicle

    print('Planning circularization burn')
    mu = vessel.orbit.body.gravitational_parameter
    r = vessel.orbit.apoapsis
    a1 = vessel.orbit.semi_major_axis
    a2 = r
    v1 = math.sqrt(mu * ((2. / r) - (1. / a1)))
    v2 = math.sqrt(mu * ((2. / r) - (1. / a2)))
    delta_v = v2 - v1
    node = vessel.control.add_node(
        ut() + vessel.orbit.time_to_apoapsis, prograde=delta_v)

    # Calculate burn time (using rocket equation)
    F = vessel.available_thrust
    Isp = vessel.specific_impulse * 9.82
    m0 = vessel.mass
    m1 = m0 / math.exp(delta_v / Isp)
    flow_rate = F / Isp
    burn_time = (m0 - m1) / flow_rate

    # Orientate ship
    vessel.control.rcs = True
    print('Orientating ship for circularization burn')
    vessel.auto_pilot.reference_frame = node.reference_frame
    vessel.auto_pilot.target_direction = (0, 1, 0)
    vessel.auto_pilot.wait()

    # Wait until burn
    print('Waiting until circularization burn')
    burn_ut = ut() + vessel.orbit.time_to_apoapsis - (burn_time / 2.)
    lead_time = 5
    conn.space_center.warp_to(burn_ut - lead_time)

    lv_pos = vessel.position(vessel.orbit.body.reference_frame)
    flight_data.update_trajectory(lv_pos)

    # Execute burn
    print('Ready to execute burn')
    time_to_apoapsis = conn.add_stream(getattr, vessel.orbit, 'time_to_apoapsis')
    while time_to_apoapsis() - (burn_time / 2.) > 0:
        lv_pos = vessel.position(vessel.orbit.body.reference_frame)
        flight_data.update_trajectory(lv_pos)
        time.sleep(0.5)

    print('Executing burn')
    vessel.control.throttle = 1.0
    time.sleep(burn_time - 0.1)
    print('Fine tuning')
    vessel.control.throttle = 0.05
    remaining_burn = conn.add_stream(node.remaining_burn_vector, node.reference_frame)
    prev_burn_residual = remaining_burn()[1]

    while remaining_burn()[1] > 0.1:
        print remaining_burn()[1]
        if prev_burn_residual > remaining_burn()[1]:
            #we missed crossing point
            print "shutting down, close enough"
            break

        lv_pos = vessel.position(vessel.orbit.body.reference_frame)
        flight_data.update_trajectory(lv_pos)
        time.sleep(0.5)
    vessel.control.rcs = False
    vessel.control.throttle = 0.0
    node.remove()


def main():
    vessel.control.sas = False
    vessel.control.rcs = False
    vessel.control.throttle = 1.0
    orbit_flight = flight_data(vessel)

    launch_init_sequence(orbit_flight)
    main_ascent_loop(orbit_flight)
    circularization(orbit_flight)






main()

