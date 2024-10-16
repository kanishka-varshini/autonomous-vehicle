import argparse
import time
from pathlib import Path
from numpy import random

import numpy as np

import serial
import pyfirmata2


com_port = 'COM8'

# ser = serial.Serial(com_port, 9600)
board = pyfirmata2.Arduino(com_port)
print('Firmata connection established')
brake_dir = board.get_pin('d:13:o')  # Example pin setup for 'a'
brake_pwm = board.get_pin('d:11:p')  # Example pin setup for 'b' as PWM
accn_1 = board.get_pin('d:7:o')
accn_2 = board.get_pin('d:8:o')
accn_pwm = board.get_pin('d:9:p')
brake_active = 0
maxspeed = 255
currentspeed=0.0
increment = 10.0
go_count = 0
max_go_count = 0


def vehicle_stop():
    global go_count
    global brake_active
    global currentspeed
    go_count = 0
              
    print("stopping vehicle")
    brake_dir.write(1)
    brake_pwm.write(1)
    # brake_motor.setSpeed(255)
    brake_active = 1
    
    accn_pwm.write(0)
    accn_1.write(1)
    accn_2.write(0)
    # motor.setSpeed(0)
    # motor.forward()
    currentspeed = 0


def vehicle_go(speed):
    global go_count
    global brake_active
    global currentspeed
    go_count+=1
    if go_count > max_go_count:
        if brake_active == 1:
            print("releasing brakes")
            brake_dir.write(0)
            brake_pwm.write(1)
            time.sleep(1)
            brake_pwm.write(0)
            brake_active = 0
        
        if currentspeed < speed:
            currentspeed+=increment
            if currentspeed > speed:
                currentspeed = maxspeed
        
        accn_pwm.write(currentspeed/255)
        accn_1.write(1)
        accn_2.write(0)
        print(f'currentspeed = {currentspeed}')


def adjust_speed(speed_level):
    if speed_level == 1:
        target_speed = 85  
    elif speed_level == 2:
        target_speed = 170  
    elif speed_level == 3:
        target_speed = maxspeed
    else:
        print("Invalid speed level, stopping vehicle.")
        vehicle_stop()
        return
    vehicle_go(target_speed)


def control_vehicle():
    global max_go_count
    parser = argparse.ArgumentParser(description="Control the golf cart's movement.")
    parser.add_argument('speed', type=int, choices=[1, 2, 3], help="Set the speed level: 1 (low), 2 (medium), 3 (high)")
    parser.add_argument('--time', type=float, default=10.0, help="Set the duration the cart should run (in seconds)")
    args = parser.parse_args()

    speed_level = args.speed
    max_go_count = int(args.time / 0.1)  # Adjust based on time resolution, assuming 0.1s increments

    try:
        print(f'Setting vehicle speed to level {speed_level}')
        adjust_speed(speed_level)
        time.sleep(args.time)  # Run for the set duration
    finally:
        vehicle_stop()  # Stop the vehicle after the time period ends


if __name__ == "__main__":
    try:
        control_vehicle()
    except KeyboardInterrupt:
        print("Process interrupted. Stopping the vehicle.")
        vehicle_stop()
