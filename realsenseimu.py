import pyrealsense2 as rs
import numpy as np
import time

class IMUDataProcessor:
    def __init__(self, threshold=0.02, alpha=0.8):
        self.speed = 0.0  # Initial speed (m/s)
        self.prev_time = time.time()  # To track the time for integration
        self.threshold = threshold  # Acceleration threshold
        self.alpha = alpha  # Low-pass filter coefficient
        self.last_accel = np.array([0.0, 0.0, 0.0])  # Previous filtered acceleration

    def calculate_speed(self, accel_data):
        current_time = time.time()
        dt = current_time - self.prev_time  # Time difference from the last reading

        # Get acceleration components
        ax, ay, az = accel_data.x, accel_data.y, accel_data.z

        # Apply low-pass filter to reduce noise
        current_accel = np.array([ax, ay, az])
        filtered_accel = self.alpha * self.last_accel + (1 - self.alpha) * current_accel
        self.last_accel = filtered_accel

        # Calculate speed only if the filtered acceleration exceeds the threshold
        if np.linalg.norm(filtered_accel[:2]) > self.threshold:  # Only consider horizontal acceleration
            # Integrate acceleration to get speed
            self.speed += np.sqrt(filtered_accel[0]**2 + filtered_accel[1]**2) * dt  # Update speed
        else:
            # Reset speed if stationary (you may want to implement a more sophisticated condition)
            self.speed = 0.0

        # Update previous time
        self.prev_time = current_time

        return self.speed

def main():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Enable accelerometer and gyroscope streams
    config.enable_stream(rs.stream.accel)  # Enable accelerometer
    config.enable_stream(rs.stream.gyro)   # Enable gyroscope

    # Start the pipeline
    pipeline.start(config)

    imu_processor = IMUDataProcessor()  # Initialize the IMU data processor

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()

            # Get accelerometer and gyroscope frames
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)

            if accel_frame and gyro_frame:
                # Get accelerometer data
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                # Get gyroscope data
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()

                # Calculate speed
                speed = imu_processor.calculate_speed(accel_data)

                # Print accelerometer and gyroscope data
                print(f"Accelerometer - X: {accel_data.x:.2f}, Y: {accel_data.y:.2f}, Z: {accel_data.z:.2f}")
                print(f"Gyroscope - X: {gyro_data.x:.2f}, Y: {gyro_data.y:.2f}, Z: {gyro_data.z:.2f}")
                print(f"Calculated Speed: {speed:.2f} m/s")

            # Sleep for a short duration to limit the output rate
            time.sleep(0.1)

    except KeyboardInterrupt:
        # Allow user to exit the loop using Ctrl+C
        print("Exiting...")
    finally:
        # Stop the pipeline
        pipeline.stop()

if __name__ == "__main__":
    main()
