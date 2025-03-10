import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import os
from scipy.optimize import least_squares
from scipy import signal
from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from geopy.distance import geodesic

def read_imu_data(bag_file):
    # Open the ROS bag file
    bag = rosbag.Bag(bag_file)
    
    # Lists to store magnetic field data and timestamps
    timestamps = []
    mag_field_x = []
    mag_field_y = []
    mag_field_z = []
    gyro_x = []
    gyro_y = []
    gyro_z = []
    accel_x = []
    accel_y = []
    accel_z = []
    
    # Iterate over all messages in the bag
    for topic, msg, t in bag.read_messages(topics=['/imu']):
        # Extract timestamp and magnetic field data
        timestamps.append(t.to_sec())  # Convert ROS Time to seconds
        mag_field_x.append(msg.mag_field.magnetic_field.x)
        mag_field_y.append(msg.mag_field.magnetic_field.y)
        mag_field_z.append(msg.mag_field.magnetic_field.z)
        
        # Extract gyroscope data (angular velocity)
        gyro_x.append(msg.imu.angular_velocity.x)
        gyro_y.append(msg.imu.angular_velocity.y)
        gyro_z.append(msg.imu.angular_velocity.z)
        
        # Extract accelerometer data
        accel_x.append(msg.imu.linear_acceleration.x)
        accel_y.append(msg.imu.linear_acceleration.y)
        accel_z.append(msg.imu.linear_acceleration.z)
    
    bag.close()
    return np.array(timestamps), np.array(mag_field_x), np.array(mag_field_y), np.array(mag_field_z), np.array(gyro_x), np.array(gyro_y), np.array(gyro_z), np.array(accel_x), np.array(accel_y), np.array(accel_z)

def read_gps_data(bag_file):
    # Open the ROS bag file
    bag = rosbag.Bag(bag_file)
    
    # Lists to store magnetic field data and timestamps
    timestamps = []
    utm_eastings = []
    utm_northings = []
    altitudes = []
    latitudes = []
    longitudes = []
    
    # Iterate over all messages in the bag
    for topic, msg, t in bag.read_messages(topics=['/gps']):
        # Extract timestamp and gps data
        timestamps.append(t.to_sec())  # Convert ROS Time to seconds
        utm_eastings.append(msg.utm_easting)
        utm_northings.append(msg.utm_northing)
        altitudes.append(msg.altitude)
        latitudes.append(msg.latitude)
        longitudes.append(msg.longitude)
    
    bag.close()
    return np.array(timestamps), np.array(utm_eastings), np.array(utm_northings), np.array(altitudes), np.array(latitudes), np.array(longitudes)

def hard_iron_calibration(mag_field_x, mag_field_y, mag_field_z):
    offset_x = (np.max(mag_field_x) + np.min(mag_field_x)) / 2
    offset_y = (np.max(mag_field_y) + np.min(mag_field_y)) / 2
    offset_z = (np.max(mag_field_z) + np.min(mag_field_z)) / 2

    # Apply hard-iron correction
    mag_x_corrected = mag_field_x - offset_x
    mag_y_corrected = mag_field_y - offset_y
    mag_z_corrected = mag_field_z - offset_z
    
    return mag_x_corrected, mag_y_corrected, mag_z_corrected, np.array([offset_x, offset_y, offset_z])


def ellipsoid_residuals(params, data):
    """
    Ellipsoid equation residuals for least squares fitting.
    :param params: ellipsoid parameters (a, b, c, dx, dy, dz)
    :param data: magnetometer data (Nx3 array)
    :return: residuals (difference between data and the ellipsoid surface)
    """
    a, b, c, dx, dy, dz = params
    
    # Translate the data by the hard-iron offset (dx, dy, dz)
    translated_data = data - np.array([dx, dy, dz])
    
    # Ellipsoid equation (x^2/a^2 + y^2/b^2 + z^2/c^2 - 1)
    residuals = (translated_data[:, 0]**2 / a**2 + 
                 translated_data[:, 1]**2 / b**2 + 
                 translated_data[:, 2]**2 / c**2 - 1)
    return residuals

def soft_iron_calibration(x, y, z):
    # Combine the x, y, z magnetometer readings into a 2D numpy array (Nx3), where N is the number of samples
    data = np.column_stack((x, y, z))
    
    # Step 1: Hard-Iron Calibration (find and subtract the mean of each axis)
    hard_iron_offset = np.mean(data, axis=0)
    data_corrected = data - hard_iron_offset
    
    # Step 2: Initial guess for the ellipsoid parameters (a, b, c, dx, dy, dz)
    # We assume an initial guess where a, b, c are the max ranges along each axis
    initial_guess = [np.ptp(data_corrected[:, 0]),  # range in x-axis
                     np.ptp(data_corrected[:, 1]),  # range in y-axis
                     np.ptp(data_corrected[:, 2]),  # range in z-axis
                     hard_iron_offset[0],           # initial dx
                     hard_iron_offset[1],           # initial dy
                     hard_iron_offset[2]]           # initial dz
    
    # Step 3: Perform least squares fitting to find the ellipsoid parameters
    result = least_squares(ellipsoid_residuals, initial_guess, args=(data_corrected,))
    
    # Extract the fitted parameters
    a, b, c, dx, dy, dz = result.x
    
    # Step 4: Apply the transformation to the data (scale to make it spherical)
    # Using the fitted ellipsoid parameters, scale the data
    data_transformed = np.copy(data_corrected)
    data_transformed[:, 0] /= a
    data_transformed[:, 1] /= b
    data_transformed[:, 2] /= c
    
    # Step 5: Return the calibrated data (x, y, z)
    return data_transformed[:, 0], data_transformed[:, 1], data_transformed[:, 2]


def calibrate_moving_data(mag_field_x, mag_field_y, mag_field_z, hard_iron_offsets, soft_iron_scales):
    # Apply hard iron calibration
    mag_field_x_cal = mag_field_x - hard_iron_offsets[0]
    mag_field_y_cal = mag_field_y - hard_iron_offsets[1]
    mag_field_z_cal = mag_field_z - hard_iron_offsets[2]
    
    # Apply soft iron calibration
    mag_field_x_cal /= soft_iron_scales[0]
    mag_field_y_cal /= soft_iron_scales[1]
    mag_field_z_cal /= soft_iron_scales[2]
    
    return mag_field_x_cal, mag_field_y_cal, mag_field_z_cal

def fit_circle(x, y):
    """
    Fit a circle to the given data points (x, y) using least squares method.
    Returns the circle's center (a, b) and radius (R).
    """
    # Calculate the mean of the data points
    N = len(x)
    xmean, ymean = x.mean(), y.mean()
    x -= xmean
    y -= ymean
    U, S, V = np.linalg.svd(np.stack((x, y)))
    
    tt = np.linspace(0, 2*np.pi, 1000)
    circle = np.stack((np.cos(tt), np.sin(tt)))    # unit circle
    transform = np.sqrt(2/N) * U.dot(np.diag(S))   # transformation matrix
    fit_x, fit_y = transform.dot(circle) + np.array([[xmean], [ymean]])
    return fit_x, fit_y

def plot_mag_field( c_mag_field_x, c_mag_field_y, c_mag_field_x_hard, c_mag_field_y_hard, c_mag_field_x_soft, c_mag_field_y_soft, c_fit_x_soft, c_fit_y_soft):
    
    # plt.figure(figsize = (14, 6))
    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(c_mag_field_x, c_mag_field_y, label='RAW X VS Y', color='blue')
    plt.title('RAW Data Magnetic Field Data Circle')
    plt.xlabel('Magnetic Field X (uT)')
    plt.ylabel('Magnetic Field Y (uT)')
    plt.grid(True)
    plt.legend()
    
    # plt.figure()
    plt.subplot(1,3,2)
    plt.plot(c_mag_field_x_hard, c_mag_field_y_hard, label='Calibrated X VS Y', color='red')
    plt.title('Hard Iron Calibrated Magnetic Field Data Circle')
    plt.xlabel('Magnetic Field X (uT)')
    plt.ylabel('Magnetic Field Y (uT)')
    plt.grid(True)
    plt.legend()
    
    # plt.figure()
    plt.subplot(1,3,3)
    plt.plot(c_mag_field_x_soft, c_mag_field_y_soft, label='Calibrated X VS Y', color='green')
    plt.plot(c_fit_x_soft, c_fit_y_soft, '--', label='Fit Circle', color='blue')
    plt.title('Soft Iron Calibrated Magnetic Field Data Circle')
    plt.xlabel('Magnetic Field X (uT)')
    plt.ylabel('Magnetic Field Y (uT)')
    plt.grid(True)
    plt.legend()

def yaw_angle_estimate(c_mag_field_x, c_mag_field_y, c_mag_mean_x, c_mag_mean_y, c_mag_field_x_soft, c_mag_field_y_soft,
                       m_mag_field_x, m_mag_field_y, m_mag_field_z, m_timestamps):
    ## PLOT2:The magnetometer yaw estimation before and after hard and soft iron calibration vs. time
    raw_yaw = np.arctan2(c_mag_field_y - c_mag_mean_y, c_mag_field_x - c_mag_mean_x)
    raw_yaw_deg = np.degrees(raw_yaw)

    # Calculating the corrected magnetometer yaw angle
    corr_yaw = np.arctan2(c_mag_field_y_soft, c_mag_field_x_soft)
    corr_yaw_deg = np.degrees(corr_yaw)

    # Plotting raw magnetometer yaw vs. corrected yaw for comparison
    plt.figure()
    plt.subplot(2, 1, 1)
    # plt.figure(figsize=(8, 6))
    plt.plot(raw_yaw_deg, label='Raw Magnetometer Yaw')
    plt.plot(corr_yaw_deg, label='Corrected Magnetometer Yaw')
    plt.xlabel('Sample')
    plt.ylabel('Yaw Angle (degrees)')
    plt.title('Raw vs. Corrected Magnetometer Yaw Circle Data')
    plt.legend()
    plt.grid(True)
    
    #The magnetometer yaw estimation before and after hard and soft iron calibration vs. time
    raw_yaw = np.arctan2(m_mag_field_y, m_mag_field_x) ## raw yaw calculation before calibration
    raw_yaw_deg = np.degrees(raw_yaw)
    
    m_mag_field_x_hard, m_mag_field_y_hard, m_mag_field_z_hard, basis_circle = hard_iron_calibration(m_mag_field_x, m_mag_field_y, m_mag_field_z)

    corr_yaw_hardiron = np.arctan2(m_mag_field_y_hard, m_mag_field_x_hard)  ## corrected Hard iron calibration
    corr_yaw_deg_hardiron = np.degrees(corr_yaw_hardiron)
    
    m_mag_field_x_soft, m_mag_field_y_soft, m_mag_field_z_soft = soft_iron_calibration(m_mag_field_x_hard, m_mag_field_y_hard, m_mag_field_z_hard)
    
    corr_yaw_softiron = np.arctan2(m_mag_field_y_soft, m_mag_field_x_soft) ## corrected soft iron yaw
    corr_yaw_deg_softiron = np.degrees(corr_yaw_softiron)

    # plt.figure(figsize=(14, 8))
    # plt.figure()
    plt.subplot(2, 1, 2)
    plt.plot(m_timestamps, raw_yaw_deg, label='Raw Magnetometer Yaw', color = 'blue')
    plt.plot(m_timestamps, corr_yaw_deg_hardiron, label='Corrected Yaw (Hard Iron Calibration)', color = 'green')
    plt.plot(m_timestamps, corr_yaw_deg_softiron, label='Corrected Yaw (Soft Iron Calibration)', color = 'red')
    plt.xlabel('Time')
    plt.ylabel('Yaw Angle (degrees)')
    plt.title('Magnetometer Yaw Estimation Before and After Calibration vs. Time Moving Data')
    plt.legend()
    plt.grid(True)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    

def gyro_yaw_rate(m_timestamps, m_gyro_z, m_mag_field_x, m_mag_field_y, m_gyro_x, m_gyro_y):
    
    yaw_rate = m_gyro_z
    time = m_timestamps

    cumulative_yaw = cumtrapz(yaw_rate, x=time, initial=0)

    # plt.figure(figsize=(10, 6))
    plt.figure()
    plt.plot(time, cumulative_yaw, color = 'red' )
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Yaw Angle (degrees)')
    plt.title('Cumulative Yaw Angle from Gyro Sensor Moving Data')
    plt.grid(True)
    
    #Low pass filter of magnetometer data, high pass filter of gyro data, complementary filter output, and IMU heading estimate as 4 subplots on one plot

    #mag_heading calculation
    mag_heading = np.arctan2(m_mag_field_y, m_mag_field_x)
    mag_heading = np.degrees(mag_heading)

    #gyro_heading
    rotational_rate_x = m_gyro_x
    rotational_rate_y = m_gyro_y
    rotational_rate_z = m_gyro_z
    rr_x = cumtrapz(rotational_rate_x, time, initial=0)
    rr_y = cumtrapz(rotational_rate_y, time, initial=0)
    rr_z = cumtrapz(rotational_rate_z, time, initial=0)

    gyro_heading = np.arctan2(rr_x, rr_y)
    gyro_heading = np.degrees(gyro_heading)

    #gyro_heading = np.degrees(rot_z)
    gyro_heading = gyro_heading+120 #Calibration offset
    for i in range(len(gyro_heading)): #Wrap angles
        if gyro_heading[i] > 180:
            gyro_heading[i] -= 360
        if gyro_heading[i] < -180:
            gyro_heading[i] += 360

    lowpassfreq_c = 0.2
    highpassfreq_c = 0.3
    nyquist_freq = 0.5*40
    order = 5
    b, a = signal.butter(order, lowpassfreq_c / nyquist_freq, 'low')
    c, d = signal.butter(order, highpassfreq_c / nyquist_freq, 'high')
    LowPassFreq_mag = signal.lfilter(b,a,mag_heading)
    HighPassFreq_gyro = signal.lfilter(c,d,gyro_heading)

    alpha = 0.8

    complementary_filter = [alpha * yaw_cal + (1-alpha) * yaw_est for yaw_cal, yaw_est in zip(LowPassFreq_mag, HighPassFreq_gyro)]
    # print(f"low pass filter {LowPassFreq_mag}")
    # print(f"high pass filter {HighPassFreq_gyro}")
    # print(f"complementary filter {complementary_filter}")
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(time, LowPassFreq_mag, label='LowPass-Filter of Magnetometer data')
    plt.title('Filter Results Moving Data')
    plt.xlabel('Time(s)')
    plt.ylabel('Heading(deg)')
    plt.grid(True)
    plt.legend()

    plt.subplot(4,1,2)
    plt.plot(time, HighPassFreq_gyro, label='HighPass-Filter of Gyro data')
    plt.xlabel('Time(s)')
    plt.ylabel('Heading(deg)')
    plt.grid(True)
    plt.legend()

    plt.subplot(4,1,3)
    plt.plot(time, complementary_filter, label='Complementary filter output')
    plt.xlabel('Time(s)')
    plt.ylabel('Heading(deg)')
    plt.grid(True)
    plt.legend()
    # plt.show()

    plt.subplot(4,1,4)
    #plt.plot(time, comp_filter, label='Complementary Filter / Sensor Fusion')
    plt.plot(time, gyro_heading, label='Gyro Yaw Heading')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (deg)')
    plt.grid(True)
    plt.legend()
    # plt.show()


def forward_velocity_estimate(m_accel_x, m_timestamps, mg_latitude, mg_longitude, mg_timestamps, mg_utm_eastings, mg_utm_northings):

    accel_x = m_accel_x 
    imu_time = m_timestamps             
    gps_lat = mg_latitude
    gps_lon = mg_longitude
    gps_time = mg_timestamps             

    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def highpass_filter(data, cutoff, fs, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    fs = 1 / np.mean(np.diff(imu_time)) 
    cutoff = 0.1  
    filtered_accel_x = highpass_filter(accel_x, cutoff, fs)

    velocity_imu_raw = cumtrapz(accel_x, imu_time, initial=0)
    velocity_imu = cumtrapz(filtered_accel_x, imu_time, initial=0)

    gps_velocity = []
    for i in range(1, len(gps_lat)):
        point1 = (gps_lat[i - 1], gps_lon[i - 1])
        point2 = (gps_lat[i], gps_lon[i])
        distance = geodesic(point1, point2).meters 
        time_interval = gps_time[i] - gps_time[i - 1]
        gps_velocity.append(distance / time_interval if time_interval != 0 else 0)
    gps_velocity = np.array([0] + gps_velocity) 

    gps_velocity_interpolated = interp1d(gps_time, gps_velocity, bounds_error=False, fill_value="extrapolate")(imu_time)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(imu_time, velocity_imu_raw, label="IMU Velocity (Raw)", color="blue") 
    plt.plot(imu_time, gps_velocity_interpolated, label="GPS Velocity (Interpolated)", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Forward Velocity Estimates (Raw IMU & Interpolated GPS)")
    plt.grid(True)
    plt.legend()
    plt.subplots_adjust(hspace=0.3)

    window_size = 50 
    gps_avg_velocity = np.convolve(gps_velocity_interpolated, np.ones(window_size)/window_size, mode='same')

    velocity_imu_corrected = velocity_imu + (gps_avg_velocity - np.mean(gps_avg_velocity[:100]))

    velocity_imu_corrected = np.clip(velocity_imu_corrected, 0, None)

    # plt.figure()
    plt.subplot(2, 1, 2)
    plt.plot(imu_time, velocity_imu_corrected, label="IMU Velocity (Corrected)", color="blue")
    plt.plot(imu_time, gps_velocity_interpolated, label="GPS Velocity (Interpolated)", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Forward Velocity Estimates (Corrected IMU & Interpolated GPS)")
    plt.legend()
    plt.grid(True)
    
    # Calculate the time differences
    dt = np.diff(mg_timestamps)
    
    # Calculate the distance differences in UTM coordinates
    dx = np.diff(mg_utm_eastings)
    dy = np.diff(mg_utm_northings)
    
    # Calculate the speed in meters per second
    distances = np.sqrt(dx**2 + dy**2)  # Euclidean distance in 2D
    velocity = distances / dt  # Speed in m/s
    
    plt.figure()
    plt.plot(mg_timestamps[1:], velocity)  # Skip the first timestamp for plotting:
    plt.xlabel('Time (s)')
    plt.ylabel('Forward Velocity (m/s)')
    plt.title('Forward Velocity from GPS Data')
    plt.grid(True)
    
    
def estimated_trajectory_gps_and_imu(m_accel_x, m_timestamps, mg_latitude, mg_longitude, mg_timestamps, m_gyro_z, m_accel_y):
    
    # Extract necessary columns
    accel_x = m_accel_x
    imu_time = m_timestamps
    gps_lat = mg_latitude
    gps_lon = mg_longitude
    gps_time = mg_timestamps

    # Step 1: High-Pass Filter to Remove Drift
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def highpass_filter(data, cutoff, fs, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    fs = 1 / np.mean(np.diff(imu_time))
    cutoff = 0.1
    filtered_accel_x = highpass_filter(accel_x, cutoff, fs)

    # Step 2: Integrate filtered acceleration to estimate forward velocity
    velocity_imu = cumtrapz(filtered_accel_x, imu_time, initial=0)

    # Step 3: Estimate GPS velocity
    gps_velocity = []
    for i in range(1, len(gps_lat)):
        point1 = (gps_lat[i - 1], gps_lon[i - 1])
        point2 = (gps_lat[i], gps_lon[i])
        distance = geodesic(point1, point2).meters
        time_interval = gps_time[i] - gps_time[i - 1]
        gps_velocity.append(distance / time_interval if time_interval != 0 else 0)
    gps_velocity = np.array([0] + gps_velocity)

    # Step 4: Interpolate GPS Velocity to Align with IMU Timestamps
    gps_velocity_interpolated = interp1d(gps_time, gps_velocity, bounds_error=False, fill_value="extrapolate")(imu_time)

    # Step 5: Apply correction to IMU-based velocity estimate
    window_size = 50
    gps_avg_velocity = np.convolve(gps_velocity_interpolated, np.ones(window_size) / window_size, mode='same')
    velocity_imu_corrected = velocity_imu + (gps_avg_velocity - np.mean(gps_avg_velocity[:100]))
    velocity_imu_corrected = np.clip(velocity_imu_corrected, 0, None)

    # Step 6: Calculate Heading
    heading_rad = np.zeros(len(imu_time))
    for i in range(1, len(imu_time)):
        dt = imu_time[i] - imu_time[i - 1]
        heading_rad[i] = heading_rad[i - 1] + m_gyro_z[i] * dt

    # Normalize heading to [0, 2*pi]
    heading_rad = np.mod(heading_rad, 2 * np.pi)

    # Step 7: Calculate velocity components
    ve = velocity_imu_corrected * np.cos(heading_rad)  # East-West component
    vn = velocity_imu_corrected * np.sin(heading_rad)  # North-South component

    # Step 8: Integrate (ve, vn) to estimate trajectory
    trajectory_easting = cumtrapz(ve, imu_time, initial=0)
    trajectory_northing = cumtrapz(vn, imu_time, initial=0)

    # Rotate IMU trajectory by 45 degrees
    theta = np.radians(90)  # Convert degrees to radians
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])

    # Stack trajectory coordinates for rotation
    trajectory_coords = np.vstack((trajectory_easting, trajectory_northing))

    # Apply rotation
    rotated_trajectory = rotation_matrix @ trajectory_coords

    # Flip the x-coordinate (Easting) to correct the mirror effect
    rotated_trajectory[0] = -rotated_trajectory[0]  # Negate easting

    # Extract rotated trajectory components
    trajectory_easting_rotated, trajectory_northing_rotated = rotated_trajectory

    # Step 9: GPS displacement calculation
    gps_displacement = np.zeros((len(gps_lat), 2))
    for i in range(1, len(gps_lat)):
        point1 = (gps_lat[i - 1], gps_lon[i - 1])
        point2 = (gps_lat[i], gps_lon[i])
        distance = geodesic(point1, point2).meters
        angle = np.arctan2(gps_lat[i] - gps_lat[i - 1], gps_lon[i] - gps_lon[i - 1])
        gps_displacement[i] = gps_displacement[i - 1] + np.array([distance * np.cos(angle), distance * np.sin(angle)])

    # Step 10: Plotting the trajectories
    # plt.figure(figsize=(10, 6))
    plt.figure()
    plt.subplot(3, 1, 1)
    # plt.plot(trajectory_easting_rotated, trajectory_northing_rotated, label='IMU Estimated Trajectory', color='blue')
    plt.plot(gps_displacement[:, 0], gps_displacement[:, 1], label='GPS Trajectory', color='red')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.title('Trajectory GPS')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.subplots_adjust(hspace=0.4)
    
    plt.subplot(3, 1, 2)
    plt.plot(trajectory_easting_rotated, trajectory_northing_rotated, label='IMU Estimated Trajectory', color='blue')
    # plt.plot(gps_displacement[:, 0], gps_displacement[:, 1], label='GPS Trajectory', color='red')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.title('Trajectory IMU')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.subplots_adjust(hspace=0.4)
    
    plt.subplot(3, 1, 3)
    plt.plot(trajectory_easting_rotated, trajectory_northing_rotated, label='IMU Estimated Trajectory', color='blue')
    plt.plot(gps_displacement[:, 0], gps_displacement[:, 1], label='GPS Trajectory', color='red')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.title('Trajectory Comparison:IMU vs GPS')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    
        
    
    # Normalize time
    time = imu_time - imu_time[0]
    dt = np.mean(np.diff(time))  # Time step

    # Assumptions: velocity in y (y') = 0, x_c = 0
    # x''_obs = x'' - ω*y' - ω^2*x_c => x'' = x''_obs
    # y''_obs = y'' - ω*x' + ω'*x_c => y'' = y''_obs (since y' = 0, x_c = 0)

    # Integrate acceleration to obtain forward velocity (x')
    forward_velocity = cumtrapz(accel_x, time, initial=0)

    # Compute ωx' and compare it with y''_obs
    omega_x = m_gyro_z * forward_velocity  # ωx'
    y_acc_obs = m_accel_y  # y''_obs from IMU

    # Plot ωx' vs y''_obs
    plt.figure(figsize=(10, 6))
    plt.plot(time, omega_x, label="ωx'")
    plt.plot(time, y_acc_obs, label="y''_obs")
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title("Comparison of ωx' and y''_obs")
    plt.legend()
    plt.grid()


if __name__ == '__main__':
    # Path to your rosbag file
    current_path = os.path.join(os.getcwd(), "../data/")
    circle_data_bag_path = os.path.join(current_path, 'circle_data_nuance.bag')
    moving_data_bag_path = os.path.join(current_path, 'moving_data_nuance.bag')
    # Load the rosbag
    c_timestamps, c_mag_field_x, c_mag_field_y, c_mag_field_z, c_gyro_x, c_gyro_y, c_gyro_z, c_accel_x, c_accel_y, c_accel_z  = read_imu_data(circle_data_bag_path)
    cg_timestamps, cg_utm_eastings, cg_utm_northings, cg_altitudes, cg_latitude, cg_longitude = read_gps_data(circle_data_bag_path)
    m_timestamps, m_mag_field_x, m_mag_field_y, m_mag_field_z, m_gyro_x, m_gyro_y, m_gyro_z, m_accel_x, m_accel_y, m_accel_z = read_imu_data(moving_data_bag_path)
    mg_timestamps, mg_utm_eastings, mg_utm_northings, mg_altitudes, mg_latitude, mg_longitude = read_gps_data(moving_data_bag_path)
    
    c_mag_field_x_hard, c_mag_field_y_hard, c_mag_field_z_hard, basis_circle = hard_iron_calibration(c_mag_field_x, c_mag_field_y, c_mag_field_z)
    c_mag_field_x_soft, c_mag_field_y_soft, c_mag_field_z_soft = soft_iron_calibration(c_mag_field_x_hard, c_mag_field_y_hard, c_mag_field_z_hard)
    c_fit_x_soft, c_fit_y_soft = fit_circle(c_mag_field_x_soft, c_mag_field_y_soft)
    plot_mag_field(c_mag_field_x, c_mag_field_y, c_mag_field_x_hard, c_mag_field_y_hard, c_mag_field_x_soft, c_mag_field_y_soft, c_fit_x_soft, c_fit_y_soft)
    
    c_mag_mean_x, c_mag_mean_y, c_mag_mean_z = np.mean(c_mag_field_x), np.mean(c_mag_field_y), np.mean(c_mag_field_z)
    m_mag_mean_x, m_mag_mean_y, m_mag_mean_z = np.mean(m_mag_field_x), np.mean(m_mag_field_y), np.mean(m_mag_field_z)
    
    ## PLOT2:The magnetometer yaw estimation before and after hard and soft iron calibration vs. time
    yaw_angle_estimate(c_mag_field_x, c_mag_field_y, c_mag_mean_x, c_mag_mean_y, c_mag_field_x_soft, c_mag_field_y_soft, 
                       m_mag_field_x, m_mag_field_y, m_mag_field_z, m_timestamps)
    
    ## PLOT3: The Gyro Yaw rate and comparison with the estimated yaw rate from magnetometer data
    gyro_yaw_rate(m_timestamps, m_gyro_z, m_mag_field_x, m_mag_field_y, m_gyro_x, m_gyro_y)
    
    forward_velocity_estimate(m_accel_x, m_timestamps, mg_latitude, mg_longitude, mg_timestamps, mg_utm_eastings, mg_utm_northings)
    
    estimated_trajectory_gps_and_imu(m_accel_x, m_timestamps, mg_latitude, mg_longitude, mg_timestamps, m_gyro_z, m_accel_y)
    
    # Show plot
    plt.tight_layout()
    plt.show()