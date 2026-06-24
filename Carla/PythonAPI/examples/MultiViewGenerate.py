import carla
import random
import time
import math
import os
import numpy as np
import cv2

# Connect to CARLA.
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprints = world.get_blueprint_library()

# Create output directories.
output_dir_vi = 'output/vi_camera_view_120'
os.makedirs(output_dir_vi, exist_ok=True)
output_dir_ir = 'output/ir_camera_view_120'
os.makedirs(output_dir_ir, exist_ok=True)

# Define the initial vehicle pose.
reference_location = carla.Location(x=200.0, y=100.0, z=0.0)
vehicle_rotation = carla.Rotation(yaw=180.0)  # Face west.
vehicle_transform = carla.Transform(reference_location, vehicle_rotation)

# Spawn the vehicle.
vehicle_bp = random.choice(blueprints.filter('vehicle.jeep.wrangler_rubicon'))
vehicle_actor = world.spawn_actor(vehicle_bp, vehicle_transform)
vehicle_actor.set_autopilot(False)  # Disable autopilot.

# Configure the spectator and RGB camera.
spectator = world.get_spectator()
spectator_location = reference_location + carla.Location(x=-4, y=8, z=1)
spectator_rotation = carla.Rotation(pitch=0, yaw=vehicle_rotation.yaw + 120)
spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

camera_bp = blueprints.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(spectator_location, spectator_rotation)
camera = world.spawn_actor(camera_bp, camera_transform)


def save_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb_img = array[:, :, :3].copy()

    filename = f"{output_dir_vi}/{image.frame:06d}.png"
    cv2.imwrite(filename, rgb_img)

    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    filename = f"{output_dir_ir}/{image.frame:06d}.png"
    cv2.imwrite(filename, gray_img)


camera.listen(save_image)

# Control the vehicle with a constant forward speed.
try:
    speed = 20.0
    start_time = time.time()

    while time.time() - start_time < 10:
        yaw_rad = math.radians(vehicle_actor.get_transform().rotation.yaw)
        vx = speed * math.cos(yaw_rad)
        vy = speed * math.sin(yaw_rad)

        vehicle_actor.set_target_velocity(carla.Vector3D(vx, vy, 0))
        time.sleep(0.05)

finally:
    # Clean up spawned actors.
    camera.stop()
    camera.destroy()
    vehicle_actor.destroy()
    print("Done.")
