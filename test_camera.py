import pyrealsense2 as rs
import time
import sys

# Try to get version information (safely)
try:
    version = rs.__version__
except AttributeError:
    version = "Unknown (no __version__ attribute)"
print(f"Using pyrealsense2 version: {version}")

# Create a context object
ctx = rs.context()

# Wait for device connection
print("Waiting for RealSense device to connect...")
try:
    devices = ctx.query_devices()
    print(f"Found {len(devices)} device(s)")
    
    # Instead of iterating through devices (which can cause the power state error),
    # just try to start the pipeline directly
    
except RuntimeError as e:
    if "failed to set power state" in str(e):
        print("Error: Failed to set power state. This is a common issue on macOS.")
        print("Please try running the script with sudo:")
        print("sudo python3 view_camera.py")
    else:
        print(f"Error: {e}")
    sys.exit(1)

# Try to start a simple pipeline
try:
    print("Attempting to start pipeline...")
    pipe = rs.pipeline()
    cfg = rs.config()
    
    # Try to enable just the depth stream at a low resolution
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    print("Starting pipeline...")
    profile = pipe.start(cfg)
    print("Pipeline started successfully!")
    
    # Get and print the depth sensor
    depth_sensor = profile.get_device().first_depth_sensor()
    print(f"Depth sensor: {depth_sensor.get_info(rs.camera_info.name)}")
    
    # Wait for a few frames
    print("Waiting for frames...")
    for i in range(10):
        frames = pipe.wait_for_frames(timeout_ms=1000)
        if frames:
            depth = frames.get_depth_frame()
            if depth:
                print(f"Received depth frame {i+1}, distance to center: "
                      f"{depth.get_distance(depth.get_width()//2, depth.get_height()//2):.2f}m")
    
    # Stop the pipeline
    pipe.stop()
    print("Pipeline stopped")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()