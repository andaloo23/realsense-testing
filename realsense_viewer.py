import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import time

class RealSenseViewer:
    def __init__(self):
        # Initialize configuration variables
        self.show_depth = True
        self.show_color = True
        self.depth_colormap = cv2.COLORMAP_JET
        self.depth_scale = 0.03  # Scale factor for depth visualization
        self.mirror = False
        self.record = False
        self.recording_file = None
        self.playback_file = None
        self.filter_depth = False
        self.depth_min = 0.1  # 10cm
        self.depth_max = 10.0  # 10m
        self.resolution = (640, 480)
        self.fps = 30
        self.point_cloud = False
        self.colormap_options = {
            0: cv2.COLORMAP_JET,
            1: cv2.COLORMAP_BONE,
            2: cv2.COLORMAP_HOT,
            3: cv2.COLORMAP_RAINBOW,
            4: cv2.COLORMAP_OCEAN,
            5: cv2.COLORMAP_WINTER,
            6: cv2.COLORMAP_PLASMA,
            7: cv2.COLORMAP_VIRIDIS
        }
        self.current_colormap = 0
        
    def setup_pipeline(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Check if playback file is specified
        if self.playback_file:
            # Configure from file
            print(f"Playing back from file: {self.playback_file}")
            self.config.enable_device_from_file(self.playback_file)
        else:
            # Configure live streams
            self.config.enable_stream(rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.resolution[0], self.resolution[1], rs.format.bgr8, self.fps)
        
        # Configure recording if enabled
        if self.record:
            filename = f"realsense_recording_{time.strftime('%Y%m%d_%H%M%S')}.bag"
            self.config.enable_record_to_file(filename)
            print(f"Recording to file: {filename}")
        
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Get depth scale for distance calculations
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.real_depth_scale = self.depth_sensor.get_depth_scale()
        print(f"Depth Scale: {self.real_depth_scale}")
        
        # Setup post-processing filters
        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        
        # Setup point cloud
        self.pc = rs.pointcloud()
        self.points = rs.points()
        
        # Create alignment object
        self.align = rs.align(rs.stream.color)
        
        # Print camera info
        device = self.profile.get_device()
        print(f"Using device: {device.get_info(rs.camera_info.name)}")
        print(f"Serial number: {device.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware version: {device.get_info(rs.camera_info.firmware_version)}")
        
    def process_frames(self):
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        
        # Align depth to color frame
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # Apply filters if enabled
        if self.filter_depth and depth_frame:
            depth_frame = self.decimation.process(depth_frame)
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)
        
        # Generate point cloud if enabled
        if self.point_cloud and depth_frame and color_frame:
            self.pc.map_to(color_frame)
            self.points = self.pc.calculate(depth_frame)
        
        return depth_frame, color_frame
    
    def create_visualization(self, depth_frame, color_frame):
        # Convert frames to numpy arrays
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Apply depth filtering
            if self.filter_depth:
                depth_meters = depth_image * self.real_depth_scale
                depth_image = np.where(
                    (depth_meters >= self.depth_min) & (depth_meters <= self.depth_max),
                    depth_image,
                    0
                )
            
            # Apply colormap
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=self.depth_scale),
                self.colormap_options[self.current_colormap]
            )
        else:
            depth_colormap = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
        else:
            color_image = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Mirror images if enabled
        if self.mirror:
            if depth_frame:
                depth_colormap = cv2.flip(depth_colormap, 1)
            if color_frame:
                color_image = cv2.flip(color_image, 1)
        
        # Create display based on enabled streams
        if self.show_depth and self.show_color:
            # Show both side by side
            display_image = np.hstack((color_image, depth_colormap))
        elif self.show_depth:
            display_image = depth_colormap
        elif self.show_color:
            display_image = color_image
        else:
            # Default to color if nothing selected
            display_image = color_image
        
        # Add distance information
        if depth_frame:
            distance = depth_frame.get_distance(depth_frame.get_width() // 2, depth_frame.get_height() // 2)
            cv2.putText(
                display_image,
                f"Distance to center: {distance:.2f}m",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        # Add recording indicator
        if self.record:
            cv2.putText(
                display_image,
                "REC",
                (display_image.shape[1] - 70, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        return display_image
    
    def display_controls(self):
        print("\nKeyboard Controls:")
        print("  ESC/Q - Quit")
        print("  D - Toggle depth view")
        print("  C - Toggle color view")
        print("  M - Toggle mirror")
        print("  R - Toggle recording")
        print("  F - Toggle depth filtering")
        print("  + - Increase depth scale")
        print("  - - Decrease depth scale")
        print("  P - Toggle point cloud (experimental)")
        print("  [ - Decrease min depth")
        print("  ] - Increase min depth")
        print("  ; - Decrease max depth")
        print("  ' - Increase max depth")
        print("  TAB - Cycle through colormaps")
    
    def handle_key(self, key):
        if key == 27 or key == ord('q'):  # ESC or Q
            return False
        elif key == ord('d'):
            self.show_depth = not self.show_depth
            print(f"Depth view: {'ON' if self.show_depth else 'OFF'}")
        elif key == ord('c'):
            self.show_color = not self.show_color
            print(f"Color view: {'ON' if self.show_color else 'OFF'}")
        elif key == ord('m'):
            self.mirror = not self.mirror
            print(f"Mirror: {'ON' if self.mirror else 'OFF'}")
        elif key == ord('r'):
            # Can't toggle recording after pipeline start
            print("Recording can only be set at startup")
        elif key == ord('f'):
            self.filter_depth = not self.filter_depth
            print(f"Depth filtering: {'ON' if self.filter_depth else 'OFF'}")
        elif key == ord('+'):
            self.depth_scale += 0.01
            print(f"Depth scale: {self.depth_scale:.2f}")
        elif key == ord('-'):
            self.depth_scale = max(0.01, self.depth_scale - 0.01)
            print(f"Depth scale: {self.depth_scale:.2f}")
        elif key == ord('p'):
            self.point_cloud = not self.point_cloud
            print(f"Point cloud: {'ON' if self.point_cloud else 'OFF'}")
        elif key == ord('['):
            self.depth_min = max(0.0, self.depth_min - 0.1)
            print(f"Min depth: {self.depth_min:.1f}m")
        elif key == ord(']'):
            self.depth_min += 0.1
            print(f"Min depth: {self.depth_min:.1f}m")
        elif key == ord(';'):
            self.depth_max = max(self.depth_min + 0.1, self.depth_max - 0.1)
            print(f"Max depth: {self.depth_max:.1f}m")
        elif key == ord('\''):
            self.depth_max += 0.1
            print(f"Max depth: {self.depth_max:.1f}m")
        elif key == 9:  # TAB
            self.current_colormap = (self.current_colormap + 1) % len(self.colormap_options)
            colormap_names = ["JET", "BONE", "HOT", "RAINBOW", "OCEAN", "WINTER", "PLASMA", "VIRIDIS"]
            print(f"Colormap: {colormap_names[self.current_colormap]}")
        return True
    
    def run(self):
        try:
            # Setup pipeline
            self.setup_pipeline()
            
            # Display controls
            self.display_controls()
            
            # Create window
            cv2.namedWindow('RealSense Viewer', cv2.WINDOW_AUTOSIZE)
            
            # Main loop
            while True:
                # Process frames
                depth_frame, color_frame = self.process_frames()
                
                # Create visualization
                display_image = self.create_visualization(depth_frame, color_frame)
                
                # Show image
                cv2.imshow('RealSense Viewer', display_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                if not self.handle_key(key):
                    break
                
        finally:
            # Stop streaming
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("Viewer closed")

def parse_args():
    parser = argparse.ArgumentParser(description='RealSense Camera Viewer')
    parser.add_argument('--record', action='store_true', help='Record to a bag file')
    parser.add_argument('--playback', type=str, help='Play back from a bag file')
    parser.add_argument('--resolution', type=str, default='640x480', help='Resolution (WIDTHxHEIGHT)')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create viewer
    viewer = RealSenseViewer()
    
    # Configure from arguments
    viewer.record = args.record
    viewer.playback_file = args.playback
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        viewer.resolution = (width, height)
    except:
        print(f"Invalid resolution format: {args.resolution}. Using default 640x480.")
    
    viewer.fps = args.fps
    
    # Run viewer
    viewer.run()