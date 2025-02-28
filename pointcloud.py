import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import os

# Create a directory to save the files if it doesn't exist
output_dir = "output_files"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream depth and color
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
print("Starting RealSense pipeline...")
pipeline.start(config)

# Create an align object
# Align depth frame to color frame
align = rs.align(rs.stream.color)

# Wait for the camera to warm up
print("Waiting for camera to initialize...")
for _ in range(30):
    pipeline.wait_for_frames()

# Capture a single frame
print("Capturing frame...")
frames = pipeline.wait_for_frames()

# Align depth frame to color frame
aligned_frames = align.process(frames)
color_frame = aligned_frames.get_color_frame()
depth_frame = aligned_frames.get_depth_frame()

if not color_frame or not depth_frame:
    print("Error: Could not get frames")
    exit(1)

# Convert to numpy arrays
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

# Save the color image as a PNG file
color_image_filename = os.path.join(output_dir, "color_image.png")
cv2.imwrite(color_image_filename, color_image)
print(f"Color image saved as {color_image_filename}")

# Save the depth image as a PNG file (for visualization)
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
depth_image_filename = os.path.join(output_dir, "depth_image.png")
cv2.imwrite(depth_image_filename, depth_colormap)
print(f"Depth image saved as {depth_image_filename}")

# Get camera intrinsics
depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
width, height = depth_image.shape[1], depth_image.shape[0]

# Convert depth image to 3D point cloud with colors
points = []
colors = []

# Skip some pixels for better performance (adjust step as needed)
step = 2

for v in range(0, height, step):
    for u in range(0, width, step):
        depth = depth_image[v, u] * 0.001  # Scale depth to meters
        if depth == 0 or depth > 5:  # Ignore invalid depth points or points too far away
            continue

        # Compute 3D coordinates in camera frame
        x = (u - depth_intrin.ppx) * depth / depth_intrin.fx
        y = (v - depth_intrin.ppy) * depth / depth_intrin.fy
        z = depth
        points.append([x, y, z])
        
        # Get the color for this point (BGR to RGB)
        color = color_image[v, u][::-1] / 255.0  # Convert BGR to RGB and normalize to [0,1]
        colors.append(color)

# print intrinsics
print(depth_intrin.ppx, depth_intrin.ppy, depth_intrin.fx, depth_intrin.fy)

# Convert the points and colors to numpy arrays
points = np.array(points)
colors = np.array(colors)

print(f"Generated point cloud with {len(points)} points")

# Create a point cloud from the 3D points with colors
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)  # Add colors to the point cloud

# Optional: Remove outliers for cleaner visualization
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"After outlier removal: {len(np.asarray(pcd.points))} points")

# Save the point cloud as a PLY file
point_cloud_filename = os.path.join(output_dir, "colored_pointcloud.ply")
o3d.io.write_point_cloud(point_cloud_filename, pcd)
print(f"Colored 3D model saved as {point_cloud_filename}")

# Visualize the point cloud with a more informative message
print("\nViewing point cloud...")
print("Controls:")
print("  Left mouse button + drag: Rotate")
print("  Right mouse button + drag: Pan")
print("  Mouse wheel: Zoom")
print("  ['[' and ']']: Change point size")
o3d.visualization.draw_geometries([pcd], 
                                  window_name="Colored Point Cloud", 
                                  width=1024, 
                                  height=768,
                                  point_show_normal=False)

# Stop the pipeline
pipeline.stop()
print("Pipeline stopped")
