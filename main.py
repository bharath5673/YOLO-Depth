import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm



# Load MiDaS model
model_type = "DPT_Hybrid"
# model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform


# Open the video file
video_path = 'NuScenes/360_view/CAM_FRONT.mp4'
cap = cv2.VideoCapture(video_path)

# Output video file
output_video_path = 'output_depth_CAM_FRONT_pip.mp4'
# Video settings
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


def picture_in_picture(main_image, overlay_image, img_ratio=3, border_size=3, x_margin=30, y_offset_adjust=-100):
    """
    Overlay an image onto a main image with a white border.
    
    Args:
        main_image_path (str): Path to the main image.
        overlay_image_path (str): Path to the overlay image.
        img_ratio (int): The ratio to resize the overlay image height relative to the main image.
        border_size (int): Thickness of the white border around the overlay image.
        x_margin (int): Margin from the right edge of the main image.
        y_offset_adjust (int): Adjustment for vertical offset.

    Returns:
        np.ndarray: The resulting image with the overlay applied.
    """
    # Load images
    if main_image is None or overlay_image is None:
        raise FileNotFoundError("One or both images not found.")

    # Resize the overlay image to 1/img_ratio of the main image height
    new_height = main_image.shape[0] // img_ratio
    new_width = int(new_height * (overlay_image.shape[1] / overlay_image.shape[0]))
    overlay_resized = cv2.resize(overlay_image, (new_width, new_height))

    # Add a white border to the overlay image
    overlay_with_border = cv2.copyMakeBorder(
        overlay_resized,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    # Determine overlay position
    x_offset = main_image.shape[1] - overlay_with_border.shape[1] - x_margin
    y_offset = (main_image.shape[0] // 2) - overlay_with_border.shape[0] + y_offset_adjust

    # Overlay the image
    main_image[y_offset:y_offset + overlay_with_border.shape[0], x_offset:x_offset + overlay_with_border.shape[1]] = overlay_with_border

    return main_image





# Variables for FPS calculation
frameId = 0
start_time = time.time()
fps = str()
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

while True:
    frameId+=1

    ret, frame = cap.read()
    if not ret:
        break
    img = frame.copy()
    image = frame.copy()

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Transform the image for MiDaS
    input_batch = transform(img).to(device)

    interpolation_mode = 'bilinear'  ### Change to 'nearest', 'area', or 'bicubic' or "bilinear" or 'trilinear' or 'linear' as needed 

    # Run MiDaS Depth
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode=interpolation_mode,
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    # Normalize the depth map to [0, 1]
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    # Invert the depth map
    depth_map = 1.0 - depth_map_normalized


    # Run YOLO Detector1
    model = YOLO('yolov8n-seg.pt')  # load an official model
    image = img.copy()
    original_size = image.shape[:2]
    # Predict with the model
    results = model.predict(image, verbose=False, device=0)  # predict on an image
    for predictions in results:
        if predictions is None:
            continue  # Skip this image if YOLO fails to detect any objects
        if predictions.boxes is None or predictions.masks is None:
            continue  # Skip this image if there are no boxes or masks
        for bbox, masks in zip(predictions.boxes, predictions.masks):
            for scores, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                xmin    = bbox_coords[0]
                ymin    = bbox_coords[1]
                xmax    = bbox_coords[2]
                ymax    = bbox_coords[3]
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)
                # cv2.rectangle(depth_map, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 1)

                # Get the depth values within the bounding box
                depth_values_bbox = depth_map[int(ymin):int(ymax), int(xmin):int(xmax)]
                depth_value = np.median(depth_values_bbox)

                # # Estimate distance in meters using the depth value (adjust the scale factor as needed)
                scale_factor = 15  # Adjust this based on the scale of your depth map
                distance = depth_value * scale_factor
                # distance = distance//10000
                # distance = depth_value


                overlay_frame = image.copy()
                # Set text properties
                font_scale = 0.4
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_color = (255, 255, 255)  # White text
                background_color = (30, 30, 30)  # Dark rectangle background
                line_spacing = int(15 * font_scale)  # Adjust line spacing based on font scale
                text_x = (int(xmin)) + 5
                text_y = int(ymin) + 5

                # Calculate the maximum width and total height needed for the background rectangle
                text_lines = [str(predictions.names[int(classes)]), str(round(float(scores) * 100, 1))+ '%', f'Dist: {distance:.3f} m']
                text_sizes = [cv2.getTextSize(line, font, font_scale, 1)[0] for line in text_lines]
                max_width = max(w for w, h in text_sizes) + 3
                total_height = sum(h for w, h in text_sizes) + (len(text_lines) - 1) * line_spacing + 3
                # Draw background rectangle
                cv2.rectangle(image, (text_x - 5, text_y - text_sizes[0][1] - 5),
                              (text_x + max_width, text_y + total_height - 5),
                              background_color, cv2.FILLED)

                # Draw each line of text on top of the background
                for i, line in enumerate(text_lines):
                    line_y = text_y + i * (text_sizes[i][1] + line_spacing)
                    cv2.putText(image, line, (text_x, line_y), font, font_scale, font_color, 1)

                image = cv2.addWeighted(overlay_frame, 0.5, image, 1 - 0.5, 0)



        masks = predictions.masks
        if masks is not None:
            for mask in masks.xy:
                polygon = mask
                cv2.polylines(image, [np.int32(polygon)], True, (255, 0, 0), thickness=1)



    # Display FPS on both frames
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if frameId % 10 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps_current = 10 / elapsed_time  # Calculate FPS over the last 10 frames
        fps = f'FPS: {fps_current:.2f}'
        start_time = time.time()  # Reset start_time for the next 10 frames


    # Use 'plasma' colormap for depth visualization
    depth_map_colored = plt.cm.plasma(depth_map / depth_map.max())[:, :, :3]
    depth_map_colored = (depth_map_colored * 255).astype(np.uint8)

    image = picture_in_picture(image, depth_map_colored)
    # Overlay YOLO detections on the frame
    cv2.putText(image, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('YOLO Object Detection', image)
    out.write(image)


    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update tqdm progress bar
    progress_bar.update(1)

# Close tqdm progress bar
progress_bar.close()
# Release the video capture object
cap.release()
out.release()
# Destroy all OpenCV windows
cv2.destroyAllWindows()
