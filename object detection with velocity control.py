import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import pyrealsense2 as rs
import numpy as np

import serial
import time
import pyfirmata2

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=np.RankWarning)

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
slowsspeed = 150
currentspeed=0.0
increment = 10.0
go_count = 0
max_go_count = 0

stop_distance = 4
slow_distance = 7 #velocity control

frame = cv2.imread('obstacledetection/yolov7modified/snakeroad.jpg')  # Replace 'your_image.jpg' with the path to your image
# cv2.imshow("frame",frame)
# frame = cv2.resize(frame, (640, 480))

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

def vehicle_go():
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
        
        if currentspeed < maxspeed:
            currentspeed+=increment
            if currentspeed > maxspeed:
                currentspeed = maxspeed
        
        accn_pwm.write(currentspeed/255)
        accn_1.write(1)
        accn_2.write(0)
        print(f'currentspeed = {currentspeed}')


def vehicle_slow():
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
        
        if currentspeed > slowsspeed:
            currentspeed-=increment 
            if currentspeed < slowsspeed:
                currentspeed = slowsspeed
        
        accn_pwm.write(currentspeed/255)
        accn_1.write(1)
        accn_2.write(0)
        print(f'currentspeed = {currentspeed}')

    


def cleanup_resources(pipeline):
    cv2.destroyAllWindows()
    if pipeline:
        pipeline.stop()

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 7
    blur = cv2.GaussianBlur(gray, (kernel, kernel), sigmaX=0, sigmaY=0)
    canny = cv2.Canny(blur,80,100)
    # canny = cv2.Canny(blur,100,160)
    return canny


def region_of_interest_trapezium(img):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros_like(img)

    # Define coordinates for the trapezium
    # Adjust the points (x1, y1), (x2, y2), (x3, y3), (x4, y4) as needed
    bottom_left = (0, height)
    top_left = (0, height * 0.5)
    top_right = (width, height * 0.5)
    bottom_right = (width, height)

    # bottom_left = (width * 0.1, height)
    # top_left = (width * 0.4, height * 0.6)
    # top_right = (width * 0.6, height * 0.6)
    # bottom_right = (width * 0.9, height)

    # np.array expects points as [[first_point, second_point, third_point, fourth_point]]
    trapezium = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)

    # Fill the polygon (trapezium here) with white (255)
    cv2.fillPoly(mask, [trapezium], 255)

    # Apply the mask
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def houghLines(img):
    houghLines = cv2.HoughLinesP(img, 2, np.pi / 180, 10, np.array([]), minLineLength=30, maxLineGap=10)
    return houghLines

def preprocess_frame(frame):
    # new_width = 640
    # new_height = 480
    # aspect_ratio = original_width / original_height
    # # Calculate the new height maintaining the aspect ratio
    # new_height = int(new_width / aspect_ratio)
    # frame = cv2.resize(frame, (new_width, new_height))
    # Convert frame to float
    frame_float = frame.astype(np.float32)
    # Calculate the mean of the pixel values
    mean = np.mean(frame_float)
    # Scale factor for contrast adjustment; values < 1.0 decrease contrast
    # scale_factor = 1.0
    scale_factor = 0.8
    # Adjust the contrast
    # Moving pixel values towards the mean to reduce contrast
    frame_adjusted = (frame_float - mean) * scale_factor + mean
    # Clip values to stay between 0 and 255 and convert back to uint8
    frame = np.clip(frame_adjusted, 0, 255).astype(np.uint8)
    return frame


def display_filled_region(img, lines, init_point):
    img_copy = img.copy()
    mask = np.zeros_like(img)
    if lines is not None:
        left_line = lines[0][0]  # Assuming first line is the left
        right_line = lines[1][0]  # Assuming second line is the right

        pts = np.array([[left_line[0], left_line[1]], [left_line[2], left_line[3]],
                        [right_line[2], right_line[3]], [right_line[0], right_line[1]]], np.int32)

        bottom_left_corner = pts[1]
        bottom_right_corner = pts[2]

        top_left_corner = [bottom_left_corner[0], bottom_left_corner[1] - 300]
        top_right_corner = [bottom_right_corner[0], bottom_right_corner[1] - 300]

        pts2 = np.array([bottom_left_corner, bottom_right_corner, top_right_corner, top_left_corner], np.int32)
        pts2 = pts2.reshape((-1, 1, 2))

        pts = pts.reshape((-1, 1, 2))
        image = cv2.fillPoly(img_copy, [pts], (144, 238, 144))  # Light green color
        image = cv2.fillPoly(img_copy, [pts2], (144, 238, 144))  # Light green color
        
        # image = cv2.polylines(img_copy, [pts], isClosed=True, color=(144, 238, 144), thickness=5)
        # image = cv2.polylines(img_copy, [pts2], isClosed=True, color=(144, 238, 144), thickness=5) 

    return image


def make_points(img, lineSI):
    slope, intercept = lineSI
    height = img.shape[0]
    y1 = int(height)
    y2 = int(y1 * 3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(img,lines):
    # lower_value_slope = 0.5
    # higher_value_slope = 3
    lower_value_slope = 0.5
    higher_value_slope = 2.5
    flag_left = True
    flag_right = True
    left_fit = []
    right_fit = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            fit = np.polyfit((x1,x2),(y1,y2),1)
            slope = fit[0]
            intercept = fit[1]
            # print(intercept,slope)
            if slope < -lower_value_slope and slope >= -higher_value_slope:
                left_fit.append((slope, intercept))
            elif slope >= lower_value_slope and slope <= higher_value_slope :
                right_fit.append((slope,intercept))
    if left_fit == []:
        # left_fit = np.array([(0.0001,0.0001)])
        flag_left = False
    if right_fit == []:
        # right_fit = np.array([(0.0001,0.0001)])
        flag_right = False

    if flag_left:
        left_fit_average = np.average(left_fit,axis=0)
        left_line = make_points(img, left_fit_average)
    else:
        left_line = np.array([[None,None,None,None]])

    if flag_right:
        right_fit_average = np.average(right_fit,axis=0)
        right_line = make_points(img, right_fit_average)
    else:
        right_line = np.array([[None,None,None,None]])


    average_lines = [np.array(left_line),np.array(right_line)]
    return average_lines



def average_slope_intercept_with_centre(img, lines):
    lower_value_slope = 0.5
    higher_value_slope = 2.5
    flag_left = True
    flag_right = True
    left_fit = []
    right_fit = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Check if any of the points are None
            if any(p is None for p in [x1, y1, x2, y2]):
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < -lower_value_slope and slope >= -higher_value_slope:
                left_fit.append((slope, intercept))
            elif slope >= lower_value_slope and slope <= higher_value_slope:
                right_fit.append((slope, intercept))

    if not left_fit:
        flag_left = False
    if not right_fit:
        flag_right = False

    if flag_left:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(img, left_fit_average)
    else:
        # left_line = np.array([[None, None, None, None]])
        left_line = np.array([])

    if flag_right:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(img, right_fit_average)
    else:
        # right_line = np.array([[None, None, None, None]])
        right_line = np.array([])

    if flag_left and flag_right:
        center_line = np.array([[0, 0, 0, 0]])
        for i in range(4):
            center_line[0][i] = np.int32((left_line[0][i] + right_line[0][i]) / 2)
    else:
        # center_line = np.array([[None, None, None, None]])
        center_line = np.array([])

    average_lines = [np.array(left_line), np.array(right_line), np.array(center_line)]
    return average_lines


def update_line_history(line_history, new_line, history_length=5):
    if new_line[0].all() == np.array([None,None,None,None]).all() and line_history:
        # Use the most recent valid line if the new line is invalid
        new_line = line_history[-1]
    line_history.append(new_line)
    if len(line_history) > history_length:
        line_history.pop(0)
    # print(line_history)
    return line_history


def average_line_from_history(line_history):
    if not line_history:
        return np.array([0, 0, 0, 0])
    avg_line = np.mean(np.array(line_history), axis=0, dtype=np.int32)
    return avg_line


def lane_detection(frame):
    mask = frame.copy()
    left_line_history = []
    right_line_history = []
    history_length = 15
    init_point = (23, 384)

    try:
        canny_output = canny(frame)
        masked_output = region_of_interest_trapezium(canny_output)
        lines = houghLines(masked_output)
        average_lines = average_slope_intercept(frame,lines)
        average_lines_with_centre_avg = average_slope_intercept_with_centre(frame, lines)

        left_line = average_lines_with_centre_avg[0]
        right_line = average_lines_with_centre_avg[1]

        left_line_history = update_line_history(left_line_history, left_line, history_length)
        right_line_history = update_line_history(right_line_history, right_line, history_length)
        left_line_avg = average_line_from_history(left_line_history)
        right_line_avg = average_line_from_history(right_line_history)
        center_line_avg = np.array([[0, 0, 0, 0]])
        for i in range(4):
            center_line_avg[0][i] = np.int32((left_line_avg[0][i] + right_line_avg[0][i]) / 2)

        average_lines_with_centre_avg = np.array(
            [np.array(left_line_avg), np.array(right_line_avg), np.array(center_line_avg)])

        line_image_2_filled = display_filled_region(frame, average_lines_with_centre_avg, init_point)
        line_mask_filled = display_filled_region(mask, average_lines_with_centre_avg, init_point)
    except Exception as e:
        print("Error:", e)
        line_image_2_filled = frame
        line_mask_filled = frame

    return line_image_2_filled, line_mask_filled

def check_intersection(masked_img1, masked_img2):
    # Resize masked_img2 to match the dimensions of masked_img1
    masked_img2_resized = cv2.resize(masked_img2, (masked_img1.shape[1], masked_img1.shape[0]))
    intersection = cv2.bitwise_and(masked_img1, masked_img2)
    intersects = False
    if np.any(intersection != 0):
        intersects = True
        # print("Obstacle Entered Lane")
    return intersection, intersects


def detection():
    # board = Arduino('COM8')
    pipeline = None
    try:
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.visual_preset, 1)
        colorizer.set_option(rs.option.histogram_equalization_enabled, 1.0)  # disable histogram equalization
        colorizer.set_option(rs.option.color_scheme, 0)  # replace 'float' with your desired color scheme
        colorizer.set_option(rs.option.min_distance, 0.2)  # replace 'float' with your desired min distance
        colorizer.set_option(rs.option.max_distance, 4)  # replace 'float' with your desired max distance 

        source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA  
        # Load model
        print(device)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        # model = attempt_load(weights, map_location=torch.device('cpu')) 
        #problem^^^^^^^^^^^^^^^^^^^^
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        # if trace:
        #     model = TracedModel(model, device, opt.img_size)
        # if half:
        #     model.half()  # to FP16
        model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    

        pipeline = rs.pipeline()

        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)  ## imu data
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        profile = pipeline.start(config)


        # align_to = rs.stream.color
        # align = rs.align(align_to)

        # Set colorizer options
    
        while True:

            # frames = pipeline.wait_for_frames()
            # color_frame = frames.get_color_frame()

            # if not color_frame:
            #     continue

            # frame = np.asanyarray(color_frame.get_data())
            # lane_masked_image = lane_detection(frame)
            # obstacle_masked_image = obstacle_detection(pipeline, colorizer, device, half, names, model, colors)

            #t0 = time.time()
            # aligned_frames = align.process(frames)
            aligned_frames=pipeline.wait_for_frames()
            color_frame = aligned_frames.get_color_frame()
            lane_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            # if not depth_frame or not color_frame:
            #     continue

            # Get IMU data
            accel_frame = aligned_frames.first_or_default(rs.stream.accel)
            gyro_frame = aligned_frames.first_or_default(rs.stream.gyro)

            if accel_frame and gyro_frame:
                accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()

            # Print IMU data (Accelerometer and Gyroscope)
            print(f"Accelerometer - X: {accel_data.x:.2f}, Y: {accel_data.y:.2f}, Z: {accel_data.z:.2f}")
            print(f"Gyroscope - X: {gyro_data.x:.2f}, Y: {gyro_data.y:.2f}, Z: {gyro_data.z:.2f}")
    



            img = np.asanyarray(color_frame.get_data())
            lane = np.asanyarray(lane_frame.get_data())
            # depth_image = np.asanyarray(depth_frame.get_data())
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
            colorized_frame = colorizer.colorize(depth_frame)

            # Get lane masking for road
            frames = pipeline.wait_for_frames()
            # if color_frame is not None:
            #     color_image = color_image.reshape((480, 640, 3))

            img = preprocess_frame(img)
            lane_image, lane_masked_image = lane_detection(img)   #for live frame
            # lane_img = preprocess_frame(frame)
            # lane_masked_image = lane_detection(lane_img) #for fixed frame
            
            # cv2.imshow("color frame", color_frame)

            # Convert the colorized frame to a numpy array
            depth_colormap = np.asanyarray(colorized_frame.get_data())
            # Letterbox
            im0 = img.copy()
            img = img[np.newaxis, :, :, :]

            # Create a zero mask image
            im0_masked = np.zeros_like(im0)

            # Stack
            img = np.stack(img, 0)

            # Convert
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            img = np.ascontiguousarray(img)


            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # print(len(pred[0]))

            if len(pred[0])==0:
                print("Path is clear!! (No obstacles)")
                vehicle_go()
                # ser.write(b'stoop\n')

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write resul
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)

                    # Initialize a list to store distances of all detected objects
                    object_distances = []

                    for *xyxy, _, _ in det:
                        indv_mask = np.zeros_like(im0_masked)
                        # cv2.imshow("Masks", indv_mask)

                        # Draw bounding boxes on the zero mask image
                        cv2.rectangle(indv_mask, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 255, 255), -1)
                        
                        # Check for intersection only if both images are non-empty
                        # if lane_masked_image.size != 0 and indv_mask.size != 0:
                        intersection, intersects = check_intersection(lane_masked_image, indv_mask)

                        # else:
                        #     intersection = np.zeros_like(im0_masked)
                        #     intersects = False

                        # Display the stacked image
                        # cv2.imshow("Intersection", intersection)
                        
                        if intersects:
                            c = int(cls)  # integer class
                            label = f'{names[c]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                            depth_data = np.array(depth_frame.get_data())

                            # Extract bounding box coordinates
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                            # Get depth data within the bounding box
                            depth_region = depth_data[y1:y2, x1:x2]
                            
                            if np.count_nonzero(depth_region) > 0:
                                # Calculate the minimum distance within the bounding box
                                object_min_distance = np.min(depth_region[depth_region != 0]) * 0.001
                                object_distances.append(object_min_distance)
                            
                            else:
                                object_min_distance = float('inf')

                                # Add the distance to the list

                        else:
                            print("Path is clear!!")
                            vehicle_go()
                            # ser.write(b'stoop\n')

                    # After iterating through all detected objects, find the minimum distance
                    if object_distances:
                        min_distance = min(object_distances)
                    # Check if the minimum distance is less than a threshold
                        if min_distance < stop_distance:
                            print(f"Obstacle detected within {min_distance:.2f} meters!")
                            vehicle_stop()
                            # ser.write(b'stop\n')
                        if stop_distance < min_distance and min_distance < slow_distance:  
                            print(f"Obstacle detected within {min_distance:.2f} meters!")
                            vehicle_slow()
                        else:
                            print("Path is clear!!")
                            vehicle_go()
                            # ser.write(b'stoop\n')
                        # print(f"The minimum distance among all detected objects is: {min_distance:.2f} meters")
                    else:
                        print("No objects detected.")
                        vehicle_go()
                        # ser.write(b'stoop\n')

                # Print time (inference + NMS)
                #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                cv2.imshow("Recognition result", im0)
                # cv2.imshow("Recognition result depth",depth_colormap)
                # cv2.imshow("Masked frame", im0_masked)  # Display the masked image

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Display images for debugging
            cv2.imshow("Lane Masked Image", lane_image)
            # cv2.imshow("Obstacle Masked Image", im0_masked)

            # Check if any of the images are empty
            if lane_masked_image.size == 0:
                print("Error: Lane Masked Image is empty")
            if im0_masked.size == 0:
                print("Error: Obstacle Masked Image is empty")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # cleanup_resources(pipeline)
        cv2.destroyAllWindows()
    
    # board.exit()


if __name__ == '__main__':
    # app.run(debug=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    # Replace [0, 1, 2] with the indices of the classes you want to detect
    classes_to_detect = [0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 19, 24, 25, 32, 58, 56]

    # Set the classes argument to the specified classes_to_detect
    opt.classes = classes_to_detect
    # device = select_device(opt.device)
    # model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    # names = model.module.names if hasattr(model, 'module') else model.names

    # # Print classes and their indices
    # print("Classes and their indices:")
    # for i, name in enumerate(names):
    #     print(f"Index: {i}, Class: {name}")

    with torch.no_grad():
        detection()