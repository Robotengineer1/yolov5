from flask import Flask, render_template, request, jsonify, send_file
import cv2
import os
import subprocess
import base64
import json
from datetime import datetime
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['DOWNLOADS_FOLDER'] = 'downloads'  # New downloads folder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOADS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_yolo_detection(source, img_size, weights_path="/home/ibrahim-careyu/Desktop/Work/yolov5-Mainline/ws/best-RHS0.pt"):
    """Run YOLO detection with specified parameters"""
    try:
        # Construct the command
        cmd = [
            'python', 'detect.py',
            '--img', str(img_size),
            '--source', source,
            '--device', '0',
            '--weights', weights_path,
            '--save-txt',  # Save results as txt files - CRUCIAL for parsing
            '--save-conf',  # Save confidence scores
            '--project', app.config['RESULTS_FOLDER'],
            '--name', 'exp',
            '--exist-ok' ,
            '--hide-conf'
        ]
        
        print(f"Running YOLO command: {' '.join(cmd)}")
        
        # Run the detection
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        
        print(f"YOLO stdout: {result.stdout}")
        print(f"YOLO stderr: {result.stderr}")
        print(f"YOLO return code: {result.returncode}")
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
            
    except Exception as e:
        return False, str(e)

def get_class_names(weights_path="/home/ibrahim-careyu/Desktop/Work/yolov5-Mainline/ws/best-RHS0.pt"):
    """Get class names from the YOLO model"""
    try:
        import torch
        
        # Load the model to get class names
        model = torch.load(weights_path, map_location='cpu')
        
        # Try to get class names from model
        if 'model' in model and hasattr(model['model'], 'names'):
            class_names = model['model'].names
        elif 'names' in model:
            class_names = model['names']
        else:
            # Based on your YOLO output: "17 Presents" and class_id=1 in text files
            # This means: class_id=1 = "Present", class_id=0 = "Missing" 
            class_names = {0: 'Missing', 1: 'Present'}
            
        print(f"Raw class names from model: {class_names}")
        return class_names
        
    except Exception as e:
        print(f"Error reading class names from model: {e}")
        # Based on your actual data: class_id=1 = Present
        return {0: 'Missing', 1: 'Present'}

def parse_detection_results(results_folder, weights_path="/home/ibrahim-careyu/Desktop/Work/yolov5-Mainline/ws/best-RHS0.pt", image_filename=None):
    """Parse YOLO detection results and classify as Present/Missing"""
    detection_counts = {'Present': 0, 'Missing': 0}
    detections = []
    
    # Get actual class names from the model
    class_names = get_class_names(weights_path)
    print(f"Class names from model: {class_names}")
    
    try:
        # Look for label files in the results folder
        labels_folder = os.path.join(results_folder, 'labels')
        print(f"Looking for labels in: {labels_folder}")
        print(f"Labels folder exists: {os.path.exists(labels_folder)}")
        
        if not os.path.exists(labels_folder):
            print("No labels folder found - this means --save-txt might not be working")
            return detection_counts, detections
        
        # If we have a specific image filename, look for its corresponding label file
        target_label_file = None
        if image_filename:
            # Get base name without extension
            base_name = os.path.splitext(image_filename)[0]
            target_label_file = base_name + '.txt'
            print(f"Looking for specific label file: {target_label_file}")
        
        label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
        print(f"Found label files: {label_files}")
        
        # If we have a target label file, use only that one
        if target_label_file and target_label_file in label_files:
            label_files_to_process = [target_label_file]
            print(f"Processing specific label file: {target_label_file}")
        else:
            # Otherwise, process the most recent label file
            if label_files:
                # Sort by modification time to get the latest
                label_files.sort(key=lambda x: os.path.getmtime(os.path.join(labels_folder, x)), reverse=True)
                label_files_to_process = [label_files[0]]
                print(f"Processing latest label file: {label_files[0]}")
            else:
                print("No label files found")
                return detection_counts, detections
        
        for label_file in label_files_to_process:
            label_path = os.path.join(labels_folder, label_file)
            print(f"Processing label file: {label_path}")
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            print(f"Label file content ({len(lines)} detections):")
            for i, line in enumerate(lines[:5]):  # Show first 5 lines
                print(f"  Line {i+1}: {line.strip()}")
            if len(lines) > 5:
                print(f"  ... and {len(lines) - 5} more lines")
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    confidence = float(parts[5]) if len(parts) > 5 else 0.0
                    
                    # Use the class names directly from model
                    if class_id in class_names:
                        status = class_names[class_id]
                    else:
                        # Fallback based on your data: class_id=1 means Present
                        status = 'Present' if class_id == 1 else 'Missing'
                    
                    print(f"Detection: class_id={class_id}, mapped to={status}, confidence={confidence}")
                    
                    detection_counts[status] += 1
                    
                    detections.append({
                        'class': status,
                        'class_id': class_id,
                        'class_name': class_names.get(class_id, f'class_{class_id}'),
                        'confidence': confidence,
                        'bbox': [float(x) for x in parts[1:5]]
                    })
        
        print(f"Final detection counts: {detection_counts}")
        print(f"Total detections found: {sum(detection_counts.values())}")
                        
    except Exception as e:
        print(f"Error parsing results: {e}")
        
    return detection_counts, detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    """Capture image from camera with specified resolution"""
    try:
        data = request.get_json()
        resolution = data.get('resolution', '640')
        img_size = int(resolution)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution (using single value for both width and height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_size)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size)
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'error': 'Failed to capture image'})
        
        # Save captured image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, frame)
        
        # Convert to base64 for display
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'filename': filename,
            'resolution': img_size
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"upload_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and encode image for display
            img = cv2.imread(filepath)
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': img_base64,
                'filename': filename,
                'resolution': img.shape[1]  # Return width as single value
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid file type'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Run YOLO detection on the image"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        img_size = int(data.get('resolution', '640'))
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'})
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        # Run YOLO detection
        success, output = run_yolo_detection(filepath, img_size)
        
        if not success:
            return jsonify({'success': False, 'error': f'Detection failed: {output}'})
        
        # Find the result image (get the most recent one)
        results_exp_folder = os.path.join(app.config['RESULTS_FOLDER'], 'exp')
        result_images = [f for f in os.listdir(results_exp_folder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not result_images:
            return jsonify({'success': False, 'error': 'No result image found'})
        
        # Sort by modification time to get the latest
        result_images.sort(key=lambda x: os.path.getmtime(os.path.join(results_exp_folder, x)), reverse=True)
        result_image_path = os.path.join(results_exp_folder, result_images[0])
        
        print(f"Using result image: {result_image_path}")
        
        # Force refresh by adding timestamp to avoid browser caching
        import time
        current_time = int(time.time())
        
        # Read and encode result image
        result_img = cv2.imread(result_image_path)
        if result_img is None:
            return jsonify({'success': False, 'error': 'Could not read result image'})
        
        print(f"Result image shape: {result_img.shape}")
        
        # Try different encoding methods
        encode_success, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not encode_success:
            # Try PNG encoding as fallback
            encode_success, buffer = cv2.imencode('.png', result_img)
            if not encode_success:
                return jsonify({'success': False, 'error': 'Failed to encode result image'})
        
        result_img_base64 = base64.b64encode(buffer).decode('utf-8')
        print(f"Base64 image length: {len(result_img_base64)}")
        
        # Verify base64 encoding
        if len(result_img_base64) < 100:
            return jsonify({'success': False, 'error': 'Result image encoding too small, likely corrupted'})
            
        # Parse detection results using the same image filename for consistency
        image_filename = os.path.basename(result_image_path)
        detection_counts, detections = parse_detection_results(results_exp_folder, "/home/ibrahim-careyu/Desktop/Work/yolov5-Mainline/ws/best-RHS0.pt", image_filename)
        
        print(f"Returning detection results: counts={detection_counts}, total={sum(detection_counts.values())}")
        
        return jsonify({
            'success': True,
            'detected_image': result_img_base64,
            'counts': detection_counts,
            'detections': detections,
            'total_detections': sum(detection_counts.values()),
            'result_image_path': result_image_path,
            'timestamp': current_time,
            'image_size': result_img.shape,
            'image_filename': image_filename,
            'debug_info': f"Image: {image_filename}, Labels parsed: {len(detections)}"
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_camera_resolutions')
def get_camera_resolutions():
    """Get available camera resolutions"""
    try:
        # Default resolutions without camera testing to avoid issues
        resolutions = [640, 800, 1024, 1280, 1600, 1920, 2048, 2560, 3840]
        
        # Try to test camera if available
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                # Test a few key resolutions
                tested_resolutions = []
                for size in [640, 1280, 1920, 2048]:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size)
                    
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    
                    if actual_width >= size * 0.8:  # Allow some tolerance
                        tested_resolutions.append(size)
                
                if tested_resolutions:
                    resolutions = tested_resolutions
                    
            cap.release()
        except:
            # If camera testing fails, use default resolutions
            pass
            
        return jsonify({'success': True, 'resolutions': resolutions})
        
    except Exception as e:
        # Return default resolutions even on error
        return jsonify({'success': True, 'resolutions': [640, 1280, 1920, 2048]})

@app.route('/clear_results', methods=['POST'])
def clear_results():
    """Clear old detection results to avoid confusion"""
    try:
        results_exp_folder = os.path.join(app.config['RESULTS_FOLDER'], 'exp')
        
        if os.path.exists(results_exp_folder):
            # Remove all files in results folder
            for filename in os.listdir(results_exp_folder):
                file_path = os.path.join(results_exp_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            
            return jsonify({'success': True, 'message': 'Results cleared successfully'})
        else:
            return jsonify({'success': True, 'message': 'No results folder to clear'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/check_results')
def check_results():
    """Debug route to check what files exist in results folder"""
    try:
        results_exp_folder = os.path.join(app.config['RESULTS_FOLDER'], 'exp')
        
        debug_info = {
            'results_folder_exists': os.path.exists(results_exp_folder),
            'files_in_results': [],
            'image_files': [],
            'label_files': []
        }
        
        if os.path.exists(results_exp_folder):
            all_files = os.listdir(results_exp_folder)
            debug_info['files_in_results'] = all_files
            
            # Check for image files
            for file in all_files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(results_exp_folder, file)
                    file_size = os.path.getsize(file_path)
                    debug_info['image_files'].append({
                        'name': file,
                        'size': file_size,
                        'path': file_path,
                        'readable': os.path.isfile(file_path)
                    })
            
            # Check labels folder
            labels_folder = os.path.join(results_exp_folder, 'labels')
            if os.path.exists(labels_folder):
                label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
                debug_info['label_files'] = label_files
        
        return jsonify({'success': True, 'debug_info': debug_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/debug_classes')
def debug_classes():
    """Debug route to check class names and recent detection results"""
    try:
        # Get class names from model
        class_names = get_class_names("/home/ibrahim-careyu/Desktop/Work/yolov5-Mainline/ws/best-RHS0.pt")
        
        # Check recent detection files
        results_exp_folder = os.path.join(app.config['RESULTS_FOLDER'], 'exp')
        debug_info = {
            'model_classes': class_names,
            'recent_detections': []
        }
        
        if os.path.exists(results_exp_folder):
            labels_folder = os.path.join(results_exp_folder, 'labels')
            if os.path.exists(labels_folder):
                label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
                for label_file in label_files[-3:]:  # Last 3 files
                    label_path = os.path.join(labels_folder, label_file)
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                        debug_info['recent_detections'].append({
                            'file': label_file,
                            'content': content
                        })
        
        return jsonify({'success': True, 'debug_info': debug_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_result')
def download_result():
    """Save the latest detection result image to downloads folder"""
    try:
        results_exp_folder = os.path.join(app.config['RESULTS_FOLDER'], 'exp')
        if not os.path.exists(results_exp_folder):
            return jsonify({'success': False, 'error': 'No results folder found'})
        
        # Find the latest result image
        result_images = [f for f in os.listdir(results_exp_folder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not result_images:
            return jsonify({'success': False, 'error': 'No result image found'})
        
        # Get the latest result image by modification time
        result_images.sort(key=lambda x: os.path.getmtime(os.path.join(results_exp_folder, x)), reverse=True)
        latest_result = result_images[0]
        source_path = os.path.join(results_exp_folder, latest_result)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_filename = f"detection_result_{timestamp}.jpg"
        destination_path = os.path.join(app.config['DOWNLOADS_FOLDER'], download_filename)
        
        # Copy the file to downloads folder
        shutil.copy2(source_path, destination_path)
        
        # Also copy the corresponding label file if it exists
        label_source = os.path.join(results_exp_folder, 'labels', latest_result.rsplit('.', 1)[0] + '.txt')
        if os.path.exists(label_source):
            label_destination = os.path.join(app.config['DOWNLOADS_FOLDER'], f"detection_labels_{timestamp}.txt")
            shutil.copy2(label_source, label_destination)
        
        return jsonify({
            'success': True, 
            'message': f'Results saved to downloads folder',
            'image_path': destination_path,
            'filename': download_filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)