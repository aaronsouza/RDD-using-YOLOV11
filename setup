#DOWNLOAD DATASET FROM ROBOFLOW
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="bCFryYsEHyQ3MJEGY9Ml")
project = rf.workspace("projetos-fensd").project("analise-de-buracos-na-rua")
version = project.version(1)
dataset = version.download("yolov11")




#SHOW DATA SAMPLE
import cv2
import random
import glob as glob
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2] / 2, bboxes[1] - bboxes[3] / 2
    xmax, ymax = bboxes[0]+bboxes[2] / 2, bboxes[1] + bboxes[3] / 2
    return xmin, ymin, xmax, ymax

class_names = ['Buraco']
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

def plot_box(image, bboxes, labels):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        
        # denormalize the coordinates
        xmin = int(x1 * w)
        ymin = int(y1 * h)
        xmax = int(x2 * w)
        ymax = int(y2 * h)
        width = xmax - xmin
        height = ymax - ymin
        
        class_name = class_names[int(labels[box_num])]
        
        cv2.rectangle(
            image, 
            (xmin, ymin), (xmax, ymax),
            color=colors[class_names.index(class_name)],
            thickness=2
        ) 

        font_scale = min(1,max(3,int(w/500)))
        font_thickness = min(2, max(10,int(w/50)))
        
        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        
        # Text width and height
        tw, th = cv2.getTextSize(class_name, 0, fontScale=font_scale, thickness=font_thickness)[0]
        p2 = p1[0] + tw, p1[1] + -th - 10
        cv2.rectangle(
            image, 
            p1, p2,
            color=colors[class_names.index(class_name)],
            thickness=-1,
        )
        cv2.putText(
            image, 
            class_name,
            (xmin+1, ymin-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    return image

    


    

#FUNCTION TO PLOT IMAGES WITH BOUNDING BOXES
def plot(image_paths, label_paths, num_samples):
    all_training_images = glob.glob(image_paths)
    all_training_labels = glob.glob(label_paths)
    all_training_images.sort()
    all_training_labels.sort()
    
    num_images = len(all_training_images)
    
    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = random.randint(0,num_images-1)
        image = cv2.imread(all_training_images[j])
        with open(all_training_labels[j], 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.show()

plot(image_paths='/kaggle/working/Analise-de-buracos-na-rua-1/train/images/*',
label_paths='/kaggle/working/Analise-de-buracos-na-rua-1/train/labels/*',
num_samples=4,
)



#APPLY YOLOv11
!pip install ultralytics
from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")
# Use the model
results = model.train(data="/kaggle/working/Analise-de-buracos-na-rua-1/data.yaml", epochs=100)  # train the model




#PREDICTION
!yolo task=detect mode=predict model=/kaggle/working/runs/detect/train2/weights/best.pt source='/kaggle/working/Analise-de-buracos-na-rua-1/test/images/po_179_jpg.rf.ad5df95fea1f31d3114bd31a2360b3b0.jpg'
img_path = '/kaggle/working/runs/detect/predict/po_179_jpg.rf.ad5df95fea1f31d3114bd31a2360b3b0.jpg'
img = plt.imread(img_path)

plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis(False)
plt.show()











