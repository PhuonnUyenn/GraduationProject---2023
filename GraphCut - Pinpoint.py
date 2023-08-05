import csv
import time
import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt
import os
#from Evaluate_seg import evaluation_metrics
from skimage.measure import regionprops

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
#ap.add_argument('-m', '--mask', required=True,
                #help='path to yolo config file')
args = ap.parse_args()


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


image = cv2.imread(args.image)
org_image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

# Thực hiện xác định bằng HOG và SVM
start = time.time()

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

tumor_detect = cv2.imshow("object detection", image)

end = time.time()
print("YOLO Execution time: " + str(end-start))
print("bbox = [xmin: {}, ymin: {}, xmax: {}, ymax: {}]".format(round(x), round(y), round(x + w), round(y + h)))

cv2.waitKey()

cv2.imwrite("dect-te.jpg", image)
cv2.destroyAllWindows()

# Xác định bounding box
xmin = int(round(x))
ymin = int(round(y))
xmax = int(round(x+w))
ymax = int(round(y+h))

bbox = (xmin, ymin, xmax, ymax)
bbox = tuple(map(int, bbox))
rect = (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])

# Tạo mask với kích thước bằng kích thước ảnh, giá trị 0 ban đầu
mask = np.zeros(org_image.shape[:2], np.uint8)
#cv2.imshow("mask detection", mask)
#cv2.imwrite("mask detection-te.png", mask)

# Đặt giá trị 3 cho vùng bên trong bounding box
mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 3

# Tạo điểm "pinpoint" cho vùng đối tượng (tại chỗ giữa bounding box)
cv2.circle(mask, (round(rect[2]/2+rect[0]), round(rect[3]/2+rect[1])), 10, cv2.GC_FGD, -1)
#R = 15 --> me
#R = 10 --> pi
# Đặt điểm "pinpoint" cho vùng nền (xung quanh vùng đối tượng)
cv2.circle(mask, (100, 100), 5, cv2.GC_BGD, -1)

# Thực hiện phân đoạn GrabCut-pinpoint
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(org_image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

# Xác định vùng nền và vùng đối tượng
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img_masked = org_image * mask[:, :, np.newaxis]
#print(img_masked.shape)
plt.imshow(img_masked), plt.colorbar(), plt.show()
#cv2.imwrite('Te-me_0010.png', img_masked)
kernel = np.ones((3,3), np.uint8)
img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel, iterations = 2)

#folder_path = r"E:\DO AN TOT NGHIEP\22 - 23 FINAL PROJECTTTT\Implement\ResultSeg\gli"

#cv2.imwrite('Te-me_0010.png', img_masked)

# Chuyển sang ảnh Gray
img_grayscale = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

# Thiết lập ngưỡng và chuyển sang ảnh Binary
##binary_img = cv2.threshold(img_masked, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
thresh, binary_img = cv2.threshold(img_grayscale, 0, 255, cv2.THRESH_BINARY)
cv2.imwrite('img_bin.png', binary_img)
#print(binary_img.shape)
#w = binary_img.shape[0]
#h = binary_img.shape[1]

plt.imshow(binary_img), plt.colorbar(), plt.show()

# Tính diện tích của vùng pixel trắng trong ảnh nhị phân
##num_white_1 = cv2.countNonZero(binary_img)
num_white = np.sum(binary_img == 255)
props = regionprops(binary_img)
area = props[0].area

# Hiển thị diện tích của vùng pixel màu trắng
print("Diện tích vùng pixel màu trắng là:", num_white)
print("Diện tích u não là:", area)

#total = w*h
#print("Total pixel: ", total)

# Tìm tất cả các đường viền có trong ảnh
contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Vẽ đường biên cho từng đối tượng trong ảnh
for contour in contours:
    cv2.drawContours(org_image, [contour], 0, (0, 255, 0), 3)
    #cv2.imwrite('img_draw.png', org_image)
    cv2.fillPoly(org_image, [contour], (0, 255, 0))
    cv2.imwrite('img_full.png', org_image)
    #props = regionprops(org_image)
    #area = props[0].area
    #print("Diện tích u não là:", area)
plt.imshow(org_image), plt.colorbar(), plt.show()

#def evaluation_metrics(segmented_img, mask_img):

    #segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    #mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    # Resize ảnh về chung kích thước
    #segmented_img = cv2.resize(segmented_img, (512, 512))
    #mask_img = cv2.resize(mask_img, (512, 512))

    # Thực hiện ngưỡng hóa (threshold bin va Otsu) để chuyển đổi ảnh về dạng binary
    #_, segmented_img_binary = cv2.threshold(segmented_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #_, mask_img_binary = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Tính số lượng true positive (TP), false positive (FP) và false negative (FN) pixel
    # Tính số lượng pixel dự đoán đúng dương
    #TP = cv2.countNonZero(segmented_img_binary & mask_img_binary)
    #print("True Positive: ", TP,"pixels")
    # Tính số lượng pixel dự đoán sai âm
    #FP = cv2.countNonZero(segmented_img_binary & ~mask_img_binary)
    #print("False Positive: ", FP,"pixels")
    # Tính số lượng pixel dự đoán đúng âm
    #TN = cv2.countNonZero(~segmented_img_binary & ~mask_img_binary)
    #print("True Negative: ", TN,"pixels")
    # Tính số lượng pixel dự đoán sai dương
    #FN = cv2.countNonZero(~segmented_img_binary & mask_img_binary)
    #print("False Negative: ", FN,"pixels")

    # Tính precision (độ chính xác), recall (độ phủ), F1-score và accuracy (độ chính xác toàn phần)
    #precision = TP / (TP + FP) * 100 if TP + FP > 0 else 0
    #recall = TP / (TP + FN) * 100 if TP + FN > 0 else 0
    #f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    #accuracy = (TP + cv2.countNonZero(~segmented_img_binary & ~mask_img_binary)) / (segmented_img.shape[0] * segmented_img.shape[1]) * 100
    #accuracy = (TP + TN) / (TP + TN + FP + FN) * 100 if TP + TN + FP + FN > 0 else 0

    #print("Accuracy: ", accuracy,"%")
    #print("Precision: ", precision,"%")
    #print("Recall: ", recall,"%")
    #print("F1-score: ", f1_score,"%")

    # Lưu kết quả vào danh sách evaluation_results
    #evaluation_results.append(["Te-pi_0025.png", accuracy, precision, recall, f1_score])

    #return precision, recall, f1_score, accuracy

#mask_img = cv2.imread(args.mask)

#cv2.imwrite(os.path.join(folder_path, "Te-gl_0025.png"), img_masked)
#csv_file = r"E:\DO AN TOT NGHIEP\22 - 23 FINAL PROJECTTTT\Implement\Grabcut_GLI.csv"
##writer.writerow(["File name", "accuracy", "precision", "recall", "f1_score"])
#with open(csv_file, mode='a', newline="") as csvfile:
    #writer = csv.writer(csvfile)
    #evaluation_results = []
    #evaluation_metrics(img_masked, mask_img)
    #for i in evaluation_results:
        #writer.writerow(i)