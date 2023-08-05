
import cv2
import sys
import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1425, 803)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("QMainWindow {\n"
"      background-color:rgb(121, 182, 182);\n"
"     color: rgb(255, 255, 255)\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralwidget.setStyleSheet("QWidget {\n"
"    background-color:rgb(121, 182, 182);\n"
"    color: rgb(255, 255, 255)\n"
"}\n"
"")
        self.centralwidget.setObjectName("centralwidget")
        self.anhchon = QtWidgets.QLabel(self.centralwidget)
        self.anhchon.setGeometry(QtCore.QRect(250, 120, 391, 271))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.anhchon.sizePolicy().hasHeightForWidth())
        self.anhchon.setSizePolicy(sizePolicy)
        self.anhchon.setStyleSheet("QLabel {\n"
"    background-color:rgb(255, 255, 255);\n"
"    color:rgb(0, 0, 154);\n"
"    font-size: 33px;\n"
"}")
        self.anhchon.setFrameShape(QtWidgets.QFrame.Box)
        self.anhchon.setAlignment(QtCore.Qt.AlignCenter)
        self.anhchon.setObjectName("anhchon")
        self.anhkhoiu = QtWidgets.QLabel(self.centralwidget)
        self.anhkhoiu.setGeometry(QtCore.QRect(250, 410, 391, 271))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.anhkhoiu.sizePolicy().hasHeightForWidth())
        self.anhkhoiu.setSizePolicy(sizePolicy)
        self.anhkhoiu.setStyleSheet("QLabel {\n"
"    background-color:rgb(255, 255, 255);\n"
"    color:rgb(0, 0, 154);\n"
"    font-size: 33px\"Times New Roman\";\n"
"}\n"
"")
        self.anhkhoiu.setFrameShape(QtWidgets.QFrame.Box)
        self.anhkhoiu.setAlignment(QtCore.Qt.AlignCenter)
        self.anhkhoiu.setObjectName("anhkhoiu")
        self.anhphathien = QtWidgets.QLabel(self.centralwidget)
        self.anhphathien.setGeometry(QtCore.QRect(670, 120, 391, 271))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.anhphathien.sizePolicy().hasHeightForWidth())
        self.anhphathien.setSizePolicy(sizePolicy)
        self.anhphathien.setStyleSheet("QLabel {\n"
"    background-color:rgb(255, 255, 255);\n"
"    color:rgb(0, 0, 154);\n"
"    font-size: 33px\"Times New Roman\";\n"
"}")
        self.anhphathien.setFrameShape(QtWidgets.QFrame.Box)
        self.anhphathien.setAlignment(QtCore.Qt.AlignCenter)
        self.anhphathien.setObjectName("anhphathien")
        self.anhkhoiu_2 = QtWidgets.QLabel(self.centralwidget)
        self.anhkhoiu_2.setGeometry(QtCore.QRect(670, 410, 391, 271))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.anhkhoiu_2.sizePolicy().hasHeightForWidth())
        self.anhkhoiu_2.setSizePolicy(sizePolicy)
        self.anhkhoiu_2.setStyleSheet("QLabel {\n"
"    background-color:rgb(255, 255, 255);\n"
"    color:rgb(0, 0, 154);\n"
"    font-size: 33px\"Times New Roman\";\n"
"}")
        self.anhkhoiu_2.setFrameShape(QtWidgets.QFrame.Box)
        self.anhkhoiu_2.setAlignment(QtCore.Qt.AlignCenter)
        self.anhkhoiu_2.setObjectName("anhkhoiu_2")
        self.kqphanloai = QtWidgets.QLabel(self.centralwidget)
        self.kqphanloai.setGeometry(QtCore.QRect(1100, 260, 211, 21))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.kqphanloai.sizePolicy().hasHeightForWidth())
        self.kqphanloai.setSizePolicy(sizePolicy)
        self.kqphanloai.setAutoFillBackground(False)
        self.kqphanloai.setStyleSheet("QLabel {\n"
"    background-color: rgb(55, 165, 165);\n"
"    color: rgb(0, 0, 0);\n"
"    font-size: 16px\"Times New Roman\";\n"
"}")
        self.kqphanloai.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.kqphanloai.setScaledContents(False)
        self.kqphanloai.setObjectName("kqphanloai")
        self.hienthikqphanloai = QtWidgets.QLabel(self.centralwidget)
        self.hienthikqphanloai.setGeometry(QtCore.QRect(1100, 290, 211, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hienthikqphanloai.sizePolicy().hasHeightForWidth())
        self.hienthikqphanloai.setSizePolicy(sizePolicy)
        self.hienthikqphanloai.setAutoFillBackground(False)
        self.hienthikqphanloai.setStyleSheet("QLabel {\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    font-size: 16px\"Times New Roman\";\n"
"}")
        self.hienthikqphanloai.setFrameShape(QtWidgets.QFrame.Panel)
        self.hienthikqphanloai.setObjectName("hienthikqphanloai")
        self.Dientichkhoiu = QtWidgets.QLabel(self.centralwidget)
        self.Dientichkhoiu.setGeometry(QtCore.QRect(1100, 340, 211, 21))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Dientichkhoiu.sizePolicy().hasHeightForWidth())
        self.Dientichkhoiu.setSizePolicy(sizePolicy)
        self.Dientichkhoiu.setStyleSheet("QLabel {\n"
"    background-color: rgb(55, 165, 165);\n"
"    color: rgb(0, 0, 0);\n"
"    font-size: 16px\"Times New Roman\";\n"
"}")
        self.Dientichkhoiu.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Dientichkhoiu.setObjectName("Dientichkhoiu")
        self.hienthiS = QtWidgets.QLabel(self.centralwidget)
        self.hienthiS.setGeometry(QtCore.QRect(1100, 370, 211, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hienthiS.sizePolicy().hasHeightForWidth())
        self.hienthiS.setSizePolicy(sizePolicy)
        self.hienthiS.setStyleSheet("QLabel {\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    font-size: 16px\"Times New Roman\";\n"
"}")
        self.hienthiS.setFrameShape(QtWidgets.QFrame.Panel)
        self.hienthiS.setObjectName("hienthiS")
        self.chonanh = QtWidgets.QPushButton(self.centralwidget)
        self.chonanh.setEnabled(True)
        self.chonanh.setGeometry(QtCore.QRect(1100, 450, 209, 27))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.chonanh.sizePolicy().hasHeightForWidth())
        self.chonanh.setSizePolicy(sizePolicy)
        self.chonanh.setStyleSheet("QPushButton {\n"
"    background-color:rgb(0, 125, 125);\n"
"    color: rgb(255, 255, 255);\n"
"    font-size: 16px\"Times New Roman\";\n"
"\n"
"\n"
"\n"
"\n"
"\n"
"}\n"
" ")
        self.chonanh.setObjectName("chonanh")
        self.phanloai = QtWidgets.QPushButton(self.centralwidget)
        self.phanloai.setGeometry(QtCore.QRect(1100, 510, 209, 27))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.phanloai.sizePolicy().hasHeightForWidth())
        self.phanloai.setSizePolicy(sizePolicy)
        self.phanloai.setStyleSheet("QPushButton {\n"
"    background-color:rgb(0, 125, 125);\n"
"    color: rgb(255, 255, 255);\n"
"    font-size: 16px\"Times New Roman\";\n"
"}\n"
"")
        self.phanloai.setObjectName("phanloai")
        self.phanvung = QtWidgets.QPushButton(self.centralwidget)
        self.phanvung.setGeometry(QtCore.QRect(1100, 570, 209, 27))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.phanvung.sizePolicy().hasHeightForWidth())
        self.phanvung.setSizePolicy(sizePolicy)
        self.phanvung.setStyleSheet("QPushButton {\n"
"    background-color:rgb(0, 125, 125);\n"
"    color: rgb(255, 255, 255);\n"
"    font-size: 16px\"Times New Roman\";\n"
"}\n"
"")
        self.phanvung.setObjectName("phanvung")
        self.xoa = QtWidgets.QPushButton(self.centralwidget)
        self.xoa.setGeometry(QtCore.QRect(1100, 630, 209, 27))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.xoa.sizePolicy().hasHeightForWidth())
        self.xoa.setSizePolicy(sizePolicy)
        self.xoa.setStyleSheet("QPushButton {\n"
"    background-color:rgb(0, 125, 125);\n"
"    color: rgb(255, 255, 255);\n"
"    font-size: 16px\"Times New Roman\";\n"
"}\n"
"")
        self.xoa.setObjectName("xoa")
        self.tenanh = QtWidgets.QLabel(self.centralwidget)
        self.tenanh.setGeometry(QtCore.QRect(1100, 120, 211, 21))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tenanh.sizePolicy().hasHeightForWidth())
        self.tenanh.setSizePolicy(sizePolicy)
        self.tenanh.setAutoFillBackground(False)
        self.tenanh.setStyleSheet("QLabel {\n"
"    background-color: rgb(55, 165, 165);\n"
"    color: rgb(0, 0, 0);\n"
"    font-size: 16px\"Times New Roman\";\n"
"}")
        self.tenanh.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tenanh.setScaledContents(False)
        self.tenanh.setObjectName("tenanh")
        self.hienthitenanh = QtWidgets.QLabel(self.centralwidget)
        self.hienthitenanh.setGeometry(QtCore.QRect(1100, 150, 211, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hienthitenanh.sizePolicy().hasHeightForWidth())
        self.hienthitenanh.setSizePolicy(sizePolicy)
        self.hienthitenanh.setAutoFillBackground(False)
        self.hienthitenanh.setStyleSheet("QLabel {\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    font-size: 16px\"Times New Roman\";\n"
"}")
        self.hienthitenanh.setFrameShape(QtWidgets.QFrame.Panel)
        self.hienthitenanh.setObjectName("hienthitenanh")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(-4, 0, 1381, 111))
        self.label_2.setStyleSheet("QLabel {\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    font-size: 16px;\n"
"}")
        self.label_2.setText("")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(360, -10, 631, 51))
        self.label_3.setStyleSheet("QLabel {\n"
"    background-color:rgb(255, 255, 255);\n"
"    color:rgb(255, 0, 0);\n"
"    font-size: 23px;\n"
"}")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(235, 60, 861, 51))
        self.label_4.setStyleSheet("QLabel {\n"
"    background-color:rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 154);\n"
"    font-size: 33px;\n"
"    font: 75 21pt \"Times New Roman\";\n"
"}")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(60, 10, 131, 101))
        self.label_5.setText("")
        self.label_5.setPixmap(QtGui.QPixmap("E:/DO_AN_TOT_NGHIEP/TEST_2label/truonglogo.jpg"))
        self.label_5.setScaledContents(True)
        self.label_5.setObjectName("label_5")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1230, 10, 111, 101))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("E:/DO_AN_TOT_NGHIEP/TEST_2label/download.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(1100, 10, 121, 101))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap("E:/DO_AN_TOT_NGHIEP/TEST_2label/Logo khoa dien.png"))
        self.label_6.setScaledContents(True)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(380, 30, 551, 31))
        self.label_7.setStyleSheet("QLabel {\n"
"    background-color:rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 154);\n"
"    font-size: 20px;\n"
"    font: 75 19pt \"Times New Roman\";\n"
"}")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1425, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.chonanh.clicked.connect(self.select_image)
        self.phanloai.clicked.connect(self.process_image)
        self.phanvung.clicked.connect(self.compute_area)
        self.xoa.clicked.connect(self.reset)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CLASSIFICATION & CALCULATION"))
        self.anhchon.setText(_translate("MainWindow", "Selected Image"))
        self.anhkhoiu.setText(_translate("MainWindow", "Segmentation Image"))
        self.anhphathien.setText(_translate("MainWindow", "Classification Image "))
        self.anhkhoiu_2.setText(_translate("MainWindow", "Brain Tumor Image"))
        self.kqphanloai.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Result of classification:</span></p></body></html>"))
        self.hienthikqphanloai.setText(_translate("MainWindow", ""))
        self.Dientichkhoiu.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Area of the tumor:</span></p></body></html>"))
        self.hienthiS.setText(_translate("MainWindow", ""))
        self.chonanh.setText(_translate("MainWindow", "Choose Image"))
        self.phanloai.setText(_translate("MainWindow", "Classification"))
        self.phanvung.setText(_translate("MainWindow", "Segmentation && Calculation"))
        self.xoa.setText(_translate("MainWindow", "Reset"))
        self.tenanh.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Image name:</span></p></body></html>"))
        self.hienthitenanh.setText(_translate("MainWindow", " "))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">HCM UNIVERSITY OF TECHNOLOGY AND EDUCATION</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">CLASSIFICATION AND CALCULATION OF BRAIN TUMOR AREA</span></p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Faculty of Electrical and Electronics Engineering</span></p></body></html>"))
        self.input_image= None # đặt biến ảnh
        self.id=None # đặt biến phân loại
    def select_image(self):
        self.reset()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self.centralwidget, "Choose Image", "",
                                                            "Images (*.jpg *.jpeg *.png *.bmp *.gif)",
                                                            options=options)
        if file_name:
            self.image_path = file_name
            self.input_image = cv2.imread(file_name)
            self.name= os.path.basename(self.image_path)
        
            self.display_image(self.input_image,self.anhchon)
            self.hienthitenanh.setText(self.name)
            self.id= None


    def process_image(self):
        if self.input_image is not None :# kiểm tra xem có ảnh hay không
            
            config_path = r"E:\DO AN TOT NGHIEP\22 - 23 FINAL PROJECTTTT\Implement\yolov4-custom.cfg"
            weights_path = r"E:\DO AN TOT NGHIEP\22 - 23 FINAL PROJECTTTT\Implement\yolov4-custom_7000.weights"
            classes_path = r"E:\DO AN TOT NGHIEP\22 - 23 FINAL PROJECTTTT\Implement\yolo.names"
            
            def get_output_layers(net):
                layer_names = net.getLayerNames()

                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                return output_layers

            def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
                label = classes[class_id]

                color = COLORS[class_id]

                cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

                cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            
            Width = self.input_image.shape[1]
            Height = self.input_image.shape[0]

            scale = 0.00392
            classes = None

            with open((classes_path), 'r') as f:
                classes = [line.strip() for line in f.readlines()]

            COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

            net = cv2.dnn.readNet(weights_path, config_path)

            blob = cv2.dnn.blobFromImage(self.input_image, scale, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)

            outs = net.forward(get_output_layers(net))

            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4

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
            if len(indices) > 0:
                for i in indices:
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2] 
                    h = box[3]
                    draw_prediction(self.input_image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
                # Chú ý vị trí cần phân loại có khối u hay không.
               
                self.id=class_ids[i]
               
                if class_ids[i]== 0 :
                   self.phanloai="pituitary"
                elif class_ids[i]== 1:
                   self.phanloai="meningioma" 
                elif class_ids[i]== 2:
                   self.phanloai="glioma"
                elif class_ids[i]== 3 :
                   self.phanloai="notumor" 
                      
                self.hienthikqphanloai.setText(self.phanloai)
              
                # hiển thị kết quả sau xử lí
                self.display_image(self.input_image, self.anhphathien)
                self.x=x
                self.y=y
                self.w=w
                self.h=h
            else: 
                self.phanloai="ERROR " 
                self.hienthikqphanloai.setText(self.phanloai)
        else: # nếu chưa chọn ảnh thì hiển thị lỗi
           self.anhchon.setText("Choose image again")
           self.anhphathien.setText("")
           self.anhkhoiu.setText("")
           self.anhkhoiu_2.setText("")
        
    def compute_area(self):
    # Xác định bounding box
        if self.id != 3 and self.id is not None :
        #     x, y, w, h = self.process_image()
            
            image = cv2.imread(self.image_path)
            org_image = cv2.imread(self.image_path)
            xmin = int(round(self.x))
            ymin = int(round(self.y))
            xmax = int(round(self.x+self.w))
            ymax = int(round(self.y+self.h))

            bbox = (xmin, ymin, xmax, ymax)
            bbox = tuple(map(int, bbox))
            rect = (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])

            # Tạo mask với kích thước bằng kích thước ảnh, giá trị 0 ban đầu
            mask = np.zeros(org_image.shape[:2], np.uint8)

            # Đặt giá trị 3 cho vùng bên trong bounding box
            mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 3

            # Tạo điểm "pinpoint" cho vùng đối tượng (tại chỗ giữa bounding box)
            cv2.circle(mask, (round(rect[2]/2+rect[0]), round(rect[3]/2+rect[1])), 10, cv2.GC_FGD, -1)

            # Đặt điểm "pinpoint" cho vùng nền (xung quanh vùng đối tượng)
            cv2.circle(mask, (280, 180), 7, cv2.GC_BGD, -1)

            # Thực hiện phân đoạn GrabCut-pinpoint
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(org_image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

            # Xác định vùng nền và vùng đối tượng
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img_masked = org_image * mask[:, :, np.newaxis]
            print(img_masked.shape)
            # plt.imshow(img_masked), plt.colorbar(), plt.show()
            self.display_image(img_masked, self.anhkhoiu)
            # Chuyển sang ảnh Gray
            img_grayscale = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

            # Thiết lập ngưỡng và chuyển sang ảnh Binary
           
            thresh, binary_img = cv2.threshold(img_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            w = img_masked.shape[0]
            h = img_masked.shape[1]

            # Hiển thị diện tích của vùng pixel màu trắng
            num_white = np.sum(binary_img == 255)
            
            #total = w*h
           
            # Tìm tất cả các đường viền có trong ảnh
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Vẽ đường biên cho từng đối tượng trong ảnh
            for contour in contours:
                cv2.drawContours(org_image, [contour], 0, (0, 255, 0), 3)
                cv2.fillPoly(org_image, [contour], (0, 255, 0))
        
            self.area_image = org_image
            # hiển thị kết quả sau xử lí
            self.display_image(self.area_image, self.anhkhoiu_2)
            self.area = num_white
            # # hiển thị diện tích
            self.hienthiS.setText(f"{self.area:.2f}  pixel")
        elif self.id == 3: 
           
            self.anhkhoiu.setText("Notumor")
            self.anhkhoiu_2.setText("Notumor")
            self.hienthiS.setText("0")
               
        elif self.id is None:
           self.anhphathien.setText("Not classified yet")
           self.anhkhoiu.setText("")
           self.anhkhoiu_2.setText("")

    def display_image(self, image, label):
        if image is not None:
            # chuyển đổi ảnh OpenCV thành QPixmap để hiển thị trên QLabel
            image = cv2.resize(image, (391, 271))
            height, width, channel = image.shape
            bytes_per_line = channel * width
            q_image = QPixmap.fromImage(QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888))
            label.setPixmap(q_image.scaled(label.width(), label.height(), Qt.KeepAspectRatio))
        else:
            # hiển thị một hình ảnh trống nếu không có ảnh nào được cung cấp
            label.clear() 
    

    def reset(self):
        self.area = None
        self.kqphanloai= None
        self.name= None
        
        self.anhchon.setText("Images select")
        self.anhkhoiu.setText("Images segmentation")
        self.anhphathien.setText("Images classification")
        self.anhkhoiu_2.setText("Images Tumor")

        self.hienthiS.setText(self.area)
        self.hienthikqphanloai.setText(self.kqphanloai)
        self.hienthitenanh.setText(self.name)
     





if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
