
import cv2
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog
from metrics import dice_loss, dice_coef

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
        self.label_5.setPixmap(QtGui.QPixmap('E:/DO_AN_TOT_NGHIEP/TEST_2label/TOTAL/Logo_truong.jpg'))
        self.label_5.setScaledContents(True)
        self.label_5.setObjectName("label_5")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1230, 10, 111, 101))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap('E:/DO_AN_TOT_NGHIEP/TEST_2label/TOTAL/Logo_nganh.png'))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(1100, 10, 121, 101))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap('E:/DO_AN_TOT_NGHIEP/TEST_2label/TOTAL/Logo_khoa.png'))
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
        self.anhkhoiu.setText(_translate("MainWindow", "Image Segmentation "))
        self.anhphathien.setText(_translate("MainWindow", "Classified Image "))
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
        self.id=None
        self.predictions= None # đặt biến phân loại
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
            
    def process_image(self):
        if self.input_image is not None :# kiểm tra xem có ảnh hay không
            model = tf.keras.models.load_model('E:/DO_AN_TOT_NGHIEP/TEST_2label/TOTAL/weights.4class_120.h5')
            self.input_image = cv2.resize(self.input_image, (240, 240))
            self.output_image=self.input_image
           # Chuẩn hóa giá trị pixel của tensor
            img_tensor = np.expand_dims(self.input_image, axis=0)
            # Chuẩn hóa giá trị pixel của tensor
            img_tensor = img_tensor.astype('float32') / 255.0
            # Dự đoán xác suất của ảnh thuộc từng lớp
            predictions = model.predict(img_tensor)
            # Lấy lớp có xác suất cao nhất
            predicted_class = np.argmax(predictions)
            self.predictions= predicted_class  
            if len( predictions) > 0:
                if predicted_class== 0 :
                        self.phanloai="Glioma"
                elif predicted_class== 1:
                        self.phanloai="Meningioma" 
                elif predicted_class== 2:
                        self.phanloai="Notumor"
                elif predicted_class== 3 :
                        self.phanloai="Pituitary" 
                       
                self.hienthikqphanloai.setText(self.phanloai)
                self.display_image(self.output_image, self.anhphathien)
           
            else: 
                self.phanloai="ERROR " 
                self.hienthikqphanloai.setText(self.phanloai)
        else: # nếu chưa chọn ảnh thì hiển thị lỗi
            self.anhchon.setText("Choose image again")
            self.anhphathien.setText("")
            self.anhkhoiu.setText("")
            self.anhkhoiu_2.setText("")
    
    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_results(image, y_pred, save_image_path):
        y_pred = np.expand_dims(y_pred, axis=-1)
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
        y_pred = y_pred * 255
        cv2.imwrite(save_image_path, y_pred)

    def compute_area(self):
    # Xác định bounding box
        if self.predictions != 2 and self.predictions is not None :
            model_path = 'E:/DO_AN_TOT_NGHIEP/TEST_2label/TOTAL/model_v3.h5'
            model = load_model(model_path, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})
            image = cv2.resize(self.output_image, (256, 256))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            """ Perform prediction """
            mask = model.predict(image)[0]
            mask = np.squeeze(mask, axis=-1)

            # Load the original image
            original_image = self.output_image
            # Resize the predicted mask to match the original image size
            resized_mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
            # Create a copy of the original image to draw the mask on
            output_image = original_image.copy()
            # Apply a green color mask on the output image
            output_image[resized_mask > 0.5] = (0, 255, 0)
            # Create a new image by combining the original image and the masked output image
            combined_image = np.hstack((original_image, output_image))

            def extract_green_mask(image):
                # Xác định màu xanh lá trong không gian màu RGB
                green = np.array([0, 255, 0], dtype=np.uint8)
                # Tạo mặt nạ cho pixel có màu xanh lá
                mask = np.all(image == green, axis=2)
                # Chuyển đổi mask thành ảnh nhị phân
                binary_mask = np.where(mask, 255, 0).astype(np.uint8)
                return binary_mask
      
            def count_green_pixels(image):
                # Xác định màu xanh lá trong không gian màu RGB
                green = np.array([0, 255, 0], dtype=np.uint8)
                # Tạo mặt nạ cho pixel có màu xanh lá
                mask = np.all(image == green, axis=2)
                # Đếm số pixel có màu xanh lá trong mặt nạ
                count = np.sum(mask)
                return count  
   
            green_mask = extract_green_mask(output_image)
            
            self.display_image(green_mask, self.anhkhoiu)
            self.display_image(output_image, self.anhkhoiu_2)
            
            # # hiển thị diện tích
            self.area = count_green_pixels(output_image)
            self.hienthiS.setText(f"{self.area:.2f}  pixel")

        elif self.predictions == 2: 
            
            self.anhkhoiu.setText("Notumor")
            self.anhkhoiu_2.setText("Notumor")
            self.hienthiS.setText("0")
                
        elif self.predictions is None :
            self.anhphathien.setText("Not classified yet")
            self.anhkhoiu.setText("")
            self.anhkhoiu_2.setText("")

    def display_image(self, image, label):
        q_image = None

        if image is not None:
                if len(image.shape) == 3 and image.shape[2] == 3:
                        # Ảnh 3 kênh (BGR)
                        image = cv2.resize(image, (391, 271))
                        height, width, channel = image.shape
                        bytes_per_line = channel * width
                        q_image = QPixmap.fromImage(QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888))
                elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                        # Ảnh xám (1 kênh) hoặc ảnh 2 kênh (grayscale)
                        image = cv2.resize(image, (391, 271))
                        height, width = image.shape
                        bytes_per_line = width
                        q_image = QPixmap.fromImage(QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8))

        if q_image is not None:
                label.setPixmap(q_image.scaled(label.width(), label.height(), Qt.KeepAspectRatio))
        else:
                label.clear()



    def reset(self):
        self.area = None
        self.kqphanloai= None
        self.name= None
        
        self.input_image= None
        self.output_image= None
        self.id=None
        self.predictions= None

        self.anhchon.setText("Selected Image")
        self.anhkhoiu.setText("Segmentation Image")
        self.anhphathien.setText("Classification Image ")
        self.anhkhoiu_2.setText("Brain Tumor Image")

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
