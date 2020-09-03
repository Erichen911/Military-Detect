from PyQt5 import QtCore, QtGui, QtWidgets
import os
import keras
import pyqtgraph as pg
import sys
import windows2
import numpy as np
import test_frcnn
import train_frcnn
import train1
import train2
import predict1
import predict2
import evaluate
import measure_map
import json
import re
import testAll
import predict
import keras
import pickle
import ctypes
import inspect

num_roi=128
num_epoch=6000
model_save="./model_frcnn.hdf5"


class MainWindow(object):
    def __init__(self):
        keras.backend.clear_session()
        self.flag = True
        self.app = QtWidgets.QApplication(sys.argv)
        mainwindow = exp()
        
        self.ui = windows2.Ui_MainWindow()
        self.ui.setupUi(mainwindow)
        #*****************设置样式表*******************#
        f = open("style.qss", "r",encoding='utf-8')
        mainwindow.setStyleSheet(f.read())
        f.close()
        #**********************************************#

        mainwindow.setWindowTitle("Military_Detecting_by_Erichen")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("aimed.jpg"),QtGui.QIcon.Normal,QtGui.QIcon.Off)
        mainwindow.setWindowIcon(icon)
        mainwindow.setFixedSize(mainwindow.width(), mainwindow.height())

        # *****************设置训练表格****************#
        win = pg.PlotWidget(title="curr_loss")
        cure1 = win.plot(pen=(255,0,0))
        win2 = pg.PlotWidget(title="loss_rpn_cls")
        cure2 = win2.plot(pen=(255, 0, 255))
        win3 = pg.PlotWidget(title="loss_rpn_regr")
        cure3 = win3.plot(pen=(255, 0, 255))
        win4 = pg.PlotWidget(title="loss_class_cls")
        cure4 = win4.plot(pen=(255, 0, 255))
        win5 = pg.PlotWidget(title="loss_class_regr")
        cure5 = win5.plot(pen=(255, 0, 255))
        self.curelist = [cure1,cure2,cure3,cure4,cure5]
        self.ui.layoutforchart.addWidget(win5)
        self.ui.layoutforchart.addWidget(win2)
        self.ui.layoutforchart.addWidget(win3)
        self.ui.layoutforchart.addWidget(win4)
        self.ui.layoutforchart.addWidget(win)
        #**********************************************#

        # *****************设置标签2训练表格****************#
        m_win = pg.PlotWidget(title="curr_loss")
        self.m_cure1 = m_win.plot(pen=(255,0,0))
        self.ui.yolomilitary_chart_1.addWidget(m_win)
        #**********************************************#

        # *****************设置标签3训练表格****************#
        c_win = pg.PlotWidget(title="curr_loss")
        self.c_cure1 = c_win.plot(pen=(255,0,0))
        self.ui.yolocity_chart_2.addWidget(c_win)
        #**********************************************#

        # *****************设置标签1评估表格****************#
        b_win_pg1 = pg.PlotWidget(title="airbase")
        b_cure1_pg1 = b_win_pg1.plot(pen=(255,0,0))
        b_win_pg2 = pg.PlotWidget(title="habour")
        b_cure1_pg2 = b_win_pg2.plot(pen=(255,0,0))
        b_win_pg3 = pg.PlotWidget(title="island")
        b_cure1_pg3 = b_win_pg3.plot(pen=(255,0,0))
        self.curelist_big = [b_cure1_pg1,b_cure1_pg2,b_cure1_pg3]
        self.ui.fastercnn_chart_pingu.addWidget(b_win_pg1)
        self.ui.fastercnn_chart_pingu.addWidget(b_win_pg2)
        self.ui.fastercnn_chart_pingu.addWidget(b_win_pg3)
        #**********************************************#

        # *****************设置标签2评估表格****************#
        m_win_pg1 = pg.PlotWidget(title="missile")
        m_cure1_pg1 = m_win_pg1.plot(pen=(255,0,0))
        m_win_pg2 = pg.PlotWidget(title="warship")
        m_cure1_pg2 = m_win_pg2.plot(pen=(255,0,0))
        m_win_pg3 = pg.PlotWidget(title="plane")
        m_cure1_pg3 = m_win_pg3.plot(pen=(255,0,0))
        m_win_pg4 = pg.PlotWidget(title="oiltank")
        m_cure1_pg4 = m_win_pg4.plot(pen=(255,0,0))
        self.curelist_military = [m_cure1_pg1,m_cure1_pg2,m_cure1_pg3,m_cure1_pg4]
        self.ui.yolomilitary_chart_pingu_2.addWidget(m_win_pg1)
        self.ui.yolomilitary_chart_pingu_2.addWidget(m_win_pg2)
        self.ui.yolomilitary_chart_pingu_2.addWidget(m_win_pg3)
        self.ui.yolomilitary_chart_pingu_2.addWidget(m_win_pg4)
        #**********************************************#

        # *****************设置标签3评估表格****************#
        c_win_pg1 = pg.PlotWidget(title="bridge")
        c_cure1_pg1 = c_win_pg1.plot(pen=(255,0,0))
        c_win_pg2 = pg.PlotWidget(title="oilplatform")
        c_cure1_pg2 = c_win_pg2.plot(pen=(255,0,0))
        c_win_pg3 = pg.PlotWidget(title="train")
        c_cure1_pg3 = c_win_pg3.plot(pen=(255,0,0))
        self.curelist_city = [c_cure1_pg1,c_cure1_pg2,c_cure1_pg3]
        self.ui.yolocity_chart_pingu_2.addWidget(c_win_pg1)
        self.ui.yolocity_chart_pingu_2.addWidget(c_win_pg2)
        self.ui.yolocity_chart_pingu_2.addWidget(c_win_pg3)
        #**********************************************#

        mainwindow.show()

        self.ui.biaojiButton.triggered.connect(biaoji)
        self.ui.caijianButton.triggered.connect(caijian)
        self.ui.trainPushButton.clicked.connect(self.to_train)
        self.ui.test_begin_Button.clicked.connect(self.to_test)
        self.ui.trainstopPushButton.clicked.connect(self.stop_train)


        #New ,After I lost my Mate20
        self.ui.yolomilitary_train_button_1.clicked.connect(self.to_train_yolo_military)
        self.ui.yolocity_train_button_2.clicked.connect(self.to_train_yolo_city)
        self.ui.yolomilitary_stop_button_1.clicked.connect(self.stop_military_train)
        self.ui.yolocity_stop_button_2.clicked.connect(self.stop_city_train)
        self.ui.test_begin_Button_1.clicked.connect(self.to_test_yolo_military)
        self.ui.test_begin_Button_2.clicked.connect(self.to_test_yolo_city)
        self.ui.fastercnn_pingu_button.clicked.connect(self.to_evaluate_big)
        self.ui.yolomilitary_pingu_button_1.clicked.connect(self.to_evaluate_yolo_military)
        self.ui.yolocity_pingu_button_2.clicked.connect(self.to_evaluate_yolo_city)
        self.ui.show_1_begin.clicked.connect(self.show1)
        self.ui.show_2_begin.clicked.connect(self.show2)

        #New ,After I buy a new Mate20
        self.ui.test_PushButton.clicked.connect(lambda:self.gettestdic(self.ui.testpathEdit))
        self.ui.test_PushButton_2.clicked.connect(lambda: self.gettestdic(self.ui.testpathEditout))
        self.ui.image_folder_button_1.clicked.connect(lambda: self.gettestdic(self.ui.image_folder_text_1))
        self.ui.annot_folder_button_1.clicked.connect(lambda: self.gettestdic(self.ui.annot_folder_text_1))
        self.ui.image_folder_button_2.clicked.connect(lambda: self.gettestdic(self.ui.image_folder_text_2))
        self.ui.annot_folder_button_2.clicked.connect(lambda: self.gettestdic(self.ui.annot_folder_text_2))
        self.ui.input_PushButton_2.clicked.connect(lambda: self.gettestdic(self.ui.military_test_input_1))
        self.ui.output_PushButton_2.clicked.connect(lambda: self.gettestdic(self.ui.military_test_output_1))
        self.ui.input_PushButton_3.clicked.connect(lambda: self.gettestdic(self.ui.city_test_input_2))
        self.ui.output_PushButton_3.clicked.connect(lambda: self.gettestdic(self.ui.city_test_output_2))
       # self.ui.fastercnn_pingu_path_button.clicked.connect(lambda: self.gettestdic(self.ui.pingupathEdit))
        self.ui.show_1_inputbutton.clicked.connect(lambda :self.gettestdic(self.ui.show_1_input))
        self.ui.show_1_outputbutton.clicked.connect(lambda: self.gettestdic(self.ui.show_1_output))
        self.ui.show_2_inputbutton.clicked.connect(lambda: self.gettestdic(self.ui.show_2_input))
        self.ui.show_2_outputbutton.clicked.connect(lambda: self.gettestdic(self.ui.show_2_output))
        self.ui.modelPushButton.clicked.connect(lambda :self.getfilename(self.ui.modelpathEdit))
        self.ui.model_PushButton_2.clicked.connect(lambda :self.getfilename(self.ui.military_test_model_1))
        self.ui.model_PushButton_3.clicked.connect(lambda: self.getfilename(self.ui.city_test_model_2))
        self.ui.show_1_modelbutton.clicked.connect(lambda: self.getfilename(self.ui.show_1_model))
        self.ui.show_2_modelbutton.clicked.connect(lambda: self.getfilename(self.ui.show_2_model))
		

        self.ui.show_1_button1.clicked.connect(lambda: self.show_pic(self.ui.show_1_input.text(),True,0))
        self.ui.show_1_button4.clicked.connect(lambda: self.show_pic(self.ui.show_1_output.text(),True,0))
        self.ui.show_1_button2.clicked.connect(lambda: self.show_pic(self.ui.show_1_output.text(),False,-1))
        self.ui.show_1_button3.clicked.connect(lambda: self.show_pic(self.ui.show_1_output.text(),False,+1))
        
        self.ui.show_2_button1.clicked.connect(lambda: self.show_pic2(self.ui.show_2_input.text(),True,0))
        self.ui.show_2_button4.clicked.connect(lambda: self.show_pic2(self.ui.show_2_output.text(),True,0))
        self.ui.show_2_button2.clicked.connect(lambda: self.show_pic2(self.ui.show_2_output.text(),False,-1))
        self.ui.show_2_button3.clicked.connect(lambda: self.show_pic2(self.ui.show_2_output.text(),False,+1))
        self.piclist=[]
        self.num = 0
        
        #sys.exit(app.exec())
        self.app.exec()

    def show_pic(self,path,flag,num):
        if flag:
            self.num = 0
            self.piclist.clear()
            for root, dirs, files in os.walk(path):
                for name in files:
                    self.piclist.append(os.path.join(root, name))
                    #print(os.path.join(root, name))
            if self.piclist:
                image = QtGui.QPixmap(self.piclist[0])
                self.ui.show_1_pic.setPixmap(image)
                self.ui.show_1_pic.setScaledContents(True)
        else:
            if self.piclist:
                print(self.num)
                self.num = self.num+num
                print(self.num)
                if self.num < 0 :
                    self.num = 0
                if self.num >= len(self.piclist):
                    self.num = len(self.piclist) - 1
                image = QtGui.QPixmap(self.piclist[self.num])
                self.ui.show_1_pic.setPixmap(image)
                self.ui.show_1_pic.setScaledContents(True)
                
    def show_pic2(self,path,flag,num):
        if flag:
            self.num = 0
            self.piclist.clear()
            for root, dirs, files in os.walk(path):
                for name in files:
                    self.piclist.append(os.path.join(root, name))
                    #print(os.path.join(root, name))
            if self.piclist:
                image = QtGui.QPixmap(self.piclist[0])
                self.ui.show_2_pic.setPixmap(image)
                self.ui.show_2_pic.setScaledContents(True)
        else:
            if self.piclist:
                print(self.num)
                self.num = self.num+num
                print(self.num)
                if self.num < 0 :
                    self.num = 0
                if self.num >= len(self.piclist):
                    self.num = len(self.piclist) - 1
                image = QtGui.QPixmap(self.piclist[self.num])
                self.ui.show_2_pic.setPixmap(image)
                self.ui.show_2_pic.setScaledContents(True)

    def show1(self):
        with open("./config.pickle", 'rb') as f_in:
            C = pickle.load(f_in)
        C.model_path = self.ui.show_1_model.text()

        #if os.path.exists("./config.pickle"):
        #    os.remove("./config.pickle")

        with open("./config.pickle", 'wb') as config_f:
           pickle.dump(C, config_f)
        print(C.model_path)
        self.show1thread = Show1_Thread(self.ui.show_1_textedit,self.ui.show_1_pic,self.ui.show_1_input.text(),self.ui.show_1_model.text(),self.ui.show_1_output.text())
        self.show1thread.start()
    def show2(self):
        file_in = open("./config1.json", "r")
        json_data = json.load(file_in)
        file_in.close()
    
        json_data["train"]["saved_weights_name"] = self.ui.show_2_model.text()
    
        file_out = open("./config1.json", "w")
        file_out.write(json.dumps(json_data, indent=4, separators=(",", ":")))
        file_out.close()
        self.show2thread = Show2_Thread(self.ui.show_2_textedit,self.ui.show_2_pic,self.ui.show_2_input.text(),self.ui.show_2_model.text(),self.ui.show_2_output.text())
        self.show2thread.start()
    def gettestdic(self,textedit):
        self.dicname = QtWidgets.QFileDialog.getExistingDirectory()
        if(self.dicname!=""):
            textedit.setText(self.dicname+'/')
            
    def getfilename(self,textedit):
        self.filename = QtWidgets.QFileDialog.getOpenFileName()
        print(self.filename)
        if(self.filename!=""):
            textedit.setText(self.filename[0])

    def to_train(self):
        keras.backend.clear_session()
        traval_path = self.ui.traval_path.text()
        weights = self.ui.weights.text()
        dataAug_hf = self.ui.hf_combox.currentText()
        dataAug_vf = self.ui.vf_combox.currentText()
        dataAug_rot = self.ui.rot_combox.currentText()
        base_network = "resnet50"
        num_ep = int(self.ui.weights.text())
        self.thread = My_Thread(num_ep,self.ui.textEdit,traval_path,weights,dataAug_hf,dataAug_vf,dataAug_rot,base_network,self.curelist)
        self.thread.start()
        #train_frcnn.work(self.ui.textEdit)

    def to_test(self):
        #if self.ui.testpathEdit.text():
         #   shutil.rmtree("./testImages")
          #  os.mkdir("./testImages")
           # thedatalist = os.listdir(self.dicname)
            #for i in thedatalist:
             #   oldname = self.dicname+"/"+i
              #  newname = "./testImages"+"/"+i
               # shutil.copyfile(oldname,newname)
        with open("./config.pickle", 'rb') as f_in:
            C = pickle.load(f_in)
        C.model_path = self.ui.modelpathEdit.text()

        #if os.path.exists("./config.pickle"):
        #    os.remove("./config.pickle")

        with open("./config.pickle", 'wb') as config_f:
           pickle.dump(C, config_f)
        print(C.model_path)
        self.testthread = Test_Thread(self.ui.testpathEdit.text(),self.ui.testpathEditout.text(),self.ui.test_text,self.ui.img_detect,self.ui.img_prim)
        self.testthread.start()

    def stop_train(self):
        #dialog = QtWidgets.
        ex = exp()
        self.flag = False
        self.app.exit()
    def stop_military_train(self):
        ex = exp()
        self.flag = False
        self.app.exit()
        
    def stop_city_train(self):
        ex = exp()
        self.flag = False
        self.app.exit()
        
    #训练4类军事目标
    def to_train_yolo_military(self):
        #读取json文件默认数据
        keras.backend.clear_session()
        file_in = open("./config1.json","r")
        json_data = json.load(file_in)
        file_in.close()

        #更改数据
        json_data["model"]["min_input_size"] = int(self.ui.min_input_size_1.text())
        json_data["model"]["max_input_size"] = int(self.ui.max_input_size_1.text())
        json_data["train"]["train_times"] = int(self.ui.train_times_1.text())
        json_data["train"]["batch_size"] = int(self.ui.batch_size_1.text())
        json_data["train"]["gpus"] = self.ui.gpus_1.text()
        json_data["train"]["learning_rate"] = float(self.ui.learning_rate_1.text())
        json_data["train"]["warmup_epochs"] = int(self.ui.warmup_epochs_1.text())
        json_data["train"]["nb_epochs"] = int(self.ui.nb_epochs_1.text())
        json_data["train"]["ignore_thresh"] = float(self.ui.ignore_thresh_1.text())
        json_data["train"]["train_image_folder"] = self.ui.image_folder_text_1.text()
        json_data["train"]["train_annot_folder"] = self.ui.annot_folder_text_1.text()
        json_data["model"]["labels"] = re.sub('"',"",self.ui.labels_text_1.text().replace(" ","")).split(",")

        #更改json配置文件
        file_out = open("./config1.json", "w")
        file_out.write(json.dumps(json_data,indent=4,separators=(",",":")))
        file_out.close()

        #开始训练
        self.mthread = Yolo_Military_Train_Thread(self.ui.yolomilitary_train_text_1,self.m_cure1)
        self.mthread.start()
        

    #训练3类城市目标
    def to_train_yolo_city(self):
        #读取json文件默认数据
        keras.backend.clear_session()
        file_in = open("./config2.json","r")
        json_data = json.load(file_in)
        file_in.close()

        #更改数据
        json_data["model"]["min_input_size"] = int(self.ui.min_input_size_2.text())
        json_data["model"]["max_input_size"] = int(self.ui.max_input_size_2.text())
        json_data["train"]["train_times"] = int(self.ui.train_times_2.text())
        json_data["train"]["batch_size"] = int(self.ui.batch_size_2.text())
        json_data["train"]["gpus"] = self.ui.gpus_2.text()
        json_data["train"]["learning_rate"] = float(self.ui.learning_rate_2.text())
        json_data["train"]["warmup_epochs"] = int(self.ui.warmup_epochs_2.text())
        json_data["train"]["nb_epochs"] = int(self.ui.nb_epochs_2.text())
        json_data["train"]["ignore_thresh"] = float(self.ui.ignore_thresh_2.text())
        json_data["train"]["train_image_folder"] = self.ui.image_folder_text_2.text()
        json_data["train"]["train_annot_folder"] = self.ui.annot_folder_text_2.text()
        json_data["model"]["labels"] = re.sub('"',"",self.ui.labels_text_2.text().replace(" ","")).split(",")

        #更改json配置文件
        file_out = open("./config2.json", "w")
        file_out.write(json.dumps(json_data,indent=4,separators=(",",":")))
        file_out.close()

        #开始训练
        self.cthread = Yolo_City_Train_Thread(self.ui.yolocity_train_text_2,self.c_cure1)
        self.cthread.start()

    def to_test_yolo_military(self):
        file_in = open("./config1.json","r")
        json_data = json.load(file_in)
        file_in.close()

        json_data["train"]["saved_weights_name"] = self.ui.military_test_model_1.text()

        file_out = open("./config1.json", "w")
        file_out.write(json.dumps(json_data, indent=4, separators=(",", ":")))
        file_out.close()
        
        self.testthread1 = Yolo_Military_Test_Thread(self.ui.military_test_input_1.text(),self.ui.military_test_output_1.text(),self.ui.test_text_2,self.ui.img_prim_2,self.ui.img_detect_2)
        self.testthread1.start()

    def to_test_yolo_city(self):
        file_in = open("./config2.json","r")
        json_data = json.load(file_in)
        file_in.close()

        json_data["train"]["saved_weights_name"] = self.ui.city_test_model_2.text()

        file_out = open("./config2.json", "w")
        file_out.write(json.dumps(json_data, indent=4, separators=(",", ":")))
        file_out.close()
        self.testthread2 = Yolo_City_Test_Thread(self.ui.city_test_input_2.text(), self.ui.city_test_output_2.text(),self.ui.test_text_3,self.ui.img_prim_3,self.ui.img_detect_3)
        self.testthread2.start()

    def to_evaluate_yolo_military(self):
        self.evaluatethread1 = Yolo_Military_Evaluate_Thread(self.curelist_military,self.ui.yolomilitary_pingu_text_1)
        self.evaluatethread1.start()

    def to_evaluate_yolo_city(self):
        self.evaluatethread2 = Yolo_City_Evaluate_Thread(self.curelist_city,self.ui.yolocity_pingu_text_2)
        self.evaluatethread2.start()

    def to_evaluate_big(self):
        self.evaluatethread2 = Fastercnn_Big_Evaluate_Thread("../",self.curelist_big,self.ui.frcnn_pinggu_text)
        self.evaluatethread2.start()

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, 'Message', 'You sure to quit?',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def biaoji(self):
    os.system("label_boxed.exe")

def caijian(self):
    os.system("..\Cutexe\cutsystem_boxed.exe")

class My_Thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def  __init__(self,num_ep, textEdit,traval_path,weights,dataAug_hf,dataAug_vf,dataAug_rot,base_network,curvelist,parent=None):
        super(My_Thread, self).__init__()
        self.num_ep = num_ep
        self.edit=textEdit
        self.traval_path=traval_path
        self.weights=weights
        self.dataAug_hf=dataAug_hf
        self.dataAug_vf=dataAug_vf
        self.dataAug_rot=dataAug_rot
        self.base_network=base_network
        self.curvelist=curvelist
    def __del__(self):
        self.wait()
    def run(self):
        train_frcnn.work(self.num_ep,self.edit,self.traval_path,'weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',self.dataAug_hf,self.dataAug_vf,self.dataAug_rot,self.base_network,self.curvelist)
        self.trigger.emit()
    def callback(self, msg):
        pass

class Test_Thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def  __init__(self, input, output, textEdit,pic_label,primpic_label,parent=None):
        super(Test_Thread, self).__init__()
        self.edit = textEdit
        self.label = pic_label
        self.prim = primpic_label
        self.input = input
        self.output = output
        keras.backend.clear_session()
    def __del__(self):
        self.wait()
    def run(self):
        test_frcnn.work(self.input, self.output,self.edit,self.label,self.prim)
        self.trigger.emit()
    def callback(self, msg):
        pass

class Yolo_Military_Train_Thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def __init__(self,textedit,cure,parent=None):
        super(Yolo_Military_Train_Thread, self).__init__()
        self.textedit = textedit
        self.cure = cure
    def __del__(self):
        self.wait()
    def run(self):
        train1.work(self.textedit,self.cure)
        self.trigger.emit()
    def callback(self, msg):
        pass

class Yolo_City_Train_Thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def __init__(self,textedit,cure,parent=None):
        super(Yolo_City_Train_Thread, self).__init__()
        self.textedit = textedit
        self.cure = cure
    def __del__(self):
        self.wait()
    def run(self):
        train2.work(self.textedit,self.cure)
        self.trigger.emit()
    def callback(self, msg):
        pass

class Yolo_Military_Test_Thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def  __init__(self, input,output,textedit,label1,label2,parent=None):
        super(Yolo_Military_Test_Thread, self).__init__()
        self.input = input
        self.output = output
        self.textedit = textedit
        self.label1 = label1
        self.label2 = label2
        keras.backend.clear_session()
    def __del__(self):
        self.wait()
    def run(self):
        predict1.work(self.input,self.output,self.textedit,self.label1,self.label2)
        self.trigger.emit()
    def callback(self, msg):
        pass

class Yolo_City_Test_Thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def  __init__(self, input,output,textedit,label1,label2,parent=None):
        super(Yolo_City_Test_Thread, self).__init__()
        self.input = input
        self.output = output
        self.textedit = textedit
        self.label1 = label1
        self.label2 = label2
        keras.backend.clear_session()
    def __del__(self):
        self.wait()
    def run(self):
        predict2.work(self.input,self.output,self.textedit,self.label1,self.label2)
        self.trigger.emit()
    def callback(self, msg):
        pass

class Yolo_Military_Evaluate_Thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def  __init__(self,curelist,textedit,parent=None):
        super(Yolo_Military_Evaluate_Thread, self).__init__()
        self.curelist = curelist
        self.textedit = textedit
        keras.backend.clear_session()
    def __del__(self):
        self.wait()
    def run(self):
        evaluate.work(self.curelist,self.textedit,"config1.json")
        self.trigger.emit()
    def callback(self, msg):
        pass

class Yolo_City_Evaluate_Thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def __init__(self, curelist, textedit, parent=None):
        super(Yolo_City_Evaluate_Thread, self).__init__()
        self.curelist = curelist
        self.textedit = textedit
        keras.backend.clear_session()
    def __del__(self):
        self.wait()
    def run(self):
        evaluate.work(self.curelist, self.textedit, "config2.json")
        self.trigger.emit()
    def callback(self, msg):
        pass

class Fastercnn_Big_Evaluate_Thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def __init__(self, path,curelist, textedit, parent=None):
        super(Fastercnn_Big_Evaluate_Thread, self).__init__()
        self.path = path
        self.curelist = curelist
        self.textedit = textedit
        keras.backend.clear_session()
    def __del__(self):
        self.wait()
    def run(self):
        measure_map.work(self.path, self.curelist, self.textedit)
        self.trigger.emit()
    def callback(self, msg):
        pass

class Show1_Thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def  __init__(self, textEdit,pic_label,input,model,output,parent=None):
        super(Show1_Thread, self).__init__()
        self.edit = textEdit
        self.label = pic_label
        self.input = input
        self.model = model
        self.output = output
    def __del__(self):
        self.wait()
    def run(self):
        '''
        image = QtGui.QPixmap("../show1/testAll_15.jpg")
        self.label.setPixmap(image)
        self.label.setScaledContents(True)
        '''
        testAll.work(self.edit,self.label,self.input,self.model,self.output)
        self.trigger.emit()
    def callback(self, msg):
        pass

class Show2_Thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def  __init__(self,textEdit,pic_label,input,model,output,parent=None):
        super(Show2_Thread, self).__init__()
        self.edit = textEdit
        self.label = pic_label
        self.input = input
        self.model = model
        self.output = output
    def __del__(self):
        self.wait()
    def run(self):
        print("I'm hearing")
        predict.work(self.edit,self.label,self.input,self.model,self.output)
        self.trigger.emit()
    def callback(self, msg):
        pass

class reopen(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def  __init__(self,parent=None):
        super(reopen, self).__init__()
    def __del__(self):
        self.wait()
    def run(self):
        os.system("python G:/tju_object_detection/detect-source/military_detect/ob_main.py")
        self.trigger.emit()
    def callback(self, msg):
        pass

class exp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, 'Message', 'You sure to quit?',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    win = MainWindow()
    print("BackDoor")
    if win.flag:
        pass
    else:
        print(sys.argv)
        python = sys.executable
        os.execl(python, python, *sys.argv)