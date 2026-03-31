import sys
import math
import pickle
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QFont, QPixmap
from gui import Ui_MainWindow

# 尝试导入所需的机器学习库
try:
    import lightgbm as lgb
except ImportError:
    print("警告：未安装 lightgbm，无法加载 f_src_model.pkl")
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    print("警告：未安装 xgboost，无法加载 g_target_model.pkl")
    xgb = None


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # ========== 添加右上角 Logo ==========
        self.logo_label = QtWidgets.QLabel(self.centralwidget)
        try:
            pixmap = QPixmap("中南大学.png")
            scaled_pixmap = pixmap.scaled(150, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_pixmap)
            self.logo_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
            # 初始位置
            self.logo_label.move(self.width() - scaled_pixmap.width() - 10, 10)
        except Exception as e:
            print(f"Logo 加载失败: {e}")
            self.logo_label.hide()
        self.resizeEvent = self.on_resize

        # ========== 加载 LightGBM 模型 ==========
        self.model_f = None
        try:
            with open('f_src.pkl', 'rb') as f:
                self.model_f = pickle.load(f)
            print("LightGBM 模型加载成功")
        except Exception as e:
            print(f"LightGBM 模型加载失败: {e}")
            self.model_f = None

        # ========== 加载 XGBoost 模型 ==========
        self.model_g = None
        try:
            with open('g_target.pkl', 'rb') as f:
                self.model_g = pickle.load(f)
            print("XGBoost 模型加载成功")
        except Exception as e:
            print(f"XGBoost 模型加载失败: {e}")
            self.model_g = None

        # ========== 加载 Z-score 归一化参数 ==========
        self.means = None
        self.stds = None
        try:
            means_list = []
            with open('平均值.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        means_list.append(float(line))
            stds_list = []
            with open('方差.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        std = float(line)
                        if std == 0:
                            std = 1.0
                        stds_list.append(std)

            if len(means_list) != 20 or len(stds_list) != 20:
                raise ValueError(f"参数数量错误：均值{len(means_list)}，标准差{len(stds_list)}，应均为20")

            self.means = np.array(means_list)
            self.stds = np.array(stds_list)
            print("归一化参数加载成功")
        except Exception as e:
            print(f"归一化参数加载失败: {e}")
            self.means = None
            self.stds = None
            QtWidgets.QMessageBox.critical(self, "参数加载失败",
                                           "无法加载归一化参数文件，机器学习预测功能将不可用。")

        # 连接按钮信号
        self.pushButton04.clicked.connect(self.on_predict_clicked)
        self.pushButton03.clicked.connect(self.on_gb50936_clicked)
        self.pushButton01.clicked.connect(self.load_data_from_file)  # 新增
        self.pushButton02.clicked.connect(self.clear_all_inputs)

    def on_resize(self, event):
        """窗口大小改变时重新定位 logo"""
        if hasattr(self, 'logo_label') and self.logo_label.pixmap():
            pixmap = self.logo_label.pixmap()
            self.logo_label.move(self.width() - pixmap.width() - 10, 10)
        super().resizeEvent(event)

    def on_predict_clicked(self):
        """机器学习预测（修复版：保证与Jupyter完全一致）"""
        if self.model_f is None or self.model_g is None:
            QtWidgets.QMessageBox.warning(self, "错误", "模型未完全加载")
            return

        if self.means is None or self.stds is None:
            QtWidgets.QMessageBox.warning(self, "错误", "归一化参数未加载")
            return

        # ==============================
        # 1. 读取20个特征（严格数值化）
        # ==============================
        raw_features = []

        for i in range(20):
            line_edit = getattr(self, f'lineEdit{i:02d}')
            text = line_edit.text().strip()

            if text == "":
                QtWidgets.QMessageBox.warning(self, "输入错误", f"特征{i + 1}为空")
                return

            try:
                value = float(text)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "输入错误", f"特征{i + 1}不是数字")
                return

            raw_features.append(value)

        # ==============================
        # 2. 转 numpy
        # ==============================
        raw_features = np.array(raw_features, dtype=float)

        # ==============================
        # 3. Z-score标准化
        # ==============================
        std_safe = np.where(self.stds == 0, 1, self.stds)
        norm_features = (raw_features - self.means) / std_safe
        norm_features = np.nan_to_num(norm_features)

        # ==============================
        # 4. 模型输入
        # ==============================
        X_f = norm_features[:14].reshape(1, -1)
        X_g = norm_features[:20].reshape(1, -1)

        # ==============================
        # 5. 预测
        # ==============================
        try:
            pred_f = self.model_f.predict(X_f)[0]
            pred_g = self.model_g.predict(X_g)[0]

            total_pred = pred_f + pred_g

            self.lineEdit20.setText(f"{total_pred:.2f}")

            print("=== DEBUG ===")
            print("norm_features:", norm_features)
            print("pred_f:", pred_f)
            print("pred_g:", pred_g)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "预测失败", str(e))

        # 注：已删除 update_chart() 调用

    def on_gb50936_clicked(self):
        """基于 GB 50936-2014 和 EC4 分别计算轴压承载力"""

        # ========== GB 50936-2014 计算函数 ==========
        def f_sc_composite(f_c, f, A_c, A_s):
            """组合轴压强度设计值 f_sc (MPa)，系数B、C基于混凝土强度计算"""
            alpha_sc = A_s / A_c
            theta = (alpha_sc * f) / f_c
            B = 0.131 * f_c / 213 + 0.723
            C = -0.070 * f_c / 14.4 + 0.026
            f_sc = (1.212 + B * theta + C * theta ** 2) * f_c
            return f_sc

        def stability_factor(L, i_x, f_yk):
            """稳定系数 φ，长细比固定取 0.5 * L / i_x"""
            lambda_sc = 0.5 * L / i_x
            lambda_bar_sc = lambda_sc * 0.01 * (0.001 * f_yk + 0.781)
            if lambda_bar_sc <= 1e-6:
                return 1.0
            a = lambda_bar_sc ** 2
            b = 1 + 0.25 * lambda_bar_sc
            c = a + b
            d = c ** 2 - 4 * a
            sqrt_term = math.sqrt(max(d, 0))
            phi = (c - sqrt_term) / (2 * a)
            return phi

        def axial_compressive_strength(f_c, f, A_c, A_s, L, i_x, f_yk):
            f_sc = f_sc_composite(f_c, f, A_c, A_s)
            A_sc = A_c + A_s
            phi = stability_factor(L, i_x, f_yk)
            N = phi * f_sc * A_sc
            return N

        # ========== EC4 计算函数 ==========
        def composite_column_plastic_resistance(As, fy, gamma_s, Ac, fc_prime, gamma_c):
            steel_contribution = As * fy / gamma_s
            concrete_contribution = Ac * fc_prime / gamma_c
            return steel_contribution + concrete_contribution

        # ========== 主计算流程：获取输入参数 ==========
        try:
            values = []
            for i in range(20):
                line_edit = getattr(self, f'lineEdit{i:02d}')
                text = line_edit.text().strip()
                if not text:
                    QtWidgets.QMessageBox.warning(self, "输入错误", f"请输入特征 {i + 1} 的值")
                    return
                try:
                    value = float(text)
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, "输入错误", f"特征 {i + 1} 必须是数字")
                    return
                values.append(value)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "读取错误", f"读取输入值时出错：{e}")
            return

        # 提取所需参数（索引基于用户定义）
        try:
            A_s = values[2]          # 钢材截面面积
            L = values[3]            # 组合柱长度
            I_sx = values[4]         # 钢材X轴惯性矩
            f = values[8]            # 钢材强度设计值
            A_c = values[14]         # 混凝土截面面积
            f_c = values[15]         # 混凝土强度设计值
            I_cx = values[18]        # 混凝土X轴惯性矩
        except IndexError:
            QtWidgets.QMessageBox.warning(self, "参数错误", "特征索引超出范围，请检查输入")
            return

        # 有效性检查
        if A_s <= 0 or A_c <= 0 or f_c <= 0 or f <= 0 or I_sx <= 0 or I_cx <= 0:
            QtWidgets.QMessageBox.warning(self, "参数错误", "所有相关特征必须为正数")
            return

        # ---------- GB 50936-2014 计算 ----------
        I_x = I_sx + I_cx
        i_x = math.sqrt(I_x / (A_s + A_c))
        f_yk = f  # 钢材屈服强度特征值近似
        N_gb = axial_compressive_strength(f_c, f, A_c, A_s, L, i_x, f_yk)
        self.lineEdit21.setText(f"{N_gb / 1000:.2f}")

        # ---------- EC4 计算 ----------
        gamma_s = 1.1
        gamma_c = 1.5
        fc_prime = f_c
        N_ec4 = composite_column_plastic_resistance(A_s, f, gamma_s, A_c, fc_prime, gamma_c)
        self.lineEdit22.setText(f"{N_ec4 / 1000:.2f}")

        # 注：已删除 update_chart() 调用

    def load_data_from_file(self):
        """从文件加载数据并填充到 lineEdit00~lineEdit19"""
        # 打开文件选择对话框
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 读取每行数值，跳过空行
            values = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        values.append(float(line))
                    except ValueError:
                        # 如果不是数字，跳过或处理？这里选择跳过非数字行
                        continue

            # 填充到 lineEdit
            for i in range(20):
                line_edit = getattr(self, f'lineEdit{i:02d}')
                if i < len(values):
                    line_edit.setText(f"{values[i]:.4f}")
                else:
                    line_edit.setText("")  # 多余的行清空

            QtWidgets.QMessageBox.information(self, "成功", f"已从文件中加载 {min(20, len(values))} 个数据")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"读取文件失败：{str(e)}")

    def clear_all_inputs(self):
        """清除所有 lineEdit00~lineEdit19 的内容"""
        for i in range(20):
            line_edit = getattr(self, f'lineEdit{i:02d}')
            line_edit.clear()
        # 可选：也清除结果框
        self.lineEdit20.clear()
        self.lineEdit21.clear()
        self.lineEdit22.clear()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())