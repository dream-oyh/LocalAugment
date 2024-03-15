import os
import random
from contextlib import suppress

import cv2
import matplotlib.patches as pch
import matplotlib.pyplot as plt
import numpy as np
import pandas


class Transformer:
    def __init__(self, source_folder, target_folder, label_folder, img_number):
        os.makedirs(target_folder, exist_ok=True)
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.img_number = img_number
        self.save_names = []
        self.labels, self.images, self.width, self.height = self.random_read_image(
            source_folder, label_folder, img_number
        )

    def random_read_image(self, source_folder, label_folder, img_number):
        """
        从 source_folder 中随机读取 img_number 张图片，并读取对应标签文件，返回已读取的图片列表，标签列表和图片长宽

        注意保持同一文件夹下图片长宽的一致性，本转换器不支持处理同一文件夹下存在不同长宽的图片

        注意标签采用'.txt'格式，以空格为分隔符，标签矩形框采用矩形中点与长宽的定义方式
        """
        filenames = os.listdir(source_folder)
        selected_filenames = random.sample(filenames, img_number)
        images = []
        labels = []

        for filename in selected_filenames:
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img_path = os.path.join(source_folder, filename)
                img_name, _ = os.path.splitext(filename)
                self.save_names.append(img_name)
                img = cv2.imread(img_path)
                images.append(img)
                height, width, _ = img.shape

                label = self._read_labels(label_folder, img_name)
                labels.append(label)

        return labels, images, width, height

    def _read_labels(self, label_folder, imgname):
        """
        从 label_folder 中读取某图片对应标签
        """
        label_path = os.path.join(label_folder, imgname + ".txt")
        label = pandas.read_csv(
            label_path,
            sep=" ",
            header=None,
            names=["label", "center_x", "center_y", "width", "height"],
        ).to_numpy()
        return label

    def Crop(self, crop_size):
        """
        图像裁剪变换器
        :crop_size 裁剪框位置尺寸，tuple 格式，存储变量为：(x1,x2,y1,y2)
        """
        x1 = crop_size[0]
        x2 = crop_size[1]
        y1 = crop_size[2]
        y2 = crop_size[3]

        w = self.width
        h = self.height

        for i in range(len(self.images)):
            self.images[i] = self.images[i][y1:y2, x1:x2]

        for j in range(len(self.labels)):
            self.labels[j][0:, 1] = (self.labels[j][0:, 1] * w - x1) / (x2 - x1)
            self.labels[j][0:, 2] = (self.labels[j][0:, 2] * h - y1) / (y2 - y1)
            self.labels[j][0:, 3] = (self.labels[j][0:, 3] * w) / (x2 - x1)
            self.labels[j][0:, 4] = (self.labels[j][0:, 4] * h) / (y2 - y1)

        for k in range(len(self.save_names)):
            self.save_names[k] += "_c"

        self.width = x2 - x1
        self.height = y2 - y1

    def Rotate(self, angle: tuple, shuffle=True):
        """
        图像旋转器
        :angle 为旋转角度
        :shuffle 是否采用随机旋转，默认为 True
        开启随机旋转后，angle 需要用 tuple 格式给出旋转范围，采用角度制，逆时针为正，顺时针为负
        """
        center = (self.width / 2, self.height / 2)
        new_dim = (self.width, self.height)
        images_len = len(self.images)
        labels_len = len(self.labels)
        names_len = len(self.save_names)

        for i, j, k in zip(range(images_len), range(labels_len), range(names_len)):
            if shuffle:
                a = random.randint(angle[0], angle[1])
            else:
                a = angle
            M = cv2.getRotationMatrix2D(center=center, angle=-a, scale=1.0)
            self.images[i] = cv2.warpAffine(self.images[i], M, new_dim)
            formatted_a = "{:03d}".format(a)
            self.save_names[k] += f"_rot{formatted_a}"

            rec_loc = self.labels[j][0:, 0:3]
            center_scale = [0.5, 0.5]
            rec_label = self._read_rec(self.labels[j])
            rec_corner = self._caculate_corner(rec_label)
            rec_loc = np.hstack((rec_loc, rec_corner))

            for i in np.arange(1, 11, 2):
                rec_loc[0:, i : (i + 2)] = self._rotate_caculate(
                    center_scale,
                    rec_loc[0:, i : (i + 2)],
                    a,
                )
            self.labels[j] = self._label_trans(rec_loc)

    def flipHorizontal(self):
        """
        图像水平翻转
        """
        for i in range(len(self.images)):
            self.images[i] = cv2.flip(self.images[i], 1)

        for j in range(len(self.labels)):
            self.labels[j][0:, 1] = 1 - self.labels[j][0:, 1]

        for k in range(len(self.save_names)):
            self.save_names[k] += "_hf"

    def randomFlipHorizontal(self, p):
        """
        图像水平翻转
        :p 随机翻转的触发概率
        """
        images_len = len(self.images)
        labels_len = len(self.labels)
        names_len = len(self.save_names)

        for i, j, k in zip(range(images_len), range(labels_len), range(names_len)):
            pro = random.uniform(0, 1)
            if pro <= p:
                self.images[i] = cv2.flip(self.images[i], 1)
                self.labels[j][0:, 1] = 1 - self.labels[j][0:, 1]
                self.save_names[k] += "_hf"
            if pro > p:
                pass

    def flipVertical(self):
        """
        图像垂直翻转
        """
        for i in range(len(self.images)):
            self.images[i] = cv2.flip(self.images[i], 0)

        for j in range(len(self.labels)):
            self.labels[j][0:, 2] = 1 - self.labels[j][0:, 2]

        for k in range(len(self.save_names)):
            self.save_names[k] += "_vf"

    def randomFlipVertical(self, p):
        """
        图像垂直翻转
        :p 随机翻转的触发概率
        """
        images_len = len(self.images)
        labels_len = len(self.labels)
        names_len = len(self.save_names)

        for i, j, k in zip(range(images_len), range(labels_len), range(names_len)):
            pro = random.uniform(0, 1)
            if pro <= p:
                self.images[i] = cv2.flip(self.images[i], 0)
                self.labels[j][0:, 1] = 1 - self.labels[j][0:, 1]
                self.save_names[k] += "_vf"
            if pro > p:
                pass

    def randomGammaLight(self, dark=(0.5, 0.7), light=(2, 5)):
        """
        利用伽马函数随机调整图片对比度
        :dark 调暗范围，视生成图片实际效果而定
        :light 调亮范围，视生成图片实际效果而定
        """
        for i in range(len(self.images)):
            gamma = self._gen_gamma(dark, light)
            self.images[i] = self._adjust_gamma(self.images[i], gamma=gamma)

        for k in range(len(self.save_names)):
            self.save_names[k] += "_l"

    def randomGaussionNoise(self, noise_scale=(10, 20)):
        """
        为图片添加高斯噪声
        :noise_scale 噪声添加范围，默认为 (10,20)
        """
        for i in range(len(self.images)):
            self.images[i] = self._gen_noise(self.images[i], noise_scale=noise_scale)

        for k in range(len(self.save_names)):
            self.save_names[k] += "_n"

    def Resize(self, time, shuffle=True):
        """
        缩放图像
        :time 缩放倍数，>1 放大，<1 缩小
        :shuffle 是否随机放大，默认为 True
        开启随机缩放后，time 需要用 tuple 格式给出缩放比例抽取范围
        """
        w = self.width
        h = self.height

        for i in range(len(self.images)):
            if shuffle:
                time = random.uniform(*time)
                self.images[i] = cv2.resize(self.images[i], (time * w, time * h))

        for k in range(len(self.save_names)):
            self.save_names[k] += "_res"

        self.width = time * w
        self.height = time * h

    def Sharpen(self):
        """
        锐化图像
        """
        for i in range(len(self.images)):
            self.images[i] = self._sharpen(self.images[i])

        for k in range(len(self.save_names)):
            self.save_names[k] += "_sh"

    def Save(self):
        """
        保存图像与标签
        """
        for img, name in zip(self.images, self.save_names):
            image_path = os.path.join(self.target_folder, f"{name}.png")
            cv2.imwrite(image_path, img)

        for label, name in zip(self.labels, self.save_names):
            label = np.around(label, 3)
            label_df = pandas.DataFrame(label)
            label_path = os.path.join(self.target_folder, f"{name}.txt")
            label_df.to_csv(label_path, sep=" ", header=None, index=None)

    def Compose(self, trans_list: list):
        """
        串联多个变换器并依次转换
        """
        for trans in trans_list:
            trans

    def Show(self, num_cols, num_rows):
        """
        显示前 n 张图像，需要指定图像排列方式
        :num_cols 图像列数
        :num_rows 图像行数
        """
        num = num_rows * num_rows
        fig, axes = plt.subplots(num_rows, num_cols)
        with suppress(FileNotFoundError):
            filenames = os.listdir(self.target_folder)
        counts = 0
        files = []
        for file in filenames:
            if file.endswith(".png") or file.endswith(".jpg"):
                counts += 1
                files.append(file)

        if num > counts:
            raise Exception("图片数量不足")

        if num == 1:
            self._draw_single(files[0], axes)
        else:
            for j in range(num_cols):
                for i in range(num_rows):
                    file = files[j * num_cols + i]
                    self._draw_single(file, axes[i, j])

    def _draw_single(self, image_dir, axes):
        imgname, _ = os.path.splitext(image_dir)
        label = self._read_labels(self.target_folder, imgname)
        img = cv2.imread(os.path.join(self.target_folder, image_dir))
        axes.imshow(img)
        axes.axis("off")
        rec_list = self._gen_rec(label)
        for rec in rec_list:
            axes.add_patch(rec)

    def _sharpen(self, img):
        image_blur = cv2.GaussianBlur(img, (3, 3), 0) 
        Laplace_kernel = np.array(
            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32
        ) 
        laplace_image = cv2.filter2D(image_blur, -1, Laplace_kernel)
        return laplace_image

    def _gen_noise(self, img, noise_scale=(10, 20)):
        noise_std = random.randint(*noise_scale)
        noise = np.random.randn(*img.shape) * noise_std
        noisy_image = np.clip(img + noise, 0, 255).astype(np.uint8)
        return noisy_image

    def _adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, table)

    def _gen_gamma(self, dark, light):
        gamma1 = random.uniform(
            dark[0], dark[1]
        )  # 矫正系数：越大越亮、越小越暗（本处系数的上下限根据生成图片的实际效果而定）
        gamma2 = random.uniform(
            light[0], light[1]
        )  # 矫正系数：越大越亮、越小越暗（本处系数的上下限根据生成图片的实际效果而定）
        judge = random.uniform(0, 1)
        # 在 gamma1（调暗）和 gamma2（调亮）中进行等概率二选一
        if judge >= 0.5:
            gamma = gamma1
        else:
            gamma = gamma2
        return gamma

    def _read_rec(self, label):
        center_x = label[0:, 1].reshape(1, -1).T
        center_y = label[0:, 2].reshape(1, -1).T
        rec_width = label[0:, 3].reshape(1, -1).T
        rec_height = label[0:, 4].reshape(1, -1).T
        return (center_x, center_y, rec_width, rec_height)

    def _caculate_corner(self, rec_label):
        center_x, center_y, rec_width, rec_height = rec_label
        rec_nw_x = center_x - rec_width / 2
        rec_nw_y = center_y - rec_height / 2
        rec_ne_x = center_x + rec_width / 2
        rec_ne_y = center_y - rec_height / 2
        rec_se_x = center_x + rec_width / 2
        rec_se_y = center_y + rec_height / 2
        rec_sw_x = center_x - rec_width / 2
        rec_sw_y = center_y + rec_height / 2
        return np.hstack(
            (
                rec_nw_x,
                rec_nw_y,
                rec_ne_x,
                rec_ne_y,
                rec_se_x,
                rec_se_y,
                rec_sw_x,
                rec_sw_y,
            )
        )

    def _rotate_caculate(self, center, rec_loc, angle):
        x0 = center[0] * self.width
        y0 = center[1] * self.height
        x1 = rec_loc[0:, 0] * self.width
        y1 = rec_loc[0:, 1] * self.height
        if angle > 0:
            x2 = (
                (x1 - x0) * np.cos(np.radians(angle))
                - (y1 - y0) * np.sin(np.radians(angle))
                + x0
            ) / self.width
            y2 = (
                (y1 - y0) * np.cos(np.radians(angle))
                + (x1 - x0) * np.sin(np.radians(angle))
                + y0
            ) / self.height
        if angle < 0:
            x2 = (
                (x1 - x0) * np.cos(np.radians(-angle))
                + (y1 - y0) * np.sin(np.radians(-angle))
                + x0
            ) / self.width
            y2 = (
                (y1 - y0) * np.cos(np.radians(-angle))
                - (x1 - x0) * np.sin(np.radians(-angle))
                + y0
            ) / self.height
        return np.array([x2, y2]).T

    def _label_trans(self, rec_loc):
        rec_new_x_min, rec_new_x_max = self._rec_x(rec_loc)
        rec_new_y_min, rec_new_y_max = self._rec_y(rec_loc)
        rec_par = np.hstack(
            (rec_new_x_max - rec_new_x_min, rec_new_y_max - rec_new_y_min)
        )
        rec_center = rec_loc[0:, 1:3]
        label_new = np.hstack((rec_loc[0:, 0].reshape(1, -1).T, rec_center, rec_par))
        return label_new

    def _rec_x(self, rec_loc):
        rec_new_x = np.vstack(
            (rec_loc[0:, 3], rec_loc[0:, 5], rec_loc[0:, 7], rec_loc[0:, 9])
        )
        rec_new_x_min = np.min(rec_new_x, axis=0).reshape(1, -1).T
        rec_new_x_max = np.max(rec_new_x, axis=0).reshape(1, -1).T
        return rec_new_x_min, rec_new_x_max

    def _rec_y(self, rec_loc):
        rec_new_y = np.vstack(
            (rec_loc[0:, 4], rec_loc[0:, 6], rec_loc[0:, 8], rec_loc[0:, 10])
        )
        rec_new_y_min = np.min(rec_new_y, axis=0).reshape(1, -1).T
        rec_new_y_max = np.max(rec_new_y, axis=0).reshape(1, -1).T
        return rec_new_y_min, rec_new_y_max

    def _get_label(self, label):
        width = self.width
        height = self.height
        label[0:, 1] *= width
        label[0:, 3] *= width
        label[0:, 2] *= height
        label[0:, 4] *= height
        return label

    def _gen_rec(self, label):
        label = self._get_label(label)
        rec_num = label.shape[0]
        rec = []
        for i in range(rec_num):
            rec_width = label[i][3]
            rec_height = label[i][4]
            center_x = label[i][1]
            center_y = label[i][2]
            x = center_x - rec_width / 2
            y = center_y - rec_height / 2

            rec.append(
                pch.Rectangle(
                    xy=(x, y),
                    width=rec_width,
                    height=rec_height,
                )
            )
            rec[i].set_edgecolor(self._color_map(label[i][0]))
            rec[i].set_facecolor("none")
            rec[i].set_linewidth(2)
        return rec

    def _color_map(self, label: int):
        if label == 0:
            return "red"
        if label == 1:
            return "blue"
        if label == 2:
            return "green"
        if label == 3:
            return "yellow"
        if label == 4:
            return "black"
        if label == 5:
            return "purple"
        if label == 6:
            return "pink"
        if label == 7:
            return "orange"
        if label == 8:
            return "brown"
        if label == 9:
            return "gray"
        if label == 10:
            return "olive"
        if label == 11:
            return "cyan"
        if label == 12:
            return "teal"
        if label == 13:
            return "lime"
        if label == 14:
            return "maroon"
        if label == 15:
            return "navy"
        if label == 16:
            return "magenta"
