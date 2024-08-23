import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def jpg_2_mp4(image_path, images, save_mp4_path, fps=15, frameSize=(1920,1088)):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videowrite = cv2.VideoWriter(save_mp4_path, fourcc, fps, frameSize)

        for image in tqdm(images):
                img = cv2.imread(image_path + image)
                assert img is not None, "image error ~~~ "
                videowrite.write(img)
        print('mp4 has been successfully saved in {} \n'.format(save_mp4_path))

def concat_4_img(image, save_path, title="concat_images"):
        # cv2 : bgr
        img0 = cv2.imread(raw_path + image)
        img1 = cv2.imread(path_1 + image)
        img2 = cv2.imread(path_2 + image)
        img3 = cv2.imread(path_3 + image)
        img4 = cv2.imread(path_4 + image)
        img5 = cv2.imread(path_5 + image)

        # convert : bgr --> rgb
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
        img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)

        # # constrained_layout=True,#类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
        # fig = plt.figure(figsize =(39, 22), constrained_layout=True)
        # # GridSpec将fiure分为2行4列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure
        # gs = GridSpec(2, 4, figure=fig)

        fig = plt.figure(figsize =(59, 22), constrained_layout=True)
        gs = GridSpec(2, 6, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0:2])
        plt.imshow(img0)
        plt.title('raw', fontsize=20)
        plt.axis('off')


        ax2 = fig.add_subplot(gs[0, 2:4])
        plt.imshow(img1)
        plt.title('5', fontsize=20)
        plt.axis('off')

        plt.subplot(gs[0, 4:6])
        plt.imshow(img2)
        plt.title('6', fontsize=20)
        plt.axis('off')


        #同样可以使用基于pyplot api的方式
        plt.subplot(gs[1, 0:2])
        plt.imshow(img3)
        plt.title('7', fontsize=20)
        plt.axis('off')

        plt.subplot(gs[1, 2:4])
        plt.imshow(img4)
        plt.title('8', fontsize=20)
        plt.axis('off')

        plt.subplot(gs[1, 4:6])
        plt.imshow(img5)
        plt.title('9', fontsize=20)
        plt.axis('off')


        fig.suptitle(title, color='r', fontsize=22)
        # plt.show()
        plt.savefig(save_path + image)
        plt.close()


def main():
        global raw_path, path_1, path_2, path_3, path_4, path_5
        raw_path = '/mnt/e/datasets/dataset/presentation/temp_val_ripple/'
        path_1 = '/mnt/e/datasets/dataset/presentation/batch_0-5_640-bs16-temp_val_ripple_640-0.25/'
        path_2 = '/mnt/e/datasets/dataset/presentation/all_part_ripple-temp_val_ripple_640-0.25/'
        path_3 = '/mnt/e/datasets/dataset/presentation/all_part_ripple_single_class-temp_val_ripple_640-0.25/'
        path_4 = '/mnt/e/datasets/dataset/presentation/all_part_ripple_blank-temp_val_ripple_640-0.25/'
        path_5 = '/mnt/e/datasets/dataset/presentation/single_true_blank-temp_val_ripple_640-0.25/'

        save_path = '/mnt/e/datasets/dataset/presentation/model_res_compare/'
        # mp4_save_path = '/mnt/e/datasets/dataset/images/0427_demo/mp4/concat_2.mp4'

        os.makedirs(save_path, exist_ok=True)

        images = os.listdir(raw_path)

        print('concate result images ... \n')
        for image in tqdm(images):
               concat_4_img(image, save_path, title="ces_model_compare")
        


if __name__ == '__main__':
        main()


