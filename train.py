from utils import *

if __name__ == '__main__':
    imgs,bboxes, mean_rshape_center,mean_rshape_corner,center_shapes,corner_shapes = LoadData(data_path = args.data_dir)
    train_imgs, train_bboxes, train_init_center, train_init_corner,gt_center_shapes,gt_corner_shapes = GetTrainData(imgs, bboxes,
                                                    center_shapes,corner_shapes, mean_rshape_center, mean_rshape_corner)
    net,pca = GlobalRegression(train_imgs, train_init_center, train_init_corner,
                gt_center_shapes,gt_corner_shapes,mean_rshape_center, mean_rshape_corner,pre_train='./model/Jul25-1445_root_8/best_corner')

