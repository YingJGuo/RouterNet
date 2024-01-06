from array import array
import pickle
import time
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import copy
import random
from spineNet import SpineNet
from einops import repeat
import albumentations as A
import torch.nn as nn
from cal_cobb import calculate_cobb
import glob
import argparse

################### params ###################

parser = argparse.ArgumentParser(description='CenterNet Modification Implementation')
parser.add_argument('--param_epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('-b_size','--param_batch_size', type=int, default=1, help='Number of epochs')#default=40
parser.add_argument('--data_dir', type=str, default='./dataset/MICCAI_128/', help='data directory')
parser.add_argument('--pre_train', type=bool, default=False, help='data directory')
parser.add_argument('--init_lr', type=float, default=1e-3, help='Init learning rate')
parser.add_argument('--weights_path', type=str, default='', help='weights directory')
parser.add_argument('--param_local_win', type=array,
                    default=np.array([[96, 64], [60, 40], [60, 40], [40, 20], [32, 16]]))
parser.add_argument('-aug_num','--param_augment_num', type=int, default=80)#default=80
parser.add_argument('--param_recurrent_num', type=int, default=5)
parser.add_argument('--image_size', default=(704, 384))
parser.add_argument('--param_flipped', type=bool, default=True)
parser.add_argument('--param_shuffle', type=bool, default=True)
parser.add_argument('--param_landmark_num', type=int, default=17)

parser.add_argument('--root_idx', type=int, default=8)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


#####dataset
class MiniBatch_Generator(Dataset):
    def __init__(self, train_imgs, train_init_center, train_init_corner, gt_center_shapes, gt_corner_shapes,
                 transform=None):
        self.train_imgs = train_imgs
        self.train_init_center = train_init_center
        self.train_init_corner = train_init_corner
        self.gt_center_shapes = gt_center_shapes
        self.gt_corner_shapes = gt_corner_shapes
        self.transform = transform

        assert len(self.train_imgs) == self.train_init_center.shape[0]
        assert len(self.train_imgs) == self.train_init_corner.shape[0]

    def __getitem__(self, index):
        while True:
            initial_center_shapes = torch.from_numpy(self.train_init_center[index].astype(np.float32))
            init_corner_off, _ = GenerateCornerVec(self.train_init_corner[index])
            init_corner_off = torch.from_numpy(init_corner_off.astype(np.float32))
            img = self.train_imgs[index]
            img = repeat(img, 'h w ->h w c', c=3)
            corner_shapes = self.gt_corner_shapes[index]
            center_shapes = GetCenterFromCorner(corner_shapes)

            key_points_center = center_shapes.tolist()
            center_nums = len(key_points_center)
            key_points_corner = corner_shapes.tolist()
            key_points = key_points_center + key_points_corner
            if self.transform:
                transformed = self.transform(image=img, keypoints=key_points)
                transformed_image = torch.from_numpy((transformed['image'] / 255).transpose(2, 0, 1).astype(np.float32))
                transformed_keypoints = transformed['keypoints']
                transformed_center_pts = np.array(transformed_keypoints[:center_nums])
                transformed_corner_pts = np.array(transformed_keypoints[center_nums:])
                cv_image = transformed['image']
                for i, pt in enumerate(transformed_keypoints):
                    if i < args.param_landmark_num:
                        cv2.circle(cv_image, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), 2)
                    else:
                        cv2.circle(cv_image, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), 2)

                if transformed_center_pts.shape[0] == args.param_landmark_num and transformed_corner_pts.shape[0] == \
                        args.param_landmark_num * 4:
                    break
                else:
                    index = np.random.randint(0, len(self.train_imgs) - 1)

        center_shapes = torch.from_numpy(transformed_center_pts.reshape(args.param_landmark_num, 2).astype(np.float32))
        corner_shapes = torch.from_numpy(transformed_corner_pts.astype(np.float32))
        corner_vec, angle = GenerateCornerVec(transformed_corner_pts)
        corner_vec = torch.from_numpy(corner_vec.astype(np.float32))
        angle = torch.from_numpy(angle.astype(np.float32))
        return {'img': transformed_image,
                'init_center': initial_center_shapes,
                'init_corner_off': init_corner_off,
                'center_shapes': center_shapes,
                'corner_shapes': corner_shapes,
                'corner_vector': corner_vec,
                'angle': angle
                }

    def __len__(self):
        return len(self.train_imgs)


def GenerateCornerVec(corner_point):
    # this is the corner vector of each center point
    corner_vec = np.zeros((args.param_landmark_num * 4, 2), dtype=np.float32)
    angle = np.zeros((args.param_landmark_num, 1), dtype=np.float32)

    for k in range(args.param_landmark_num):
        pts = corner_point[4 * k:4 * k + 4, :]  # 0123
        cen_x, cen_y = np.mean(pts, axis=0)
        center_pt = np.asarray([cen_x, cen_y], dtype=np.float32)

        angle_xy1 = corner_point[4 * k + 1, :] - corner_point[4 * k, :]  # vector1
        angle_xy2 = corner_point[4 * k + 3, :] - corner_point[4 * k + 2, :]  # vector2
        xyxy = angle_xy1 + angle_xy2
        L1 = np.sqrt(xyxy.dot(xyxy))
        angle[k, :] = xyxy[1] / L1

        assert angle[k, :] <= 1
        for i in range(4):
            corner_vec[4 * k + i, :] = corner_point[k * 4 + i, :] - center_pt
    return corner_vec, angle


def GetCenterFromCorner(corner_point):
    center = np.zeros((args.param_landmark_num, 2), dtype=np.float32)  # this is the corner vector of each center point
    for i in range(args.param_landmark_num):
        pts = corner_point[4 * i:4 * i + 4, :]
        cen_x, cen_y = np.mean(pts, axis=0)
        center[i] = np.asarray([cen_x, cen_y], dtype=np.float32)
    return center


################### shape
# load shape points file
def ReadShape(path, offsets=None):
    file = open(path).readlines()
    shape = []
    for i in range(args.param_landmark_num):
        pair = file[i].split()
        if offsets == None:
            shape.append([float(pair[0]), float(pair[1])])
        else:
            shape.append([float(pair[0]) + offsets[0], float(pair[1]) + offsets[1]])
    return np.array(shape)


def ReadCornerShape(path, offsets=None):
    file = open(path).readlines()
    shape = []
    for i in range(args.param_landmark_num * 4):
        pair = file[i].split()
        if offsets == None:
            shape.append([float(pair[0]), float(pair[1])])
        else:
            shape.append([float(pair[0]) + offsets[0], float(pair[1]) + offsets[1]])
    return np.array(shape)

# transform point relative to center
def Shape2Relative(shape, bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    rshape = shape.copy()
    rshape[:, 0] = (rshape[:, 0] - cx) * 2 / w
    rshape[:, 1] = (rshape[:, 1] - cy) * 2 / h
    return rshape

# transform to absolute coord
def Shape2Absolute(rshape, bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    ashape = rshape.copy()
    ashape[:, 0] = ashape[:, 0] * w / 2 + cx  # x
    ashape[:, 1] = ashape[:, 1] * h / 2 + cy  # y
    return ashape

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# generate bbox from shape
def GenerateBBox(shape):
    h = np.max(shape[:, 1]) - np.min(shape[:, 1])
    c = np.mean(shape, axis=0)  # 中心点
    x_center = c[0]
    y_center = c[1]
    x1 = x_center - h / 8.
    x2 = x_center + h / 8.
    y1 = y_center - h / 4.
    y2 = y_center + h / 4.
    return np.array([x1, y1, x2, y2])


########################## error ###################################
def ComputeError(test_imgs, pred_shapes, gt_shapes, padding_list, pt_type='center'):
    err = 0
    or20 = 0
    or20_list = []
    or40 = 0
    or40_list = []
    err_list = []
    point_num = args.param_landmark_num if pt_type == 'center' else args.param_landmark_num * 4
    for i in range(len(pred_shapes)):
        img = test_imgs[i]
        img = img.astype('float32')
        img = np.sum(img, axis=0)
        for j in range(300):
            if img[j] > 0:
                left = j
                break

        shape = pred_shapes[i]
        gt = gt_shapes[i]
        #Metrics OR20 and OR40
        dis = np.sqrt(np.sum(np.square(shape-gt),axis=1))
        or20_i = np.sum(dis>20) / point_num * 100
        or40_i = np.sum(dis>40) / point_num * 100
        or20 += or20_i
        or40 += or40_i
        or20_list.append(or20_i)
        or40_list.append(or40_i)

        #Metrics NMSE & var
        height = float(args.image_size[0] - 0)
        width = float(args.image_size[1] - 2 * left)
        square = np.square(shape - gt)
        square[:, 0] /= np.square(width)
        square[:, 1] /= np.square(height)
        err_i = np.sum(square) / point_num
        err += err_i
        err_list.append(err_i)
    mean_or20 = np.mean(or20_list)
    mean_or40 = np.mean(or40_list)
    mean_err = err/len(pred_shapes)
    std_nmse = np.std(err_list)
    std_or20 = np.std(or20_list)
    std_or40 = np.std(or40_list)

    return mean_err,std_nmse,mean_or20,std_or20,mean_or40,std_or40


def LoadData(data_path):
    imgs = []
    bboxes = []
    center_shapes = []
    corner_shapes = []
    img_list = open(os.path.join(data_path, 'train_img_list')).readlines()
    for img in img_list:
        image = cv2.imread(data_path + img.strip(), 0)
        assert image.shape[0] == 704
        assert image.shape[1] == 384
        imgs.append(image)
        corner_shape_path = data_path + img.split('.')[0] + '_corner.pts'
        if '_raw' in img.strip():
            corner_shape_path = data_path + img.split('.')[0][:-4] + '_corner.pts'
        corner_shape = ReadCornerShape(corner_shape_path, offsets=[0, 0])
        center_shape = GetCenterFromCorner(corner_shape)

        corner_shapes.append(corner_shape)
        center_shapes.append(center_shape)
        bboxes.append(GenerateBBox(center_shapes[-1]))

        if args.param_flipped:
            flipped_img_lr, flipped_center_shape_lr, flipped_corner_shape_lr = FlipImageAndShape(imgs[-1],
                                                                                                 center_shapes[-1],
                                                                                                 corner_shapes[-1])
            imgs.append(flipped_img_lr)
            center_shapes.append(flipped_center_shape_lr)
            corner_shapes.append(flipped_corner_shape_lr)
            bboxes.append(GenerateBBox(center_shapes[-1]))

    bboxes = np.array(bboxes)
    center_shapes = np.array(center_shapes)
    corner_shapes = np.array(corner_shapes)
    mean_rshape_center = np.mean([Shape2Relative(shape, bboxes[i]) for i, shape in enumerate(center_shapes)],
                                 0)
    mean_rshape_corner = np.mean([Shape2Relative(shape, bboxes[i]) for i, shape in enumerate(corner_shapes)],
                                 0)

    print("Number of preprocessed images loaded: ", corner_shapes.shape[0])
    assert len(imgs) == corner_shapes.shape[0]
    return imgs, bboxes, mean_rshape_center, mean_rshape_corner, center_shapes, corner_shapes


#  Flip
def FlipImageAndShape(img, center_shape, corner_shape):
    flipped_img = cv2.flip(img, flipCode=1)  # Horizontal
    width = float(img.shape[1])
    flipped_center_shape = np.zeros(center_shape.shape, dtype=np.float)
    flipped_corner_shape = np.zeros(corner_shape.shape, dtype=np.float)
    flipped_corner_shape_cache = np.zeros((4, 2), dtype=np.float)
    for i in range(center_shape.shape[0]):
        flipped_center_shape[i, 0] = width - center_shape[i, 0]
        flipped_center_shape[i, 1] = center_shape[i, 1]

    for i in range(corner_shape.shape[0]):
        flipped_corner_shape_cache[i % 4, 0] = width - corner_shape[i, 0]
        flipped_corner_shape_cache[i % 4, 1] = corner_shape[i, 1]
        if i % 4 == 3:
            flipped_corner_shape_cache[[0, 1, 2, 3], :] = flipped_corner_shape_cache[[1, 0, 3, 2],
                                                          :]  # corner point exchange order
            flipped_corner_shape[i - 3:i + 1, :] = flipped_corner_shape_cache
    return flipped_img, flipped_center_shape, flipped_corner_shape


# prepare training data, data augmentation, initialize train_shape
def GetTrainData(imgs, bboxes, center_shapes, corner_shapes, mean_shape_center, mean_shape_corner):
    train_imgs = []
    train_init_center = []  # using each shapes box transfered mean shape
    train_init_corner = []
    train_bboxes = []
    gt_center_shapes = []
    gt_corner_shapes = []
    # for every training sample choose $augment_num other training shape as initialization
    for i in range(len(imgs)):
        aug_inds = []
        train_imgs.append(imgs[i])
        train_init_center.append(
            Shape2Absolute(mean_shape_center, bboxes[i]))
        train_init_corner.append(Shape2Absolute(mean_shape_corner, bboxes[i]))

        train_bboxes.append(bboxes[i])
        gt_center_shapes.append(center_shapes[i])
        gt_corner_shapes.append(corner_shapes[i])

        for j in range(args.param_augment_num):
            aug_ind = i
            while aug_ind == i or aug_ind in aug_inds:
                aug_ind = np.random.randint(len(imgs))
            aug_inds.append(aug_ind)
            train_imgs.append(imgs[i])
            train_init_center.append(Shape2Absolute(Shape2Relative(center_shapes[aug_ind], bboxes[aug_ind]), bboxes[i]))
            train_init_corner.append(Shape2Absolute(Shape2Relative(corner_shapes[aug_ind], bboxes[aug_ind]), bboxes[i]))
            train_bboxes.append(bboxes[i])
            gt_center_shapes.append(center_shapes[i])
            gt_corner_shapes.append(corner_shapes[i])

    train_init_center = np.array(train_init_center)
    train_init_corner = np.array(train_init_corner)
    train_bboxes = np.array(train_bboxes)
    gt_center_shapes = np.array(gt_center_shapes)
    gt_corner_shapes = np.array(gt_corner_shapes)

    assert len(train_imgs) == train_init_center.shape[0]
    assert len(train_imgs) == train_init_corner.shape[0]
    assert len(train_imgs) == train_bboxes.shape[0]
    assert len(train_imgs) == gt_center_shapes.shape[0]
    # shuffle
    print("Number of images for training: ", len(train_imgs))
    indexs = np.arange(len(train_imgs))
    np.random.shuffle(indexs)

    train_imgs = [train_imgs[indexs[i]] for i in range(len(train_imgs))]
    return train_imgs, train_bboxes[indexs], train_init_center[indexs], train_init_corner[indexs], gt_center_shapes[
        indexs], gt_corner_shapes[indexs]


def Evaluate(imgs, mean_center_shapes, mean_corner_shapes_off, model):
    model.eval()
    predictions_center = np.zeros((len(imgs), args.param_landmark_num * 2), dtype=np.float32)
    predictions_corner = np.zeros((len(imgs), args.param_landmark_num * 2 * 4), dtype=np.float32)

    mean_center_shapes = torch.from_numpy(mean_center_shapes.astype(np.float32)).cuda()
    mean_corner_shapes_off = torch.from_numpy(mean_corner_shapes_off.astype(np.float32)).cuda()

    for i in range(len(imgs)):
        b_x = torch.from_numpy(imgs[i].astype(np.float32)).cuda()
        b_x = b_x / 255
        b_x = b_x.reshape(1, 1, args.image_size[0], args.image_size[1])
        b_x = b_x.repeat((1, 3, 1, 1))

        predict_dict = model(b_x, 1, mean_center_shapes[i].unsqueeze(0), mean_corner_shapes_off[i].unsqueeze(0),
                             args.root_idx)
        center_pos = predict_dict['center_pts_pos_2'].data.cpu().numpy().reshape(-1)
        corner_pos = predict_dict['corner_pts_pos_2'].data.cpu().numpy().reshape(-1)  ####rle loss
        predictions_center[i, :] = center_pos
        predictions_corner[i, :] = corner_pos

    updates_center = predictions_center.reshape(len(imgs), args.param_landmark_num, 2)
    updates_corner = predictions_corner.reshape(len(imgs), args.param_landmark_num * 4, 2)

    return updates_center, updates_corner


def TestGlobalRegression(imgs, mean_center_shapes, mean_corner_shapes_off, net):
    heads = {
        'corner': 2 * 4, \
        'center': 2 * args.param_landmark_num, \
        'center_pt': 2 * 1}
    model = SpineNet(heads=heads, center_num=args.param_landmark_num, recurrent_num=args.param_recurrent_num,
                     local_win=args.param_local_win, pretrained=True)
    model.load_state_dict(net)
    model.to(device)
    model.eval()
    predictions_center = np.zeros((len(imgs), args.param_landmark_num * 2), dtype=np.float32)
    predictions_corner = np.zeros((len(imgs), args.param_landmark_num * 2 * 4), dtype=np.float32)

    mean_center_shapes = torch.from_numpy(mean_center_shapes.astype(np.float32)).cuda()
    mean_corner_shapes_off = torch.from_numpy(mean_corner_shapes_off.astype(np.float32)).cuda()


    for i in range(len(imgs)):
        b_x = torch.from_numpy(imgs[i].astype(np.float32)).cuda()
        b_x = b_x / 255
        b_x = b_x.reshape(1, 1, args.image_size[0], args.image_size[1])
        b_x = b_x.repeat((1, 3, 1, 1))

        predict_dict = model(b_x, 1, mean_center_shapes[i].unsqueeze(0), mean_corner_shapes_off[i].unsqueeze(0),
                             args.root_idx)
        center_pos = predict_dict['center_pts_pos_2'].data.cpu().numpy().reshape(-1)
        corner_pos = predict_dict['corner_pts_pos_2'].data.cpu().numpy().reshape(-1)  ####rle loss
        predictions_center[i, :] = center_pos
        predictions_corner[i, :] = corner_pos

    updates_center = predictions_center.reshape(len(imgs), args.param_landmark_num, 2)
    updates_corner = predictions_corner.reshape(len(imgs), args.param_landmark_num * 4, 2)

    return updates_center, updates_corner


def load_checkpoint(model, pre_trained_model):
    print("loading checkpoint...")
    model_dict = model.state_dict()  # torch.load(checkpoint)
    pretrained_dict = pre_trained_model  # .state_dict()
    new_dict = {k: v for k, v in pretrained_dict.items() if
                k in model_dict and not any(x in k for x in ()) and v.shape == model_dict[k].shape}
    model_dict.update(new_dict)
    print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
    model.load_state_dict(model_dict)
    print("loaded finished!")
    return model


def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2),
                         nn.Tanh())


# do global regression
def GlobalRegression(train_imgs, train_init_center, train_init_corner, gt_center_shapes, gt_corner_shapes,
                     mean_rshape_center, mean_rshape_corner, pre_train=None):
    # loading test data
    test_imgs, test_gt_center_shapes, test_gt_corner_shapes, test_bboxes, _, padding_list, angle_gt = LoadValData(
        args.data_dir)
    test_mean_center = np.array([Shape2Absolute(mean_rshape_center, test_bboxes[i]) for i in range(len(test_imgs))])
    test_mean_corner_off = []
    for i in range(len(test_imgs)):
        test_mean_corner = Shape2Absolute(mean_rshape_corner, test_bboxes[i])
        test_mean_corner_offset, _ = GenerateCornerVec(test_mean_corner)
        test_mean_corner_off.append(test_mean_corner_offset)
    test_mean_corner_off = np.array(test_mean_corner_off)

    corner_errors_cache = 50
    smape_cache = 20
    heads = {
        'corner': 2 * 4, \
        'center': 2 * args.param_landmark_num, \
        'center_pt': 2 * 1}

    model = SpineNet(heads=heads, center_num=args.param_landmark_num, recurrent_num=args.param_recurrent_num,
                     local_win=args.param_local_win, pretrained=True)

    if pre_train != None:  # None
        print("pre_train", pre_train)
        saved_model, _, _ = LoadModels(pre_train)
        load_checkpoint(model, saved_model)

    model.to(device)
    model.train()
    loss_func_1 = torch.nn.SmoothL1Loss(beta=1)  # beta=0.01
    loss_func_2 = torch.nn.SmoothL1Loss(beta=0.001)  # beta=0.001  0.01

    optimizer_1 = torch.optim.Adam(model.parameters(), lr=args.init_lr)  # -3
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1, gamma=0.99, last_epoch=-1)

    image_aug = A.Compose([
        A.GaussianBlur(blur_limit=(3, 5), always_apply=False, p=0.5),
        A.RandomGamma(gamma_limit=(50, 150), p=0.5),
        A.RandomBrightnessContrast(p=0.5), A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, \
                                                              rotate_limit=10, interpolation=1, border_mode=4,
                                                              value=None, mask_value=None,
                                                              always_apply=False, p=0.5)],
        keypoint_params=A.KeypointParams(format='xy'))

    #train__dataset
    training_dataset = MiniBatch_Generator(train_imgs, train_init_center, train_init_corner, gt_center_shapes,
                                           gt_corner_shapes, transform=image_aug)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader = DataLoader(training_dataset, batch_size=args.param_batch_size, num_workers=32,
                            pin_memory=True, shuffle=True, drop_last=True, worker_init_fn=seed_worker)

    logdir = os.path.join('logs', time.strftime("%b%d-%H%M", time.localtime()) + '_root_' + str(args.root_idx))  #
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    modeldir = os.path.join('model', time.strftime("%b%d-%H%M", time.localtime()) + '_root_' + str(args.root_idx))
    if not os.path.isdir(modeldir):
        os.makedirs(modeldir)
    save_args(modeldir)
    writer = SummaryWriter(logdir)
    num = 0

    ######## start training
    for ep in range(args.param_epoch):
        t2 = time.time()
        loss_list = []
        for train_data in dataloader:
            gt_images = train_data['img'].cuda()
            batch_initial_center = train_data['init_center'].cuda()
            batch_initial_corner_off = train_data['init_corner_off'].cuda()

            predict_dict = model(gt_images, args.param_batch_size, batch_initial_center, batch_initial_corner_off,
                                 args.root_idx)

            #gt
            gt_center = train_data['center_shapes'].cuda()
            gt_corner = train_data[
                'corner_shapes'].cuda()

            center_loss = 0
            corner_loss = 0
            center_pt_loss = loss_func_1(predict_dict['center_pt_pos'],
                                         train_data['center_shapes'][:, args.root_idx, :].cuda().reshape(
                                             args.param_batch_size, 1 * 2))
            center_loss += loss_func_2(predict_dict['center_pts_pos_2'], gt_center)
            corner_loss += loss_func_2(predict_dict['corner_pts_pos_2'],
                                       gt_corner)
            loss = center_loss + corner_loss
            loss_list.append(loss.item())
            writer.add_scalar('loss', loss, global_step=num)
            writer.add_scalar('corner_loss', corner_loss, global_step=num)
            writer.add_scalar('center_loss', center_loss, global_step=num)
            writer.add_scalar('center_pt_loss', center_pt_loss, global_step=num)  ###center_pt_loss

            num += 1
            optimizer_1.zero_grad()
            loss.backward()
            optimizer_1.step()

            if num % 5 == 0:
                updates_center, update_corner = Evaluate(test_imgs, test_mean_center, test_mean_corner_off,
                                                         model)

                updates_center_cp = copy.deepcopy(updates_center)
                update_corner_cp = copy.deepcopy(update_corner)

                center_err,center_err_std,center_or20_mean,center_or20_std,center_or40_mean,center_or40_std = ComputeError(test_imgs, updates_center_cp, test_gt_center_shapes, padding_list, 'center')
                corner_err,corner_err_std,corner_or20_mean,corner_or20_std,corner_or40_mean,corner_or40_std = ComputeError(test_imgs, update_corner_cp, test_gt_corner_shapes, padding_list, 'corner')

                MD,SMAPE,pred_cobb,ED,CD,CMAE,SMAPE_Min,SMAPE_Max = calculate_cobb(update_corner_cp, test_gt_corner_shapes, test_imgs)

                print('step = {},center_error = {}'.format(num, center_err))
                print('step = {},error = {}'.format(num, corner_err))
                print('CMAE = {}, SMAPE = {}'.format(CMAE, SMAPE))

                writer.add_scalar('var_center_error', center_err, global_step=num)
                writer.add_scalar('var_corner_error', corner_err, global_step=num)
                if corner_err < corner_errors_cache:
                    corner_errors_cache = corner_err
                    print('saving the best model! corner error = {}'.format(corner_err))
                    SaveModels(model, mean_rshape_center, mean_rshape_corner, os.path.join(modeldir, 'best_corner'))
                    with open(os.path.join(modeldir, 'log.txt'), 'a') as log_file:
                        log_file.write('best corner mse = {}, step = {} \n'.format(corner_err, num))
                if SMAPE < smape_cache:
                    smape_cache = SMAPE
                    print('saving the best model! smape = {}'.format(smape_cache))
                    SaveModels(model, mean_rshape_center, mean_rshape_corner, os.path.join(modeldir, 'best_smape'))
                    with open(os.path.join(modeldir, 'log.txt'), 'a') as log_file:
                        log_file.write(
                            'best smape , corner_err= {},{} step = {} \n'.format(smape_cache, corner_err, num))
                if SMAPE < smape_cache and corner_err < corner_errors_cache:
                    smape_cache = SMAPE
                    corner_errors_cache = corner_err
                    print('saving the best model! smape = {}'.format(smape_cache))
                    SaveModels(model, mean_rshape_center, mean_rshape_corner,
                               os.path.join(modeldir, 'best_smape_and_corner'))
                    with open(os.path.join(modeldir, 'log.txt'), 'a') as log_file:
                        log_file.write(
                            'best smape , corner_err= {},{} step = {} \n'.format(smape_cache, corner_err, num))

                model.train()
        scheduler.step()
        print('epoch:{:2d}, mean_loss:{:.4f}, use:{:.4f}s'.format(ep, np.mean(loss_list), time.time() - t2))

    return model


def save_args(model_path):
    argsDict = args.__dict__
    with open(os.path.join(model_path, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


######################### val phase ##############################
# load val data
def LoadValData(data_path):
    imgs = []
    center_shapes = []
    corner_shapes = []
    bboxes = []
    name = []
    padding_list = []
    angle_gt = []
    test_img_list = glob.glob(data_path + 'refactory/training/*.jpg')
    for i, img in enumerate(test_img_list):
        image = cv2.imread(img.strip(), 0)
        padding_list.append([0, 0])
        assert image.shape[0] == 704
        assert image.shape[1] == 384
        imgs.append(image.astype(np.float))

        corner_shape = ReadCornerShape(img.split('.jpg')[0] + '_corner.pts', offsets=[0, 0])
        center_shape = GetCenterFromCorner(corner_shape)
        corner_vec, angle_gt1 = GenerateCornerVec(corner_shape)

        angle_gt.append(angle_gt1)
        center_shapes.append(center_shape)
        corner_shapes.append(corner_shape)

        bboxes.append(GenerateBBox(center_shapes[-1]))
        name.append(os.path.basename(img))

    angle_gt = np.array(angle_gt)

    center_shapes = np.array(center_shapes)
    bboxes = np.array(bboxes)
    corner_shapes = np.array(corner_shapes)
    padding_list = np.array(padding_list)

    return imgs, center_shapes, corner_shapes, bboxes, name, padding_list, angle_gt


######################### test phase ##############################
# load test data
def LoadTestData(data_path):
    imgs = []
    center_shapes = []
    corner_shapes = []
    bboxes = []
    name = []
    padding_list = []
    angle_gt = []
    test_img_list = open(data_path + 'test_img_list').readlines()
    for i, img in enumerate(test_img_list):
        img = data_path + img.strip()
        image = cv2.imread(img.strip(), 0)
        padding_list.append([0, 0])
        assert image.shape[0] == 704
        assert image.shape[1] == 384
        imgs.append(image.astype(np.float))

        corner_shape = ReadCornerShape(img.split('.jpg')[0] + '_corner.pts', offsets=[0, 0])
        center_shape = GetCenterFromCorner(corner_shape)
        corner_vec, angle_gt1 = GenerateCornerVec(corner_shape)

        angle_gt.append(angle_gt1)
        center_shapes.append(center_shape)
        corner_shapes.append(corner_shape)

        bboxes.append(GenerateBBox(center_shapes[-1]))
        name.append(os.path.basename(img))

    angle_gt = np.array(angle_gt)

    center_shapes = np.array(center_shapes)
    bboxes = np.array(bboxes)
    corner_shapes = np.array(corner_shapes)
    padding_list = np.array(padding_list)

    return imgs, center_shapes, corner_shapes, bboxes, name, padding_list, angle_gt


######################### model ##################################
# save trained models
def SaveModels(model, mean_rshape_center, mean_rshape_corner, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump([model.state_dict(), mean_rshape_center, mean_rshape_corner], f)


# load models for test
def LoadModels(filename):
    with open(filename, 'rb') as f:
        model_dict, mean_rshape_center, mean_rshape_corner = pickle.load(f)
    return model_dict, mean_rshape_center, mean_rshape_corner
