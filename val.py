import numpy as np

from utils import *
font = cv2.FONT_HERSHEY_SIMPLEX
import os

if __name__=='__main__':

    # dataset for training and validation
    # 481 images for training are in folder ./MICCAI_128/refactory/training, image list is in ./MICCAI_128/train_img_list
    # 128 images for validation are in folder ./MICCAI_128/refactory/val, image list is in ./MICCAI_128/test_img_list
    dataset = 'MICCAI_128'

    imgs, gt_center_shapes,gt_corner_shapes, bboxes,name,padding_list,gt_angle=LoadTestData('./dataset/'+dataset+'/')
    nets, mean_rshape_center, mean_rshape_corner = LoadModels('./model/Jul25-1445_root_8/best_corner')
    test_mean_center = np.array([Shape2Absolute(mean_rshape_center, bboxes[i]) for i in range(len(imgs))])
    test_mean_corner_off = []

    for i in range(len(imgs)):
        test_mean_corner = Shape2Absolute(mean_rshape_corner, bboxes[i])
        test_mean_corner_offset,_ = GenerateCornerVec(test_mean_corner)
        test_mean_corner_off.append(test_mean_corner_offset)
    test_mean_corner_off = np.array(test_mean_corner_off)
    
    colour = [(255, 0, 150),(0, 150, 255),(266, 150, 0),(0, 0, 255)]
    for i in range(len(imgs)):
        img = np.zeros_like(imgs[i]).astype(np.uint8)
        img = 255-img
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for k in range(args.param_landmark_num):
            pd_s = test_mean_center[i, k].astype(int)
            if k==8:
                cv2.circle(img, (int(pd_s[0]), int(pd_s[1])),  4, (0, 0, 255), -1)
            else:
                cv2.circle(img, (int(pd_s[0]), int(pd_s[1])),  4, (0, 255, 0), -1)
            for pt_num in range(4):
                pd_corner = test_mean_corner_off[i,4*k+pt_num]
                cv2.circle(img, (int(pd_corner[0]), int(pd_corner[1])), 4, colour[pt_num], 2)
        cv2.imwrite("./mean_shape/{}_.png".format(name[i]),img)
    
    if not os.path.exists('./results_final/result_{}'.format(dataset)):
        os.makedirs('./results_final/result_{}'.format(dataset))
    if not os.path.exists('./results_final/result_{}/prediction_center'.format(dataset)):
        os.makedirs('./results_final/result_{}/prediction_center'.format(dataset))

    center_errors = []
    corner_errors = []
    corner_or20s = []
    corner_or40s = []

    t1=time.time()
    updates_center,update_corner=TestGlobalRegression(imgs, test_mean_center, test_mean_corner_off,nets) #, pre_angle
    print('global regression:{:.4f}s'.format(time.time()-t1))

    #center points
    errors = []
    for j in range(len(updates_center)):
        img = imgs[j].copy().astype(np.uint8)
        img_copy = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_copy = np.sum(img_copy, axis=0)
        for m in range(300):
            if img_copy[m]>0:
                left = m
                break
        dist = 0.0
        height = float(img.shape[0])
        width = float(img.shape[1]-2*left)
        for k in range(args.param_landmark_num):
            gt_s = gt_center_shapes[j, k].astype(int)
            pd_s = updates_center[j, k].astype(int)
            dist += ((gt_s[0] - pd_s[0]) / width) * (gt_s[0] - pd_s[0]) / width
            dist += ((gt_s[1] - pd_s[1]) / height) * (gt_s[1] - pd_s[1]) / height
            cv2.circle(img, (pd_s[0], pd_s[1]), 6, (0, 255, 0), -1)
            cv2.circle(img, (gt_s[0], gt_s[1]), 4, (255, 0, 0), 2)
        errors.append(dist / args.param_landmark_num)
        cv2.putText(img, 'MSE:' + str(round(errors[-1], 5)), (190, 20), font, 0.5, (0, 255, 255), 1)
        cv2.imwrite("./results_final/result_{}/{}_center.png".format(dataset,name[j]), img)

    #corner and center points
    imggg=[]
    for j in range(len(test_mean_center)):
        img = imgs[j].copy().astype(np.uint8)
        img_gt = imgs[j].copy().astype(np.uint8)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2RGB)

        img_copy = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_copy = np.sum(img_copy,axis = 0)
        imggg.append(img)
        for m in range(300):
            if img_copy[m]>0:
                left = m
                break
        height = float(img.shape[0]-0)
        width = float(img.shape[1]-2*left)
        dist_center = 0.0
        dist_corner = 0.0
        corner_or20 = 0.0
        corner_or40 = 0.0
        with open('./results_final/result_{}/prediction_center/'.format(dataset) + name[j] + '.pts', 'w') as fp:
            for k in range(args.param_landmark_num):
                gt_s = gt_center_shapes[j, k].astype(int)
                pd_s = updates_center[j, k].astype(int)
                dist_center += ((gt_s[0] - pd_s[0]) / width) * (gt_s[0] - pd_s[0]) / width
                dist_center += ((gt_s[1] - pd_s[1]) / height) * (gt_s[1] - pd_s[1]) / height
                for pt_num in range(4):
                    pd_corner = update_corner[j,4*k+pt_num]
                    gt_corner = gt_corner_shapes[j,4*k+pt_num]
                    if (np.sqrt(np.square(gt_corner[0]-pd_corner[0])+np.square(gt_corner[1]-pd_corner[1])))>20:
                        corner_or20 +=1
                    if (np.sqrt(np.square(gt_corner[0] - pd_corner[0]) + np.square(gt_corner[1] - pd_corner[1]))) > 40:
                        corner_or40 +=1
                    dist_corner += ((gt_corner[0] - pd_corner[0]) / width) * (gt_corner[0] - pd_corner[0]) / width
                    dist_corner += ((gt_corner[1] - pd_corner[1]) / height) * (gt_corner[1] - pd_corner[1]) / height
                    cv2.circle(img, (int(pd_corner[0]), int(pd_corner[1])), 4, colour[pt_num], 2)
                    cv2.circle(img_gt, (int(gt_corner[0]), int(gt_corner[1])), 3, (0, 255, 0), -1)
                for _ in range(4):
                    fp.write(str(pd_s[0]) + ' ' + str(pd_s[1]) + '\n')
            center_errors.append(dist_center / args.param_landmark_num)
            corner_errors.append(dist_corner / (args.param_landmark_num*4))
            corner_or20s.append(corner_or20 / (args.param_landmark_num*4) * 100)
            corner_or40s.append(corner_or40 / (args.param_landmark_num * 4) * 100)

            cv2.imwrite("./results_final/result_{}/{}_corner.png".format(dataset,name[j]),img)
            for gt_pt in gt_corner_shapes[j]:
                cv2.circle(img, (int(gt_pt[0]), int(gt_pt[1])), 3, (0, 255, 0), -1)
            cv2.imwrite("./results_final/result_{}/{}_add_gt.png".format(dataset,name[j]),img)
            cv2.imwrite("./results_final/result_{}/{}_corner_gt.png".format(dataset,name[j]),img_gt)

    
    MD,SMAPE,pred_cobb,ED,CD,CMAE,SMAPE_Min,SMAPE_Max = calculate_cobb(update_corner,gt_corner_shapes,imgs,name)
    print('Cobb-> CMAE = {}, ED = {}, MD = {},\n CD = {}, SMAPE = {}[{},{}]'.format(CMAE,ED,MD,CD,SMAPE,SMAPE_Min,SMAPE_Max))
    print('Stage', '9', 'center NMSE_mean:{}, center NMSE_std:{}'.format(np.mean(np.array(center_errors)),np.std(np.array(center_errors))))
    print('Stage', '9', 'corner NMSE_mean:{}, corner NMSE_std:{}'.format(np.mean(np.array(corner_errors)),np.std(np.array(corner_errors))))
    print('Stage', '9', 'corner or20_mean:{}, corner or20_std:{}'.format(np.mean(np.array(corner_or20s)),np.std(np.array(corner_or20s))))
    print('Stage', '9', 'corner or40_mean:{}, corner or40_std:{}'.format(np.mean(np.array(corner_or40s)),np.std(np.array(corner_or40s))))

