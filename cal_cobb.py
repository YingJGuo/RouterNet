import numpy as np
import cv2
import copy 

def calculate_cobb(pr_landmarks,gt_landmarks,ori_image,img_names=None):
    pr_cobb_angles = []
    gt_cobb_angles = []
    pr_landmarks = copy.deepcopy(pr_landmarks)
    gt_landmarks = copy.deepcopy(gt_landmarks)
    ori_image = copy.deepcopy(ori_image)


    for i in range(len(pr_landmarks)):   #128

        if img_names is not None:
            pr_cobb_angles.append(cobb_angle_calc(pr_landmarks[i], ori_image[i],img_names[i]+'pre') )
            gt_cobb_angles.append(cobb_angle_calc(gt_landmarks[i], ori_image[i],img_names[i]+'gt') )
        else:
            pr_cobb_angles.append(cobb_angle_calc(pr_landmarks[i], ori_image[i]))
            gt_cobb_angles.append(cobb_angle_calc(gt_landmarks[i], ori_image[i]))


    pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
    gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)

    out_abs = abs(gt_cobb_angles - pr_cobb_angles)
    out_add = gt_cobb_angles + pr_cobb_angles

    term1 = np.sum(out_abs, axis=1)
    term2 = np.sum(out_add, axis=1)

    sin = np.sum((np.sin(out_abs/180*3.1415)), axis=1)
    cos = np.sum((np.cos(out_abs/180*3.1415)), axis=1)

    CMAE = np.mean(np.arctan(sin/cos)*180/3.1415)
    SMAPE = np.mean(term1 / term2 * 100)
    SMAPE_std = np.std(term1 / term2 * 100)
    SMAPE_95_min = SMAPE-1.96*SMAPE_std/np.sqrt(len(pr_landmarks))
    SMAPE_95_max = SMAPE+1.96*SMAPE_std/np.sqrt(len(pr_landmarks))

    MD = np.mean(term1) #MD
    ED = np.mean(np.sqrt(np.sum((np.square(gt_cobb_angles - pr_cobb_angles)),axis=1)))
    CD = np.mean(np.max(out_abs,axis=1))


    if img_names is not None:
        return MD,SMAPE,pr_cobb_angles,ED,CD,CMAE,SMAPE_95_min,SMAPE_95_max
    else:return MD,SMAPE,ED,CD,CMAE,SMAPE_95_min,SMAPE_95_max
    
def cobb_angle_calc(pts, image,name=None):
    pts = np.asarray(pts, np.float32)
    h,w = image.shape
    num_pts = pts.shape[0]
    vnum = num_pts//4-1

    mid_p_v = (pts[0::2,:]+pts[1::2,:])/2
    mid_p = []
    for i in range(0, num_pts, 4):
        pt1 = (pts[i,:]+pts[i+2,:])/2
        pt2 = (pts[i+1,:]+pts[i+3,:])/2
        mid_p.append(pt1)
        mid_p.append(pt2)
    mid_p = np.asarray(mid_p, np.float32)
    img =  np.stack((image,)*3, axis=-1)
    for pt in mid_p:
        cv2.circle(img,
                   (int(pt[0]), int(pt[1])),
                   2, (0,255,255), -1, 1)
    for pt1, pt2 in zip(mid_p[0::2,:], mid_p[1::2,:]):
        cv2.line(img,
                 (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])),
                 color=(0,255,255),
                 thickness=2, lineType=1)

    vec_m = mid_p[1::2,:]-mid_p[0::2,:]
    dot_v = np.matmul(vec_m, np.transpose(vec_m))
    mod_v = np.sqrt(np.sum(vec_m**2, axis=1))[:, np.newaxis]
    mod_v = np.matmul(mod_v, np.transpose(mod_v))
    
    cosine_angles = np.clip(dot_v/mod_v, a_min=0., a_max=1.)
    angles = np.arccos(cosine_angles)
    pos1 = np.argmax(angles, axis=1)
    maxt = np.amax(angles, axis=1)
    pos2 = np.argmax(maxt)
    cobb_angle1 = np.amax(maxt)   
    cobb_angle1 = cobb_angle1/np.pi*180
    flag_s = is_S(mid_p_v)
   
    if not flag_s:
        cobb_angle2 = angles[0, pos2]/np.pi*180
        cobb_angle3 = angles[vnum, pos1[pos2]]/np.pi*180
        
        cv2.line(img,
                 (int(mid_p[pos2 * 2, 0] ), int(mid_p[pos2 * 2, 1])),
                 (int(mid_p[pos2 * 2 + 1, 0]), int(mid_p[pos2 * 2 + 1, 1])),
                 color=(0, 255, 0), thickness=3, lineType=2)
        cv2.line(img,
                 (int(mid_p[pos1[pos2] * 2, 0]), int(mid_p[pos1[pos2] * 2, 1])),
                 (int(mid_p[pos1[pos2] * 2 + 1, 0]), int(mid_p[pos1[pos2] * 2 + 1, 1])),
                 color=(0, 255, 0), thickness=3, lineType=2)
    else:
        if (mid_p_v[pos2*2, 1]+mid_p_v[pos1[pos2]*2,1])<h:
            angle2 = angles[pos2,:(pos2+1)]
            cobb_angle2 = np.max(angle2)
            pos1_1 = np.argmax(angle2)
            cobb_angle2 = cobb_angle2/np.pi*180

            angle3 = angles[pos1[pos2], pos1[pos2]:(vnum+1)]
            cobb_angle3 = np.max(angle3)
            pos1_2 = np.argmax(angle3)
            cobb_angle3 = cobb_angle3/np.pi*180
            pos1_2 = pos1_2 + pos1[pos2]-1

            cv2.line(img,
                     (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1])),
                     (int(mid_p[pos1_1 * 2+1, 0]), int(mid_p[pos1_1 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=3, lineType=2)

            cv2.line(img,
                     (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1])),
                     (int(mid_p[pos1_2 * 2+1, 0]), int(mid_p[pos1_2 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=3, lineType=2)
        else:
            angle2 = angles[pos2,:(pos2+1)]
            cobb_angle2 = np.max(angle2)
            pos1_1 = np.argmax(angle2)
            cobb_angle2 = cobb_angle2/np.pi*180

            angle3 = angles[pos1_1, :(pos1_1+1)]
            cobb_angle3 = np.max(angle3)
            pos1_2 = np.argmax(angle3)
            cobb_angle3 = cobb_angle3/np.pi*180

            cv2.line(img,
                     (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1])),
                     (int(mid_p[pos1_1 * 2+1, 0]), int(mid_p[pos1_1 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=3, lineType=2)

            cv2.line(img,
                     (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1])),
                     (int(mid_p[pos1_2 * 2+1, 0]), int(mid_p[pos1_2 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=3, lineType=2)
    return [cobb_angle1, cobb_angle2, cobb_angle3]

def is_S(mid_p_v):
    ll = []
    num = mid_p_v.shape[0]
    for i in range(num-2): #32
        term1 = (mid_p_v[i, 1]-mid_p_v[num-1, 1])/(mid_p_v[0, 1]-mid_p_v[num-1, 1])
        term2 = (mid_p_v[i, 0]-mid_p_v[num-1, 0])/(mid_p_v[0, 0]-mid_p_v[num-1, 0])
        ll.append(term1-term2)
    ll = np.asarray(ll, np.float32)[:, np.newaxis]
    ll_pair = np.matmul(ll, np.transpose(ll))
    a = sum(sum(ll_pair))
    b = sum(sum(abs(ll_pair)))
    if abs(a-b)<1e-4:
        return False
    else:
        return True