import numpy as np
import torch 
import cv2
import random
from PIL import Image


def gaussian_blure(img, intens = 5):
    """
    高斯模糊
    :param image_path:
    :intens 5,10,15,20
    :return:
    """
    img = np.array(img).astype(np.uint8)
    result = cv2.GaussianBlur(img, (0, 0), intens)
    result = Image.fromarray(result)
    return result

def random_mask(mask):
    h,w = mask.shape[0], mask.shape[1]
    mask_black = np.zeros_like(mask)
    box_w = random.uniform(0.4, 0.9) * w
    box_h = random.uniform(0.4, 0.9) * h
    box_w = int(box_w)
    box_h = int(box_h)
    y1 = random.randint(0, h - box_h)
    y2 = y1 + box_h
    x1 = random.randint(0, w - box_w)
    x2 = x1 + box_w
    mask_black[y1:y2,x1:x2] = 1
    mask_black = mask_black.astype(np.uint8)
    return mask_black

'''
def random_mask_grid(mask, p=0.50):
    # 创建一个 h x w 的全零数组，作为初始掩膜
    h,w = mask.shape[0],mask.shape[1]
    mask = np.zeros((h, w), dtype=np.uint8)
    n = random.choice([3,4,5,6,7,8,9,10])
    
    # 计算小块的大小
    block_h = h // n
    block_w = w // n
    
    # 在每个小块中以概率 p 设置为 1
    for i in range(n):
        for j in range(n):
            if np.random.rand() < p:
                mask[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w] = 1
    return mask
'''

def get_SIFT(image):
    orb = cv2.ORB_create(nfeatures=200, edgeThreshold=50)
    keypoint, descriptor = orb.detectAndCompute(image, None)
    coordinates = [(int(kp.pt[1]), int(kp.pt[0])) for kp in keypoint]
    return coordinates


'''
def random_mask_grid(mask, points_list, p=0.0):
    # 创建一个 h x w 的全零数组，作为初始掩膜
    h, w = mask.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    n = random.choice([3,4,5,6,7,8,9,10])
    
    # 计算小块的大小
    block_h = h // n
    block_w = w // n
    
    # 统计每个小块内的点个数
    block_counts = np.zeros((n, n), dtype=np.int32)
    for point in points_list:
        y, x = point
        i = min(y // block_h, n-1)
        j = min(x // block_w, n-1)
        block_counts[i, j] += 1
    
    # 找出包含点最多的前5个小块
    top5_blocks = np.argpartition(-block_counts.flatten(), 5)[:5]
    
    # 将这些小块对应的像素设为1
    for idx in top5_blocks:
        i, j = divmod(idx, n)
        mask[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w] = 1
    
    # 在其他小块中按照概率p设置为1
    for i in range(n):
        for j in range(n):
            if (i*n + j) not in top5_blocks and np.random.rand() < p:
                mask[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w] = 1
    
    return mask
'''

def random_mask_grid(mask, points_list, p=0.50, top5_p=0.70, other_p=0.30):
    # 创建一个 h x w 的全零数组，作为初始掩膜
    h, w = mask.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    n = random.choice([3,4,5,6,7,8,9,10])
    
    # 计算小块的大小
    block_h = h // n
    block_w = w // n
    
    # 统计每个小块内的点个数
    block_counts = np.zeros((n, n), dtype=np.int32)
    for point in points_list:
        y, x = point
        i = min(y // block_h, n-1)
        j = min(x // block_w, n-1)
        block_counts[i, j] += 1
    
    # 找出包含点最多的前5个小块
    top5_blocks = np.argpartition(-block_counts.flatten(), 5)[:5]
    
    # 将这些小块对应的像素设为1
    for idx in top5_blocks:
        i, j = divmod(idx, n)
        if np.random.rand() < top5_p:
            mask[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w] = 1
    
    # 在其他小块中按照概率p设置为1
    for i in range(n):
        for j in range(n):
            if (i*n + j) not in top5_blocks and np.random.rand() < other_p:
                mask[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w] = 1
    
    return mask

def random_perspective_transform(image, intensity):
    """
    对图像进行随机透视变换

    参数:
    image: 要进行变换的输入图像
    intensity: 变换的强度,范围从0到1,值越大,变换越明显

    返回值:
    变换后的图像
    """
    height, width = image.shape[:2]

    # 生成随机透视变换的四个目标点
    x_offset = width * 0.4 * intensity
    y_offset = height * 0.4 * intensity
    dst_points = np.float32([[random.uniform(-x_offset, x_offset), random.uniform(-y_offset, y_offset)],
                             [width - random.uniform(-x_offset, x_offset), random.uniform(-y_offset, y_offset)],
                             [random.uniform(-x_offset, x_offset), height - random.uniform(-y_offset, y_offset)],
                             [width - random.uniform(-x_offset, x_offset), height - random.uniform(-y_offset, y_offset)]])

    # 对应的源点是图像的四个角
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 进行透视变换
    transformed_image = cv2.warpPerspective(image, M, (width, height))
    mask = np.ones_like(transformed_image)
    transformed_mask = cv2.warpPerspective(mask, M, (width, height))> 0.5

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    transformed_mask = cv2.erode(transformed_mask.astype(np.uint8), kernel, iterations=1).astype(np.uint8)

    white_back = np.ones_like(transformed_image) * 255
    transformed_image = transformed_image * transformed_mask + white_back * (1-transformed_mask)
    return transformed_image




def mask_score(mask):
    '''Scoring the mask according to connectivity.'''
    mask = mask.astype(np.uint8)
    if mask.sum() < 10:
        return 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    conc_score = np.max(cnt_area) / sum(cnt_area)
    return conc_score


def sobel(img, mask, thresh = 50):
    '''Calculating the high-frequency map.'''
    H,W = img.shape[0], img.shape[1]
    img = cv2.resize(img,(256,256))
    mask = (cv2.resize(mask,(256,256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 2)
    
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)
    sobel_X = cv2.convertScaleAbs(sobelx)
    sobel_Y = cv2.convertScaleAbs(sobely)
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1) * mask    
    
    scharr[scharr < thresh] = 0.0
    scharr = np.stack([scharr,scharr,scharr],-1)
    scharr = (scharr.astype(np.float32)/255 * img.astype(np.float32) ).astype(np.uint8)
    scharr = cv2.resize(scharr,(W,H))
    return scharr


def resize_and_pad(image, box):
    '''Fitting an image to the box region while keeping the aspect ratio.'''
    y1,y2,x1,x2 = box
    H,W = y2-y1, x2-x1
    h,w =  image.shape[0], image.shape[1]
    r_box = W / H 
    r_image = w / h
    if r_box >= r_image:
        h_target = H
        w_target = int(w * H / h) 
        image = cv2.resize(image, (w_target, h_target))

        w1 = (W - w_target) // 2
        w2 = W - w_target - w1
        pad_param = ((0,0),(w1,w2),(0,0))
        image = np.pad(image, pad_param, 'constant', constant_values=255)
    else:
        w_target = W 
        h_target = int(h * W / w)
        image = cv2.resize(image, (w_target, h_target))

        h1 = (H-h_target) // 2 
        h2 = H - h_target - h1
        pad_param =((h1,h2),(0,0),(0,0))
        image = np.pad(image, pad_param, 'constant', constant_values=255)
    return image



def expand_image_mask(image, mask, ratio=1.4, random = False):
    # expand image and mask 
    # pad image with 255 
    # pad mask with 0
    h,w = image.shape[0], image.shape[1]
    H,W = int(h * ratio), int(w * ratio) 
    if random:
        h1 = np.random.randint(0, int(H - h))
        w1 = np.random.randint(0, int(W - w)) 
    else:
        h1 = int((H - h) // 2)
        w1 = int((W -w) // 2)
    h2 = H - h - h1
    w2 = W -w - w1
    pad_param_image = ((h1,h2),(w1,w2),(0,0))
    pad_param_mask = ((h1,h2),(w1,w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=255)
    mask = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image, mask


def resize_box(yyxx, H,W,h,w):
    y1,y2,x1,x2 = yyxx
    y1,y2 = int(y1/H * h), int(y2/H * h)
    x1,x2 = int(x1/W * w), int(x2/W * w)
    y1,y2 = min(y1,h), min(y2,h)
    x1,x2 = min(x1,w), min(x2,w)
    return (y1,y2,x1,x2)


def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]

    if mask.sum() < 10:
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2)


def expand_bbox(mask,yyxx,ratio=[1.2,2.0], min_crop=0):
    y1,y2,x1,x2 = yyxx
    ratio = np.random.randint( ratio[0] * 10,  ratio[1] * 10 ) / 10
    H,W = mask.shape[0], mask.shape[1]
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def box2squre(image, box):
    H,W = image.shape[0], image.shape[1]
    y1,y2,x1,x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h,w = y2-y1, x2-x1

    if h >= w:
        x1 = cx - h//2
        x2 = cx + h//2
    else:
        y1 = cy - w//2
        y2 = cy + w//2
    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def pad_to_square(image, pad_value = 255, random = False):
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0,padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if H > W:
        pad_param = ((0,0),(padd_1,padd_2),(0,0))
    else:
        pad_param = ((padd_1,padd_2),(0,0),(0,0))

    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
    return image



def box_in_box(small_box, big_box):
    y1,y2,x1,x2 = small_box
    y1_b, _, x1_b, _ = big_box
    y1,y2,x1,x2 = y1 - y1_b ,y2 - y1_b, x1 - x1_b ,x2 - x1_b
    return (y1,y2,x1,x2 )



def shuffle_image(image, N):
    height, width = image.shape[:2]
    
    block_height = height // N
    block_width = width // N
    blocks = []
    
    for i in range(N):
        for j in range(N):
            block = image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]
            blocks.append(block)
    
    np.random.shuffle(blocks)
    shuffled_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(N):
        for j in range(N):
            shuffled_image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = blocks[i*N+j]
    return shuffled_image


def get_mosaic_mask(image, fg_mask, N=16, ratio = 0.5):
    ids = [i for i in range(N * N)]
    masked_number = int(N * N * ratio)
    masked_id = np.random.choice(ids, masked_number, replace=False)
    

    
    height, width = image.shape[:2]
    mask = np.ones((height, width))
    
    block_height = height // N
    block_width = width // N
    
    b_id = 0
    for i in range(N):
        for j in range(N):
            if b_id in masked_id:
                mask[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = mask[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] * 0
            b_id += 1
    mask = mask * fg_mask
    mask3 = np.stack([mask,mask,mask],-1).copy().astype(np.uint8)
    noise = q_x(image)
    noise_mask = image * mask3 + noise * (1-mask3)
    return noise_mask

def extract_canney_noise(image, mask, dilate=True):
    h,w = image.shape[0],image.shape[1]
    mask = cv2.resize(mask.astype(np.uint8),(w,h)) > 0.5
    kernel = np.ones((8, 8), dtype=np.uint8)
    mask =  cv2.erode(mask.astype(np.uint8), kernel, 10)

    canny = cv2.Canny(image, 50,100) * mask
    kernel = np.ones((8, 8), dtype=np.uint8)
    mask = (cv2.dilate(canny, kernel, 5) > 128).astype(np.uint8)
    mask = np.stack([mask,mask,mask],-1)

    pure_noise = q_x(image, t=1) * 0 + 255
    canny_noise = mask * image + (1-mask) * pure_noise
    return canny_noise


def get_random_structure(size):
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)


def select_max_region(mask):
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels==max_idx+1, 1, 0)

    return max_region.astype(np.uint8)



def perturb_mask(gt, min_iou = 0.3,  max_iou = 0.99):
    iou_target = np.random.uniform(min_iou, max_iou)
    h, w = gt.shape
    gt = gt.astype(np.uint8)
    seg = gt.copy()
    
    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.1:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            # Dilate/erode
            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])
            
            seg = np.logical_or(seg, gt).astype(np.uint8)
            #seg = select_max_region(seg) 

        if compute_iou(seg, gt) < iou_target:
            break
    seg = select_max_region(seg.astype(np.uint8)) 
    return seg.astype(np.uint8)


def q_x(x_0,t=65):
    '''Adding noise for and given image.'''
    x_0 = torch.from_numpy(x_0).float() / 127.5 - 1
    num_steps = 100
    
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

    alphas = 1-betas
    alphas_prod = torch.cumprod(alphas,0)
    
    alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise).numpy()  * 127.5 + 127.5 


def extract_target_boundary(img, target_mask):
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)

    # sobel-x
    sobel_X = cv2.convertScaleAbs(sobelx)
    # sobel-y
    sobel_Y = cv2.convertScaleAbs(sobely)
    # sobel-xy
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1).astype(np.float32)/255
    scharr = scharr *  target_mask.astype(np.float32)
    return scharr