# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:19:37
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:34:26



from common import *

def random_scale(image, mask, size_=-1, center_p=1):
    H, W, C = image.shape
    _, _, C_mask = mask.shape
    if size_ == -1:
        sizes = [0.85, 1, 1.15]
        size_ = random.choice(sizes)
    if size_ == 1:
        return image, mask

    new_H, new_W = round(size_*H), round(size_*W)
    image = cv2.resize(image, (new_W, new_H))
    mask = cv2.resize(mask, (new_W, new_H))

    y_start = abs(W - new_W) // 2
    x_start = abs(H - new_H) // 2
    if random.random() <= (1 - center_p):
        y_start = random.randint(0, abs(W - new_W))
        x_start = random.randint(0, abs(H - new_H))
    if size_ > 1:
        return image[x_start:x_start + H, y_start:y_start + W, :], mask[x_start:x_start + H,
                                                                        y_start:y_start + W]
    else:
        image_zero = np.zeros((H, W, C))
        image_zero[x_start:x_start + new_H, y_start:y_start + new_W, :] = image
        mask_zero = np.zeros((H, W, C_mask))
        mask_zero[x_start:x_start + new_H, y_start:y_start + new_W, :] = mask
        return image_zero.astype(np.uint8), mask_zero


def random_erase(image, mask, p=0.5):
    if random.random() < p:
        width, height, _ = image.shape
        x = random.randint(0, width)
        y = random.randint(0, height)
        b_w = random.randint(5, 15)
        b_h = random.randint(5, 15)
        image[x:x + b_w, y:y + b_h, :] = 0
        mask[x:x + b_w, y:y + b_h, :] = 0
    return image, mask


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def random_angle_rotate(image, mask, angles=(-30, 30)):
    angle = random.randint(0, angles[1] - angles[0]) + angles[0]
    image = rotate(image, angle)
    mask = rotate(mask, angle)
    return image, mask


def do_gaussian_noise(image, sigma=0.5):
    gray = image.astype(np.float32) / 255
    H, W = gray.shape

    noise = np.random.normal(0, sigma, (H, W))
    noisy = gray + noise

    noisy = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
    return noisy


def do_color_shift(image, alpha0=0, alpha1=0, alpha2=0):
    image = image.astype(np.float32) + np.array([alpha0, alpha1, alpha2]) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_hue_shift(image, alpha=0):
    h = int(alpha * 180)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def do_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32) / 255
    H, W = gray.shape

    noise = sigma * np.random.randn(H, W)
    noisy = gray + gray * noise

    noisy = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
    lab = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

def do_random_log_contrast(image):
    gain = np.random.uniform(0.70,1.30,1)
    inverse = np.random.choice(2,1)

    image = image.astype(np.float32)/255
    if inverse==0:
        image = gain*np.log(image+1)
    else:
        image = gain*(2**image-1)

    image = np.clip(image*255,0,255).astype(np.uint8)
    return image

## illumination ====================================================================================

def do_brightness_shift(image, alpha=0.125):
    image = image.astype(np.float32)
    image = image + alpha * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_brightness_multiply(image, alpha=1):
    image = image.astype(np.float32)
    image = alpha * image
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_contrast(image, alpha=1.0):
    image = image.astype(np.float32)
    gray = image * np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    image = alpha * image + gray
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)  # apply gamma correction using the lookup table


def do_custom_process1(image, gamma=2.0, alpha=0.8, beta=2.0):
    image1 = image.astype(np.float32)
    image1 = image1 ** (gamma)
    image1 = image1 / image1.max() * 255

    image2 = (alpha) * image1 + (1 - alpha) * image
    image2 = np.clip(beta * image2, 0, 255).astype(np.uint8)

    image = image2
    return image


def do_clahe(image, clip=2, grid=16):
    grid = int(grid)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)
    lab = cv2.merge((gray, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image
