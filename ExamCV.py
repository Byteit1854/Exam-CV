
##Импортируем нужные библиотеки и загружаем нужную нам фотографию

import cv2, numpy as np, itertools, os

src_path = "c:/Users/dshynggys/Downloads/Exam CV/image.jpg"
img = cv2.imread(src_path)
if img is None:
    raise RuntimeError("Image not found")

##Ищем на холсте наши фотографий для сшивания.

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

pieces = []
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    if w>20 and h>50:
        pieces.append(img[y:y+h, x:x+w])

##Спомошью SIFT определяем точки покоторым можно будет выровнять изображения

try:
    detector = cv2.SIFT_create()
    norm = cv2.NORM_L2
except Exception:
    detector = cv2.ORB_create(5000)
    norm = cv2.NORM_HAMMING

bf = cv2.BFMatcher(norm, crossCheck=False)

def kd(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return detector.detectAndCompute(g, None)

kps, desc = [], []
for p in pieces:
    k,d = kd(p)
    kps.append(k); desc.append(d)

def good_matches(d1, d2, ratio=0.75):
    if d1 is None or d2 is None or len(d1)<4 or len(d2)<4:
        return 0
    m = bf.knnMatch(d1, d2, k=2)
    return sum(1 for a,b in m if a.distance < ratio*b.distance)


##С центра сшиваем на основе хороших вариантов для сшивания.

n = len(pieces)
M = np.zeros((n,n), dtype=int)
for i,j in itertools.combinations(range(n),2):
    ab = good_matches(desc[i], desc[j])
    ba = good_matches(desc[j], desc[i])
    score = min(ab, ba)  
    M[i,j] = M[j,i] = score

##Выбираем центр и удаляем случайные совпадения с помощью RANSAC.

totals = M.sum(1)
ref_index = int(np.argmax(totals))

def ransac_homography_from_kp(i, j):
    d1, d2 = desc[i], desc[j]
    if d1 is None or d2 is None:
        return None
    m = bf.knnMatch(d1, d2, k=2)
    good = [x for x,y in m if x.distance < 0.75*y.distance]
    if len(good) < 6: 
        return None
    pts1 = np.float32([kps[i][g.queryIdx].pt for g in good]).reshape(-1,1,2)
    pts2 = np.float32([kps[j][g.trainIdx].pt for g in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
    return H

def warp_blend(base, img, H):
    h1,w1 = base.shape[:2]
    h2,w2 = img.shape[:2]
    corners = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(corners, H)
    allc = np.concatenate((warped, np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)))
    (xmin,ymin) = np.int32(allc.min(0).ravel() - 1)
    (xmax,ymax) = np.int32(allc.max(0).ravel() + 1)
    tx,ty = -xmin,-ymin
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
    W,Hh = xmax-xmin, ymax-ymin
    warped_img = cv2.warpPerspective(img, T@H, (W,Hh))
    canvas = np.zeros_like(warped_img)
    canvas[ty:ty+h1, tx:tx+w1] = base

    mask_c = (canvas.sum(2)>0).astype(np.float32)
    mask_w = (warped_img.sum(2)>0).astype(np.float32)
    blended = canvas.copy()
    overlap = mask_c*mask_w
    ys,xs = np.where(overlap>0)
    for y,x in zip(ys,xs):
        blended[y,x] = (0.5*canvas[y,x] + 0.5*warped_img[y,x]).astype(np.uint8)
    blended[(mask_c==0)&(mask_w==1)] = warped_img[(mask_c==0)&(mask_w==1)]
    return blended

##На основе балла соответсвия сшиваем фотографий вместе и вырезаем панораму.

used = {ref_index}
mosaic = pieces[ref_index]

while len(used) < n:
    best_idx, best_score = None, -1
    for j in range(n):
            if j in used: 
                continue
            score = max(M[j,k] for k in used)
            if score > best_score:
                best_score = score
                best_idx = j

    j = best_idx
    if j is None:
        break
    H = ransac_homography_from_kp(ref_index, j)
    nxt = pieces[j]
    used.add(j)
    if H is None:
        h = max(mosaic.shape[0], nxt.shape[0])
        c = np.zeros((h, mosaic.shape[1]+nxt.shape[1],3),dtype=np.uint8)
        c[:mosaic.shape[0], :mosaic.shape[1]] = mosaic
        c[:nxt.shape[0], mosaic.shape[1]:] = nxt
        mosaic = c
    else:
        mosaic = warp_blend(mosaic, nxt, H)

gm = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
coords = cv2.findNonZero(gm)
if coords is not None:
    x,y,w,h = cv2.boundingRect(coords)
    mosaic = mosaic[y:y+h, x:x+w]


##Вывод в виде фотографий.


out_path = "c:/Users/dshynggys/Downloads/Exam CV/panorama_central_bidirectional.jpg"
cv2.imwrite(out_path, mosaic)
out_path

