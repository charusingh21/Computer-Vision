# import necessary libraries
from os.path import join
import cv2
import numpy as np

# function to read the intrensic parameters
def intrinsic_reader(txt_file):
    with open(txt_file) as f:
        lines = f.readlines()
    return np.array(
        [l.strip().split(' ') for l in lines],
        dtype=np.float32
    )

# function for scaling the intrensic parameters
def rescale_intrinsic():
        # scale focal length and principal points wrt image resizeing
        if downscale > 1:
            K = K_orig.copy()
            K[0, 0] /= float(downscale)
            K[1, 1] /= float(downscale)
            K[0, 2] /= float(downscale)
            K[1, 2] /= float(downscale)
            intrinsic = K
        else:
            K = intrinsic = K_orig.copy()

# function for saving 3D points in object file
def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    """
    Saves 3d points which can be read in meshlab
    """
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        if mesh_f is not None:
            for f in mesh_f+1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', filepath)

# function for visualising intermediate outputs
def visualize_matches(img1, img2, kp1, kp2, good, mask=None, save_path=None):
    """
    The function visualizes the sift matches.
    img1, img2 are two images whose matches we need
    to compare
    kp1, kp2 are keypoints in img1, img2. In this case,
    it would be sift keypoints
    good: is a list of matches which pass the ratio test
    mask: is an output array with inlier_match as 1,
    outliers as 0.
    save_path: destination to save the visualization image
    """
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)
    if mask is not None:
        if not isinstance(mask, list):
            matchesMask = mask.ravel().tolist()
        else:
            matchesMask = mask
        draw_params['matchesMask'] = matchesMask
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, good, None, **draw_params)
    cv2.imwrite(save_path, img_matches)

# Function for drawing lines when doing epi polar computation 
def drawlines(img1, img2, lines, pts1, pts2, line_num=None):
    """
    Draw line connecting points in two images.
    """
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        r, c = img1.shape
    else:  # 3
        r, c, _ = img1.shape
    if line_num is not None:
        draw_list = np.random.choice(
            pts1.shape[0], line_num, replace=False)
    else:
        draw_list = np.arange(pts1.shape[0])
    for idx, (r, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
        if idx not in list(draw_list):
            continue
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 2)
        img1 = cv2.circle(img1, tuple(pt1.ravel()), 5, color, 3)
        img2 = cv2.circle(img2, tuple(pt2.ravel()), 5, color, 3)
    return img1, img2

# function for computing epi polar lines
def visualize_epipolar_lines(img1, img2, p1, p2, E, save_path):
    """
    This function visualizes the epipolar lines
    img1, img2: are the two images
    p1, p2: are the good keypoints
    E: Essential matrix
    save_path: destination to save the visualization image
    """
    # get fundamental matrix
    F, mask_fdm = cv2.findFundamentalMat(p1, p2, cv2.RANSAC)
    p1_selected = p1[mask_fdm.ravel() == 1]
    p2_selected = p2[mask_fdm.ravel() == 1]

    # draw lines
    lines1 = cv2.computeCorrespondEpilines(
        p2_selected.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    img5, _ = drawlines(
        img1, img2, lines1, p1_selected, p2_selected, 100)

    lines2 = cv2.computeCorrespondEpilines(
        p1_selected.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    img3, _ = drawlines(
        img2, img1, lines2, p2_selected, p1_selected, 100)
    canvas = np.concatenate((img5, img3), axis=1)
    cv2.imwrite(save_path, canvas)

# initialising the parameters
downscale = 1
img_1 = 'C:/Users/ravle_ty3y2wl/OneDrive/Desktop/CV/HW 3/HW3_Data/rdimage.000.ppm'
img_2 = 'C:/Users/ravle_ty3y2wl/OneDrive/Desktop/CV/HW 3/HW3_Data/rdimage.002.ppm'
intrinsic = intrinsic_reader('C:/Users/ravle_ty3y2wl/OneDrive/Desktop/CV/HW 3/HW3_Data/intrinsics.txt')
filepath = "C:/Users/ravle_ty3y2wl/OneDrive/Desktop/CV/HW 3/output/"
K_orig = intrinsic_orig = intrinsic.copy()
downscale = downscale
rescale_intrinsic()

# load the 2 images and scale them if needed
this_img_1 = cv2.imread(img_1)
if downscale > 1:
    this_img_1 = cv2.resize(this_img_1, (0, 0),
                          fx=1/float(downscale),
                          fy=1/float(downscale),
                          interpolation=cv2.INTER_LINEAR)
    
this_img_2 = cv2.imread(img_2)
if downscale > 1:
    this_img_2 = cv2.resize(this_img_2, (0, 0),
                          fx=1/float(downscale),
                          fy=1/float(downscale),
                          interpolation=cv2.INTER_LINEAR)

# compute discriptors of sift
sift = cv2.xfeatures2d.SIFT_create()
kp_1, des_1 = sift.detectAndCompute(this_img_1, None)
kp_2, des_2 = sift.detectAndCompute(this_img_2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

# Match sift features using flann matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_1, des_2, k=2)

# https://www.geeksforgeeks.org/python-opencv-epipolar-geometry/
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
# https://www.programcreek.com/python/example/89342/cv2.drawMatchesKnn

# Ratio test
goodMatches = [] 
p1 = [] 
p2 = []    
for m, n in matches:      
    if m.distance < 0.75 * n.distance:          
        goodMatches.append(m) 

goodMatches =sorted(goodMatches, key = lambda x:x.distance)
p1 = np.float32([ kp_1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
p2 = np.float32([ kp_2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

visualize_matches(this_img_1, this_img_2, kp_1, kp_2, goodMatches, save_path=join(filepath,'sift.png'))

# computing Eseential matrix        
E, mask = cv2.findEssentialMat(p1, p2, method = cv2.RANSAC, prob = 0.999, threshold = 1.0) 
print("Essential Matrix:", E)
matchesMask = mask.ravel().tolist()
visualize_matches(this_img_1, this_img_2, kp_1, kp_2, goodMatches, mask= matchesMask, save_path=join(filepath, 'inlier_match.png'))
visualize_epipolar_lines(this_img_1, this_img_2, p1, p2, E, save_path=join(filepath,'epipolar_lines_1_2.png'))

# compute pose
[_, R, t, _] = cv2.recoverPose(E, p1, p2, K_orig)
print("Rotation Matrix:", R)
print("Triangulation Matrix:", t)

# compute projection matrix 
proj_2 = np.hstack((R, t))
proj_1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

proj_pt1 = np.dot(K_orig, proj_1)
proj_pt2 = np.dot(K_orig, proj_2)
print("Projection Matrix 1:", proj_pt1)
print("Projection Matrix 2:", proj_pt2)

# Normailsing the image co ordinates
p1 = p1[np.asarray(matchesMask) == 1, :, :]
p2 = p2[np.asarray(matchesMask) == 1, :, :]
p1_un = cv2.undistortPoints(p1, K_orig, None)
p2_un = cv2.undistortPoints(p2, K_orig, None)
p1_un = np.squeeze(p1_un)
p2_un = np.squeeze(p2_un)

# triangulate points this requires points in normalized coordinate
point_4d_hom = cv2.triangulatePoints(proj_pt1, proj_pt2, p1_un.T, p2_un.T)
point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_3d = point_3d[:3, :].T
#print('Output', point_3d)

filepath = "C:/Users/ravle_ty3y2wl/OneDrive/Desktop/CV/HW 3/output/output.obj"  
write_simple_obj(point_3d, None, filepath)