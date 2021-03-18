'''
CS 677 Fall 2019 - HW 3
SIFT Matching and SFM

note:
Since SIFT is patented, you might need to choose (reinstall) appropriate opencv version:
Try (in your virtual env):
pip install opencv-contrib-python==3.4.2.16
https://stackoverflow.com/questions/52305578/sift-cv2-xfeatures2d-sift-create-not-working-even-though-have-contrib-instal
'''

import os
import cv2
import numpy as np
from glob import glob
from os.path import join
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------


class SFMSolver(object):
   

    def __init__(self, img_pattern, intrinsic, output_dir, downscale=1):
        """
        img_pattern: regex pattern used by glob to read the files
        instrinsic:
        """
        self.img_pattern = img_pattern
        self.K_orig = self.intrinsic_orig = intrinsic.copy()
        self.output_dir = output_dir
        self.downscale = downscale
        self.rescale_intrinsic()

    def rescale_intrinsic(self):
        """
        if we downscale the image, the intrinsic matrix
        also needs to be changed.
        """
        # scale focal length and principal points wrt image resizeing
        if self.downscale > 1:
            self.K = self.K_orig.copy()
            self.K[0, 0] /= float(self.downscale)
            self.K[1, 1] /= float(self.downscale)
            self.K[0, 2] /= float(self.downscale)
            self.K[1, 2] /= float(self.downscale)
            self.intrinsic = self.K
        else:
            self.K = self.intrinsic = self.K_orig.copy()

    def load_images(self):
        """
        Loads a set of images to self.imgs list
        """
        self.img_paths = sorted(glob(self.img_pattern))
        self.imgs = []
        for idx, this_path in enumerate(self.img_paths):
            try:
                this_img = cv2.imread(this_path)
                if self.downscale > 1:
                    this_img = cv2.resize(this_img, (0, 0),
                                          fx=1/float(self.downscale),
                                          fy=1/float(self.downscale),
                                          interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                print("error loading img: %s" % (this_path))
            if this_img is not None:
                self.imgs.append(this_img)
                print("loaded img %d size=(%d,%d): %s" %
                      (idx, this_img.shape[0], this_img.shape[1], this_path))
        print("loaded %d images" % (len(self.imgs)))

    def visualize_matches(self, img1, img2,
                          kp1, kp2, good,
                          mask= None, save_path=None):
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
                           singlePointColor= None,
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

    def drawlines(self, img1, img2, lines, pts1, pts2, line_num=None):
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

    def visualize_epipolar_lines(self, img1, img2, p1, p2, E, save_path):
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
        img5, _ = self.drawlines(
            img1, img2, lines1, p1_selected, p2_selected, 100)

        lines2 = cv2.computeCorrespondEpilines(
            p1_selected.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
        img3, _ = self.drawlines(
            img2, img1, lines2, p2_selected, p1_selected, 100)
        canvas = np.concatenate((img5, img3), axis=1)
        cv2.imwrite(save_path, canvas)

    def write_simple_obj(self, mesh_v, mesh_f, filepath, verbose=False):
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

    def detect_and_match_feature(self, img1, img2):
        """
        img1, img2: are input images
        The following outputs are needed:
        kp1, kp2: keypoints (here sift keypoints) of the two images
        matches_good: matches which pass the ratio test
        p1, p2: only the 2d points in the respective images
        pass ratio test. These points should correspond to each other.

        Steps:
        1. Compute sift descriptors.
        2. Match sift across two images.
        3. Use ratio test to get good matches.
        4. Store points retrieved from the good matches.

        Hints: See SIFT_create
        For feature matching you could use
        - FLANN Matcher:
        (https://docs.opencv.org/3.4/dc/de2/classcv_1_1FlannBasedMatcher.html)
        """
        # TODO: step 1 and 2
        # ...
        # keypoints
        sift = cv2.xfeatures2d.SIFT_create()


        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        p1 = []
        p2 = []

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                # 0.7 for image 1 n 2
                # 0.6 for image 2 n 3
                good_matches.append(m)
                #p1.append(kp1[m.trainIdx].pt)
                #p2.append(kp2[n.trainIdx].pt)

        # good_matches = np.sort(good_matches)
        good_matches =sorted(good_matches, key = lambda x:x.distance)

        p1 = np.float32([ kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        p2 = np.float32([ kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        #p1 = np.asarray(p1)
        #p2 = np.asarray(p2)
        #p1 = p1.reshape(-1, 1, 2)
        #p2 = p2.reshape(-1, 1, 2)

        # draw_params = dict(matchColor=(0, 255, 0),
        #                    singlePointColor=(255, 0, 0),
        #                    matchesMask=None,
        #                    flags=0)
        #
        # img_sift = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, **draw_params)
        #
        # plt.imshow(img_sift)
        # plt.show()

        return p1, p2, good_matches, kp1, kp2

    def compute_essential(self, p1, p2):
        """
        p1, p2: only the 2d points in the respective images
        pass ratio test. These points should correspond to each other.
        Outputs:
        Essential Matrix (E), and corresponding (mask)
        used in its computation. The mask contains the inlier_matches
        to compute E

        Hint: findEssentialMat
        """
        # TODO: step 3
        # ...

        E, mask = cv2.findEssentialMat(p1,p2, self.intrinsic,
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        print('size of E',E)
                                    
        matchesMask = mask.ravel().tolist()
        print('size of mask',len(mask))
        print('size of mask_m',len(matchesMask))
        #
        # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
        #                    singlePointColor=None,
        #                    matchesMask=matchesMask,  # draw only inliers
        #                    flags=2)
        # img_ess = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
        print('Essential Matrix \n', E)

        print('\n Shape of E matrix:', E.shape)
        return E, matchesMask

    def compute_pose(self, p1, p2, E):
        """
        p1, p2: only the 2d points in the respective images
        pass ratio test. These points should correspond to each other.
        E: Essential matrix
        Outputs:
        R, trans: Rotation, Translation vectors

        Hint: recoverPose
        """

        # TODO: step 4
        # ...
        print('Intrinsic Matrix :\n', self.intrinsic)
        # cv2.recoverPose(E, p1, p2, self.intrinsic, [[[[]]]])
        # someC, R, trans, _ = cv2.recoverPose(E, np.array(p1), np.array(p2), self.intrinsic)
        someC, R, trans, _ = cv2.recoverPose(E, p1, p2, self.intrinsic)

        print('Rotation matrix \n', R,'\n')
        print('Translation Matrix \n', trans,'\n')
        return R, trans

    def triangulate(self, p1, p2, R, trans, mask):
        """
        p1,p2: Points in the two images which correspond to each other
        R, trans: Rotation and translation matrix.
        mask: is obtained during computation of Essential matrix

        Outputs:
        point_3d: should be of shape (NumPoints, 3). The last dimension
        refers to (x,y,z) co-ordinates

        Hint: triangulatePoints
        """
        # TODO: step 5
        # ...
        # print(np.shape(np.array(p1)))
        # print(np.shape(np.array(p2)))
        # print(np.shape(np.array(R)))
        # print(np.shape(np.array(trans)))
        # print(np.shape(np.array(mask)))


        # kp1_3D = np.ones((3, p1.shape[0]))
        # kp2_3D = np.ones((3, p2.shape[0]))
        # kp1_3D[0], kp1_3D[1] = p1[:, 0].copy(), p1[:, 1].copy()
        # kp2_3D[0], kp2_3D[1] = p2[:, 0].copy(), p2[:, 1].copy()
        # TT = np.concatenate((R, trans),axis=1)
        # print(np.shape(TT))
        # X = cv2.triangulatePoints(TT, TT, kp1_3D[:2], kp2_3D[:2])

        proj_2 = np.hstack((R, trans))
        proj_1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

        proj_pt1 = np.dot(self.intrinsic, proj_1)
        proj_pt2 = np.dot(self.intrinsic, proj_2)
        print("Projection Matrix 1:", proj_pt1)
        print("Projection Matrix 2:", proj_pt2)

        # undistort points
        # matchesMask = mask.ravel().tolist()

        p1 = p1[np.asarray(mask) == 1, :, :]
        p2 = p2[np.asarray(mask) == 1, :, :]
        p1_un = cv2.undistortPoints(p1, self.intrinsic, None)
        p2_un = cv2.undistortPoints(p2, self.intrinsic, None)
        p1_un = np.squeeze(p1_un)
        p2_un = np.squeeze(p2_un)

        # triangulate points this requires points in normalized coordinate
        point_4d_hom = cv2.triangulatePoints(proj_pt1, proj_pt2, p1_un.T, p2_un.T)
        point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_3d[:3, :].T

        print('Output', point_3d)
        #point_3d = cv2.triangulatePoints(np.array(p1), np.array(p2))
        return point_3d

    def run(self):

        self.load_images()

        # pair processing

        # step 1 and 2: detect and match feature
        p1, p2, good_matches, kp1, kp2 = self.detect_and_match_feature(
            self.imgs[0], self.imgs[1])

        self.visualize_matches(
            self.imgs[0], self.imgs[1], kp1, kp2, good_matches,
            save_path=join(self.output_dir, 'sift_match_0_1.png'))

        # step 3: compute essential matrix
        E, mask = self.compute_essential(p1, p2)

        self.visualize_matches(
            self.imgs[0], self.imgs[1], kp1, kp2, good_matches, mask=mask,
            save_path=join(self.output_dir, 'inlier_match_0_1.png'))

        self.visualize_epipolar_lines(
            self.imgs[0], self.imgs[1], p1, p2, E,
            save_path=join(self.output_dir, 'epipolar_lines_0_1.png'))

        # step 4: recover pose
        R, trans = self.compute_pose(p1, p2, E)

        # step 5: triangulation
        point_3d = self.triangulate(p1, p2, R, trans, mask)

        self.write_simple_obj(point_3d, None, filepath=join(
            self.output_dir, 'output_0_1.obj'))

        # (optional, not scored) we can process all image pairs
        # ...

# -----------------------------------------------------------------------------


def safe_mkdir(file_dir):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)


# -----------------------------------------------------------------------------


def intrinsic_reader(txt_file):
    with open(txt_file) as f:
        lines = f.readlines()
    return np.array(
        [l.strip().split(' ') for l in lines],
        dtype=np.float32
    )


# -----------------------------------------------------------------------------


def main():

    img_pattern = r'C:/Users/charu/Desktop/cv/HW3/HW3_Data/cs677_hw3_data/rdimage.???.ppm'
    
    
    intrinsic = intrinsic_reader(r'C:/Users/charu/Desktop/cv/HW3/HW3_Data/cs677_hw3_data/intrinsics.txt')
    output_dir = './output'
    safe_mkdir(output_dir)

    sfm_solver = SFMSolver(img_pattern, intrinsic, output_dir, downscale=1)
    sfm_solver.run()


if __name__ == '__main__':

    main()


