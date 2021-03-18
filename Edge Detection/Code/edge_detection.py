import xml.etree.ElementTree as ET

def boxes_in_groundtruth(gt_image_path):
     
    ground_truth_boxes = []
    tree = ET.parse(gt_image_path)
    root = tree.getroot()
    for items in root.findall('object/bndbox'):
        xmin = items.find('xmin')
        ymin = items.find('ymin')
        xmax = items.find('xmax')
        ymax = items.find('ymax')
        ground_truth_boxes.append([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)])
        #print("Ground truth images",ground_truth_boxes)
    return ground_truth_boxes


def area_of_intersection(box_100,groud_truth_box):
    #print(box_100,groud_truth_box)
    x1 = max(box_100[0], groud_truth_box[0])
    x2 = min(box_100[2], groud_truth_box[2])
    y1 = max(box_100[1], groud_truth_box[1])
    y2 = min(box_100[3], groud_truth_box[3])
    #print(x1,x2,y1,y2)
    if (x2 - x1 < 0) or (y2 - y1 < 0):
        return 0
    else:
        #print((x2 - x1 + 1) * (y2 - y1 + 1))
        return (x2 - x1 ) * (y2 - y1)


def find_IOU(boxes_100,ground_truth_boxes):
    iou_qual_boxes=[]
    boxes_best_iou=[]
    for box_g in ground_truth_boxes:
        best_iou=0
        best_box=0
        area_groundtruth_box= (box_g[2]-box_g[0] )* (box_g[3]-box_g[1])
        #print(box_g)
        #print(area_groundtruth_box)
        for box in boxes_100:
            area_box_100=(box[2]-box[0] )* (box[3]-box[1])
            area_of_intersec = area_of_intersection(box, box_g)
            #print(area_of_intersec)
            area_of_union = area_box_100 + area_groundtruth_box - area_of_intersec 
            #print(area_of_union)
            iou = float(area_of_intersec) / float(area_of_union)
            #print(iou)
            if iou > 0.5:
                iou_qual_boxes.append(box)
                if iou > best_iou:
                    best_iou = iou
                    best_box = box
        if best_iou != 0:
            boxes_best_iou.append(best_box)
    return iou_qual_boxes, boxes_best_iou


import cv2
import numpy as np
import argparse
image= cv2.imread("./HW2_Data/JPEGImages/000220.jpg")
cv2.imshow("000220",image)
model_path = "./model.yml.gz"
Edge_Detection = cv2.ximgproc.createStructuredEdgeDetection(model_path)
RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Get the edges
edges = Edge_Detection.detectEdges(np.float32(RGB_image)/255.0)
# Create an orientation map
orient_map = Edge_Detection.computeOrientation(edges)
# Suppress edges
edges = Edge_Detection.edgesNms(edges, orient_map)
cv2.imshow("Edges", edges)
# k = cv2.waitKey()
# cv2.destroyAllWindows()
 #Create edge box:
edge_boxes = cv2.ximgproc.createEdgeBoxes()
#edge_boxes.setMaxBoxes(100)
edge_boxes.setAlpha(0.6)
edge_boxes.setBeta(0.5)
get_boxes,scores = edge_boxes.getBoundingBoxes(edges, orient_map)

# Convert (x,y,w,h) parameters for the top 100 proposal boxes into (x, y, x+w, y+h) parameters
# to be consistent with the xml tags of the ground truth boxes where (x,y) indicates the
# top left corner and (x+w,y+h) indicates the bottom right corner of bounding box
first_100_box= 100
boxes_100=[]


for box in get_boxes:
    boxes_100.append([box[0],box[1],box[0]+box[2],box[1]+box[3]])
#print(boxes_100)
print (len(boxes_100))
#Get all ground truth images from annotated xml files
output_image_100= image.copy()
# Fetch all ground truth boxes from the annotated image file
ground_truth_boxes= boxes_in_groundtruth("./HW2_Data/Annotations/000220.xml")
print("Number of Ground Truth Boxes = ", len(ground_truth_boxes))
for i in range(0,len(boxes_100)):
    x_top, y_top, x_bottom, y_bottom= boxes_100[i]
    #cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.rectangle(output_image_100, (x_top, y_top), (x_bottom, y_bottom), (0, 255, 0), 1, cv2.LINE_AA)
#cv2.imshow("output_image_100", output_image_100)
cv2.imwrite("./Results/output_image_100_000220_ED.png",output_image_100)

IOU_qual_boxes, final_boxes = find_IOU(boxes_100, ground_truth_boxes)
print("Number of Qualified Boxes with IOU > 0.5 = ", len(IOU_qual_boxes))
#print("Qualified Boxes = ", IOU_qual_boxes)

output_img_iou_qualified = image.copy()
image_final = image.copy()


#Draw bounding boxes for iou_qualified_boxes
for i in range(0, len(IOU_qual_boxes)):
    top_x, top_y, bottom_x, bottom_y = IOU_qual_boxes[i]
    cv2.rectangle(output_img_iou_qualified, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1, cv2.LINE_AA)
for i in range(0, len(ground_truth_boxes)):
    top_x, top_y, bottom_x, bottom_y = ground_truth_boxes[i]
    cv2.rectangle(output_img_iou_qualified, (top_x, top_y), (bottom_x, bottom_y), (0, 0, 255), 1, cv2.LINE_AA)
cv2.imshow("Output_IOU_Qualified_Proposals", output_img_iou_qualified)
cv2.imwrite("./Results/Output_IOU_Qualified_Proposals_000220_ED.png", output_img_iou_qualified)
#cv2.waitKey(1000)



print("Number of final boxes = ", len(final_boxes))
print("Final boxes = ", final_boxes)

# Recall is calculated as the fraction of ground truth boxes that overlap with at least one proposal box with
# Intersection over Union (IoU) > 0.5
recall = len(final_boxes) / len(ground_truth_boxes)
print("Recall is = ", recall)

# Draw bounding boxes for final_boxes
for i in range(0, len(final_boxes)):
    top_x, top_y, bottom_x, bottom_y = final_boxes[i]
    cv2.rectangle(image_final, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1, cv2.LINE_AA)
for i in range(0, len(ground_truth_boxes)):
    top_x, top_y, bottom_x, bottom_y = ground_truth_boxes[i]
    cv2.rectangle(image_final, (top_x, top_y), (bottom_x, bottom_y), (0, 0, 255), 1, cv2.LINE_AA)
cv2.imshow("Output_Final_Boxes", image_final)
cv2.imwrite("./Results/output_img_final_000220_ED.png", image_final)
#cv2.waitKey(1000)
































