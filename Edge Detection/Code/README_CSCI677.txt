To run the code, download it and execute using the following command in terminal/command prompt:

For Selective Search:


1. python hom2_selective_search.py <strategy>
2. Specify image path in : image= cv2.imread("./HW2_Data/JPEGImages/000220.jpg")
3. Specify xml path in : ground_truth_boxes= boxes_in_groundtruth("./HW2_Data/Annotations/000220.xml")
input_image_path: Enter the image file name including path
annotated_image_path: Enter the annotated image file name including path
strategy: Enter the strategy – ‘color’ for color strategy, ‘texture’ for texture strategy‘all’, for all strategies
Eg: python selective_search.py  color

For Edge Boxes:
1. python edge_detection.py

