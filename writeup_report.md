##Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/bboxes_and_heat.png
[video1]: ./project_video.mp4


###Histogram of Oriented Gradients (HOG)

####1. Explain how (see third code cell of the IPython notebook `vehicle_detection.py`) you extracted HOG features from the training images.

The function definitions used for this step are contained in the third code cell of the IPython notebook `vehicle_detection.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I used a small sample set of roughly a 1,000 images and did many experiments training my classifier with different HOG parameters to see what would yield the best results. By using a small set of images I was able to experiment really quickly with different HOG parameters.

####3. Describe how (see fifth code cell in `vehicle_detection.py`) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. I also used binned color parameters of `size=(32, 32)` and color histogram parameters of `nbins=32, bins_range(0, 256)` to improve my results.

###Sliding Window Search

####1. Describe how (see seventh code cell in `vehicle_detection.py`) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I searched in the bottom half of the image below the horizon and above the car hood. To decide on the scales and overlap values I ran many experiments on the test images to get a feel of what worked well.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:
![alt text][image3]

####3. Describe how (see ninth code cell in `vehicle_detection.py`) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

After finding positive detections using my sliding windows technique, I created a heatmap and then thresholded (I used a value of 2 since I search many different scales) that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

Here is an example of resulting bounding boxes and corresponding heat map:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./final_video.mp4)






---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I took a naive approach to solving this problem because of time constraints. I didn't keep track of the positive detections from frame to frame, so I used many scales and a high threshold to try and filter out false positives. The downside to this naive approach is that it is not very trustworthy and it's slow to process each frame. My pipeline will probably fail in most condition besides optimum conditions such as the one in the video I used for testing.

When I have more time, I plan to come back to this project and keep track of detections from frame to frame to filter out false positives and make better predictions about the next frame. I'd also like to experiment with using deep learning techniques to build my classifier.
