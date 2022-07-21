# Object Detection using the YOLO CNN with OpenCV and Python 
#### Modified by [VHEED](https://twitter.com/The_Vheed)

The OpenCV _**dnn**_ module supports running inference on pre-trained deep learning models from popular frameworks like Caffe, Torch and TensorFlow.

When it comes to object detection, popular detection frameworks are
 * YOLO
 * SSD
 * Faster R-CNN
 
 Support for running YOLO/DarkNet has been added to OpenCV _**dnn**_ module

## Dependencies
  * opencv
  * numpy
  * wget
  
```commandline
$ pip3 install numpy opencv-python wget
```

**Note: Compatability with Python 2.x has not been officially tested.**

 ## YOLO (You Only Look Once)
 
 Download the pre-trained YOLO v3 weights file from this [link](https://pjreddie.com/media/files/yolov3.weights) and place it in the current directory or you can directly download to the current directory in terminal using

(_If the weights aren't in the same folder as the script, the script would download one automatically!_) 

 ```commandline
 $ wget https://pjreddie.com/media/files/yolov3.weights
 ```

 Checkout this [blog post](http://www.arunponnusamy.com/yolo-object-detection-opencv-python.html) to learn more.
 
## Running the Code

The source can be changed by explicitly changing the ```source``` variable located at top of the main.py script.

Only a video source is to be used. An Image source could also be used with some slight modiifications of the code

After that, just run the script with:
```commandline
python3 main.py
```

### Additional Info

* The ```skip_factor``` variable determines the number of frames to skip which would also affect the framerate of the final output.

* The output video has a width of 960px regardless of the input size
* The Code can run on a CUDA enabled gpu by the uncommenting of the following lines:
    ```python
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    ```
  
    to:
    ```python
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    ```

### Reference
[Yolo Object Detection (OpenCV Python)](http://www.arunponnusamy.com/yolo-object-detection-opencv-python.html)