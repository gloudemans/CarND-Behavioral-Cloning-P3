import csv
import cv2
import math
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from random import shuffle
from skimage import transform 
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint

def transform_image(image, translation_distance, rotation_angle, camera_distance=200, horizon=60):
    """Return an image with perspective transformed as though the
    camera had been translated sideways and then rotated about the 
    vertical axis. The transformation assumes that the original image
    lies on the x-y plane, that the camera points down the y axis, that
    the camera height above the x-y plane is equal to the y dimension
    of the image, and that the camera is at a specified y projected 
    distance from points imaged at the bottom of the original image.
    The image is rotated about the z axis passing through the 
    center bottom point of the image following translation. The 
    rotation operation may introduce distortion.

    Args:
        image (numpy[Y,X,C]): Input image.
        translation_distance (float): Distance to translate the
          image in units of pixels. 
        rotation_angle (float): The rotation angle in degrees.
        camera_distance (float): Camera y distance from the points 
          imaged by lower edge of original image. Units are distances
          equivalent to the width of a pixel along the lower image edge.
        horizon (int): y coordiante of the horizon

    Returns:
        (numpy[Y,X,C]) Output image.

    """   
    
    # Make a copy of the image
    image = np.copy(image)
    
    # Convert to radians
    rotation_angle *= math.pi/180
    
    # Measure the source image
    y, x, c = image.shape
    
    # Adjust y measurement to reflect horizon
    y -= horizon

    # 3 space image frame is on x-z plane with camera on y axis
    # camera is at y=-yc where yc is distance from origin to camera
    # ground is at z=-y where y is image height

    # 3 space coordinates of lower image frame corners are:
    # (x0l, 0, y)
    # (x1l, 0, y)
    x0l = -x/2
    x1l =  x/2

    # Translate the lower image frame corners
    x0l += translation_distance
    x1l += translation_distance

    # Rotate the lower image frame corners on z axis
    # this moves y coordinates off the plane
    x0n =  x0l*math.cos(rotation_angle)
    y0n = -x0l*math.sin(rotation_angle)
    x1n =  x1l*math.cos(rotation_angle)
    y1n = -x1l*math.sin(rotation_angle)

    # Project ray from camera to lower corners onto the xy plane
    # we want these points in the original image to move to the lower
    # image corners following the perspective transformation
    s0 = camera_distance/(camera_distance+y0n)
    s1 = camera_distance/(camera_distance+y1n)

    x0s = x0n*s0
    x1s = x1n*s1
    z0s = y*s0
    z1s = y*s1

    # Translation has no effect on the upper points, but rotation
    # causes the upper corners to undergo a nonlinear horizontal
    # transformation. For large caera_distance/x the transformation is 
    # approximately a linear transformation. Since we don't have 
    # a good way to implement the nonlinear operation, we'll
    # aproximate as a translation.
    dx = math.tan(rotation_angle)*camera_distance 
   
    # Get integer version of x shift
    idx = round(dx)
  
    # If positive shift...
    if(idx>0):   

        # Shift pixels left above horizon
        image[:horizon,:-idx,:] = image[:horizon,idx:,:]
        image[:horizon,-idx:,:] = 0
        
    # If negative shift
    if(idx<0): 
        
        # Shift pixels right above horizon
        image[:horizon,-idx:,:] = image[:horizon,:idx,:]  
        image[:horizon,:-idx,:] = 0

    # Specify source points
    src = np.float32([[dx,0],[x+dx,0],[x0s+x/2,z0s],[x1s+x/2,z1s]])       

    # Specify destination points 
    dst = np.float32([[0,0],[x,0],[0,y],[x,y]])

    # Build transform matix
    transform = cv2.getPerspectiveTransform(src,dst)
    
    # Apply the transform
    image[horizon:,:,:] = cv2.warpPerspective(image[horizon:,:,:],transform,(x,y))

    return(image)

def get_augmented_image(sample, translation_distance=0, rotation_angle=0, camera_spacing=60):
    """Return an image with perspective transformed as though the
    camera had been translated sideways and then rotated about the 
    vertical axis. The image is formed by loading the closest (left, 
    center, or right) image from one of three cameras and shifting 
    and rotating that image. The images come from cameras pointed 
    in the same direction but laterally offset so that pixels along 
    the lower edge of the image are offset by the the specified camera 
    spacing.

    Args:
        sample (list): List of path names to center, left, and right
          images.
        translation_distance (float): Distance to translate the
          image in units of pixels. 
        rotation_angle (float): The rotation angle in degrees.
        camera_spacing (float): Distance in pixels by which the
          center image can be shifted so that pixels at the bottom
          edge of image are aligned.

    Returns:
        (numpy[Y,X,C]) Output image.

    """ 
    
    # Default to center camera
    camera = 0
    
    # If left camera is closer...
    if(translation_distance<-camera_spacing/2):
        
        # Use the left camera
        camera = 1
        
        # Adjust the image offset
        translation_distance += camera_spacing
        
    # If right camera is closer...
    if(translation_distance>camera_spacing/2):
    
        # Use the right camera
        camera = 2
        
        # Adjust the image offset
        translation_distance -= camera_spacing
        
    # load camera image
    image = plt.imread(sample[camera])
    
    # Transform the image
    image = transform_image(image, translation_distance, rotation_angle)
    
    return image

def load_sample_list(paths, test_size=0.1, rate=1):
    """Load image sample metadata data from driving_log.csv files
    in directories at the specified paths, shuffle the data,
    and split the data into train and test sets. Optionally
    decimate samples from files as they are retrieved such that
    fewer than all possible samples are produced.

    Args:
        paths (list): list of paths from which to read metadata.
        test_size (float): fraction of the samples to be used for test
        rate (float): fraction of the available samples to load.

    Returns:
        train_samples (list): training sample metadata
        test_samples (list): test sample metadata

    """ 
  
    # Empty list
    samples = []
    
    # For each path...
    for path in paths:
        
        # Open the drving log file
        with open(path+'/driving_log.csv') as csvfile:
            
            # Initialize csv reader
            reader = csv.reader(csvfile)
            
            # Discard the header row
            next(reader)
            
            # Clear the rate counter
            r = 0
            
            # For each line in the csv file..
            for line in reader:
                
                # Increment the rate counter
                r += rate
                
                # While the rate counter exceeds one
                while r >= 1:
                    
                    # Decrement it
                    r -= 1
                    
                    # For each image filename...
                    for i in range(3):
                        
                        # Construct full image path
                        s = line[i].strip().replace('\\','/')
                        s = s.split('/')[-1]
                        line[i] = path + '/IMG/' + s
                        
                    # Add the sample to the pile
                    samples.append(line)
    
    # Shuffle all of the samples
    shuffle(samples)
    
    # Split training and test samples
    train_samples, validation_samples = train_test_split(samples, test_size=test_size) 
    
    return(train_samples, validation_samples)

def generate_samples(samples, batch_size=32, max_rotation=20, max_translation=40):
    """Generate sample batches. Each sample in a batch contains an
    image and a steering angle.

    Args:
        samples (list): row of sample metadata containing paths to center,
        left, and right images; steering angle, and other metadata.
        batch_size (int): number of samples to yield per batch.

    Returns:
        x (numpy[img,y,x,c]) - image data
        y (numpy[img]): normalized steering angle (+/-1) = (+/-25 degrees)

    """ 
    # Measure the sample set
    num_samples = len(samples)
    
    while 1:
        
        # Shuffle each epoch
        shuffle(samples)  
        
        # For each batch...
        for offset in range(0, num_samples, batch_size):
            
            # Empty list
            x = []
            y = []            
            
            # For each sample of the batch...
            for batch_sample in samples[offset:offset+batch_size]:
                
                # Unpack fields
                center, left, right, steering, throttle, brake, speed = batch_sample
                
                # Convert string steering to float
                steering  = float(steering)
                
                # Decide whether to flip
                flip = np.random.randint(2)
                
                # Get random rotation angle
                rotation_angle = np.random.uniform(-max_rotation,max_rotation)
                
                # Get random translation amount
                translation_distance = np.random.uniform(-max_translation,max_translation)
                
                # Fetch the augmented image
                image = get_augmented_image(batch_sample[0:3], translation_distance, rotation_angle)
                
                # Isolate the region of interest
                image = image[60:130,60:260,:]
                
                # Adjust steering to reflect change in perspective
                steering -= translation_distance/max_translation*.05
                steering -= rotation_angle/max_rotation*.20
                
                # If flipping this image..
                if flip:    

                    # Flip the image and the sttering angle
                    image = np.fliplr(image) 
                    steering = -steering
                    
                # Add image and steering to their lists
                x.append(image)
                y.append(steering)
                
            # Yield the batch
            yield(np.array(x),np.array(y))
            
def train_model():
    """Construct and train model"""
     
    # Set dropout rate
    dropout_rate = .5
    
    # Set batch size
    batch_size = 32
        
    # Get forward samples from each track
    t1, v1 = load_sample_list(['T1F', 'T2F'], rate=1)
    
    # Get challenge samples from each track
    t2, v2 = load_sample_list(['T1C', 'T2C'], rate=2)
    
    # Get problem area samples from track 2
    t3, v3 = load_sample_list(['T2P'], rate=1)

    # Pile the samples into one large set
    train = t1+t2+t3
    valid = v1+v2+v3

    # Initialize sequential keras model
    model = Sequential()

    # Disable convolutional bias with batch norm
    conv_bias = False
    
    # We're ignoring pixels above the horizon, below the hood, and near the left and right edges.
    # The resulting region of interest is (200x70)
    
    # The model below parallels that described int he Nvidia end-to-end learning paper with
    # the following differences:
    #
    # * 200x70x3 input rather than 200x66x3
    # * No fixed input normalization
    # * No YUV conversion
    # * Batch normalization prior to every layer to make sure that
    #   model remains well scaled throughout.
    # * Elu activation functions rather than Relu activation functions since elu performance
    #   is generally slightly better (though computational complexity may be higher).
    # * Explicit glorot_uniform (He) initialization
    
    model.add(BatchNormalization(input_shape=(70, 200, 3)))

    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='elu', init='glorot_uniform', bias=conv_bias))
    model.add(BatchNormalization())

    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='elu', init='glorot_uniform', bias=conv_bias))
    model.add(BatchNormalization())

    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='elu', init='glorot_uniform', bias=conv_bias))
    model.add(BatchNormalization())

    model.add(Convolution2D(64,3,3,activation='elu', init='glorot_uniform', bias=conv_bias))
    model.add(BatchNormalization())

    model.add(Convolution2D(64,3,3,activation='elu', init='glorot_uniform', bias=conv_bias))
    model.add(BatchNormalization())

    model.add(Flatten())

    # Disable dense bias with batch norm
    dense_bias = False
    model.add(Dropout(dropout_rate))
    model.add(Dense(1154, activation='elu', init='glorot_uniform', bias=dense_bias))
    model.add(BatchNormalization())

    model.add(Dropout(dropout_rate))
    model.add(Dense( 100, activation='elu',  init='glorot_uniform', bias=dense_bias))
    model.add(BatchNormalization())

    model.add(Dropout(dropout_rate))          
    model.add(Dense(  50, activation='elu', init='glorot_uniform', bias=dense_bias))
    model.add(BatchNormalization())

    model.add(Dropout(dropout_rate))
    model.add(Dense(  10, activation='elu', init='glorot_uniform', bias=dense_bias))
    model.add(BatchNormalization())

    model.add(Dense(   1, activation='linear', init='glorot_uniform'))

    # Use MSE loss metric and adam optimizer
    model.compile(loss='mse', optimizer='adam')
    
    # Train for 20 epochs
    epochs = 20
    
    # Make generators
    train_generator = generate_samples(train, batch_size = batch_size)
    validation_generator = generate_samples(valid, batch_size = batch_size)  
    
    # Make checkpoint callback to save model each epoch
    model_save = ModelCheckpoint(filepath='model.h5', mode='auto', period=1)
    
    # Train the model
    model.fit_generator(train_generator, samples_per_epoch= \
                len(train), validation_data=validation_generator, \
                nb_val_samples=len(valid), nb_epoch=epochs, callbacks=[model_save])
    
train_model()
