# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:36:11 2018

@author: Kevin Beltran
"""

from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils

import tensorflow.contrib as tc 

from layers_slim import *



def FCN_Seg(self, is_training=True):

    #Set training hyper-parameters
    self.is_training = is_training
    self.normalizer = tc.layers.batch_norm
    self.bn_params = {'is_training': self.is_training}

      
    print("input", self.tgt_image)

    with tf.variable_scope('First_conv'):
        conv1 = tc.layers.conv2d(self.tgt_image, 32, 3, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

        print("Conv1 shape")
        print(conv1.get_shape())

    x = inverted_bottleneck(conv1, 1, 16, 0,self.normalizer, self.bn_params, 1)
    #print("Conv 1")
    #print(x.get_shape())

    #180x180x24
    x = inverted_bottleneck(x, 6, 24, 1,self.normalizer, self.bn_params, 2)
    x = inverted_bottleneck(x, 6, 24, 0,self.normalizer, self.bn_params, 3)
    
    print("Block One dim ")
    print(x)

    DB2_skip_connection = x    
    #90x90x32
    x = inverted_bottleneck(x, 6, 32, 1,self.normalizer, self.bn_params, 4)
    x = inverted_bottleneck(x, 6, 32, 0,self.normalizer, self.bn_params, 5)
    
    print("Block Two dim ")
    print(x)

    DB3_skip_connection = x
    #45x45x96
    x = inverted_bottleneck(x, 6, 64, 1,self.normalizer, self.bn_params, 6)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 7)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 8)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 9)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 10)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 11)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 12)
    
    print("Block Three dim ")
    print(x)

    DB4_skip_connection = x
    #23x23x160
    x = inverted_bottleneck(x, 6, 160, 1,self.normalizer, self.bn_params, 13)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 14)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 15)
    
    print("Block Four dim ")
    print(x)

    #23x23x320
    x = inverted_bottleneck(x, 6, 320, 0,self.normalizer, self.bn_params, 16)
    
    print("Block Four dim ")
    print(x)
    

    # Configuration 1 - single upsampling layer
    if self.configuration == 1:

        upsampled_features = TransitionUp_elu(x,120,16,'current_up5')
        print("Configuration 1:")
        print(upsampled_features)
        current_up5 = crop(upsampled_features,self.tgt_image)
        print("Cropped data:")
        print(current_up5)

        #input is features named 'x'

        # TODO(1.1) - incorporate a upsample function which takes the features of x 
        # and produces 120 output feature maps, which are 16x bigger in resolution than 
        # x. Remember if dim(upsampled_features) > dim(input image) you must crop
        # upsampled_features to the same resolution as input image
        # output feature name should match the next convolution layer, for instance
        # current_up5

        End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
        
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    # Configuration 2 - single upsampling layer plus skip connection
    if self.configuration == 2:

        upsampled_features = TransitionUp_elu(x,120,2,'current_up5')
        print("Configuration 2 first upsample shape:")
        print(upsampled_features)
        print("DB4_skip_connection here!!:")
        print(DB4_skip_connection)
        concatenation = Concat_layers(DB4_skip_connection, upsampled_features)
        print("Concatenation here!!:")
        print(concatenation)
        convolution_concat = Convolution(concatenation,256,3,"convolution_concat")
        upsampled_features2 = TransitionUp_elu(convolution_concat,120,8,'current_up3')
        print("Configuration 2 second upsample shape:")
        current_up3 = crop(upsampled_features2,self.tgt_image)
        print("Cropped data:")
        print(current_up3)
        

        #input is features named 'x'

        # TODO (2.1) - implement the refinement block which upsample the data 2x like in configuration 1 
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps 
        
        
        # TODO (2.2) - incorporate a upsample function which takes the features from TODO (2.1) 
        # and produces 120 output feature maps, which are 8x bigger in resolution than 
        # TODO (2.1). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up3

        End_maps_decoder1 = slim.conv2d(current_up3, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
        
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    # Configuration 3 - Two upsampling layer plus skip connection
    if self.configuration == 3:

        # Upsample features maintaining number of features
        upsampled_features = TransitionUp_elu(x,120,2,'current_up5')
        #Shows shape of upsampled_features
        print("Configuration 3 first upsample shape:")
        print(upsampled_features)
        #Shows shape of DB4_skip_connection
        print("DB4_skip_connection here!!:")
        print(DB4_skip_connection)
        if dim(upsampled_features)>dim(DB4_skip_connection):
            upsampled_features = crop(upsampled_features,DB4_skip_connection)
        # Makes concatenation DB4_skip_connection - upsampled_features 
        concatenation = Concat_layers(DB4_skip_connection, upsampled_features)
        #Shows shape of concatenation
        print("Concatenation here!!:")
        print(concatenation)
        #Makes convolution of concatenation and gives 256 features
        convolution_concat = Convolution(concatenation,256,3,"convolution_concat")
        #Shows shape of convolution of concatenation
        print("convolution_concat here!!:")
        print(convolution_concat)
        #Upsample features maintaining number of features - 256
        upsampled_features2 = TransitionUp_elu(convolution_concat,160,'current_up3')
        #Shows shape of upsampled_features
        print("Configuration 3 second upsample shape:")
        print("DB3_skip_connection here!!:")
        print(DB3_skip_connection)
        if dim(upsampled_features2)>dim(DB3_skip_connection):
            led_features2 = crop(upsampled_features2,DB3_skip_connection)
        # Makes concatenation DB3_skip_connection - upsampled_features2 
        concatenation2 = Concat_layers(DB3_skip_connection, upsampled_features2, "upconv2")
        #Shows shape of concatenation 2
        print("Concatenation2 here!!:")
        print(concatenation2)
        #Makes convolution of concatenation2 and gives 160 features
        convolution_concat2 = Convolution(concatenation2,160,3,"convolution_concat2")
        #Shows shape of convolution of concatenation
        print("Convolution_concat2!!:")
        print(convolution_concat2)
        #Upsample features downgrading numbers of features - 120
        upsampled_features3 = TransitionUp_elu(convolution_concat2,120,4,'current_up4')
        #Shows shape of upsampled_features
        print("Configuration 3 third upsample shape:")
        print(upsampled_features3)
        if dim(upsampled_features3)>dim(self.tgt_image):
			current_up4 = crop(upsampled_features3,self.tgt_image)
        #Crops upsampled features dim to original input dim
        current_up4 = upsampled_features3
        print("Cropped data:")
        print(current_up4)
        #input is features named 'x'

        # TODO (3.1) - implement the refinement block which upsample the data 2x like in configuration 1 
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps 
       
        # TODO (3.2) - Repeat TODO(3.1) now producing 160 output feature maps and fusing the upsampled features 
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.

        # TODO (3.3) - incorporate a upsample function which takes the features from TODO (3.2)  
        # and produces 120 output feature maps which are 4x bigger in resolution than 
        # TODO (3.2). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4  
              

        End_maps_decoder1 = slim.conv2d(current_up4, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
        
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    #Full configuration 
    if self.configuration == 4:

        # Upsample 
        upsampled_features = TransitionUp_elu(x,120,2,'current_up1')
        #Shows shape of upsampled_features
        print("Configuration 4 first upsample shape:")
        print(upsampled_features)
        #Shows shape of DB4_skip_connection
        print("DB4_skip_connection here!!:")
        print(DB4_skip_connection)
        # Makes concatenation DB4_skip_connection - upsampled_features 
        if dim(upsampled_features)>dim(DB4_skip_connection):
            upsampled_features = crop(upsampled_features,DB4_skip_connection)
        concatenation = Concat_layers(DB4_skip_connection, upsampled_features)
        #Shows shape of concatenation
        print("Concatenation here!!:")
        print(concatenation)
        #Makes convolution of concatenation and gives 256 features
        convolution_concat = Convolution(concatenation,256,3,"convolution_concat")
        #Shows shape of convolution of concatenation
        print("convolution_concat here!!:")
        print(convolution_concat)
        upsampled_features2 = TransitionUp_elu(convolution_concat,160,2,'current_up3')
        #Shows shape of upsampled_features
        print("Configuration 4 second upsample shape:")
        print("DB3_skip_connection here!!:")
        print(DB3_skip_connection)
        if dim(upsampled_features2)>dim(DB3_skip_connection):
            led_features2 = crop(upsampled_features2,DB3_skip_connection)
        # Makes concatenation DB3_skip_connection - upsampled_features2 
        concatenation2 = Concat_layers(DB3_skip_connection, upsampled_features2, "upconv2")
        #Shows shape of concatenation 2
        print("Concatenation2 here!!:")
        print(concatenation2)
        #Makes convolution of concatenation2 and gives 160 features
        convolution_concat2 = Convolution(concatenation2,160,3,"convolution_concat2")
        #Shows shape of convolution of concatenation
        print("Convolution_concat2!!:")
        print(convolution_concat2)
        #Upsample features maintaining numbers of features - 160
        upsampled_features3 = TransitionUp_elu(convolution_concat2,96,2,'current_up4')
        #Shows shape of upsampled_features
        print("Configuration 4 third upsample shape:")
        print(upsampled_features3)
        print("DB2_skip_connection here!!:")
        print(DB2_skip_connection)
        if dim(upsampled_features3)>dim(DB2_skip_connection):
			upsampled_features3 = crop(upsampled_features3,DB2_skip_connection)
        concatenation3 = Concat_layers(DB2_skip_connection, upsampled_features3, "upconv3")
        #Shows shape of concatenation 3
        print("Concatenation2 here!!:")
        print(concatenation3)
        #Makes convolution of concatenation3 and gives 96 features
        convolution_concat3 = Convolution(concatenation3,96,3,"convolution_concat3")
        #Shows shape of convolution of concatenation
        print("Convolution_concat3!!:")
        print(convolution_concat3)
        #Upsample features upgrading numbers of features - 120
        upsampled_features4 = TransitionUp_elu(convolution_concat3,120,2,'current_up5')
        if dim(upsampled_features4)>dim(self.tgt_image):
			current_up5 = crop(upsampled_features4,self.tgt_image)
        current_up5 = upsampled_features4
        print("Cropped data:")
        print(current_up5)

        ######################################################################################
        ######################################### DECODER Full #############################################

       
        
        # TODO (4.1) - implement the refinement block which upsample the data 2x like in configuration 1 
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps 
       
        # TODO (4.2) - Repeat TODO(4.1) now producing 160 output feature maps and fusing the upsampled features 
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.

        # TODO (4.3) - Repeat TODO(4.2) now producing 96 output feature maps and fusing the upsampled features 
        # with the corresponding skip connection (DB2_skip_connection) through concatenation.

        # TODO (4.4) - incorporate a upsample function which takes the features from TODO(4.3) 
        # and produce 120 output feature maps which are 2x bigger in resolution than 
        # TODO(4.3). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4 
        
        End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
        
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    
    return Reshaped_map

