import tensorflow as tf
import keras
import keras.backend as K
from keras.utils import plot_model
from model import pastiche_model
from training import get_loss_net, get_content_losses, get_style_losses, tv_loss

import pdb

"""Visualize the pastiche model
"""
width_factor = 2
nb_classes = 1

def visualize_pastiche_model():
  # visualize pastiche mdoel
  print('Creating pastiche model...')
  class_targets = K.placeholder(shape=(None,), dtype=tf.int32)
  # The model will be trained with 256 x 256 images of the coco dataset.
  # Using following parameters: 
  # image_size = 256
  # width_factor (a multiplier to number of filters each layer) = 2
  # nb_classes = number of styles to be trained.
  # targets = some parameters used only with multiple classes in conditional instance norm. 
  pastiche_net = pastiche_model(256, width_factor=width_factor,
                                nb_classes=nb_classes,
                                targets=class_targets)
  plot_model(pastiche_net, to_file='pastiche.png')
  """Note:
  Pastiche network is the network for which you train to generate style.
  The architecture of the pastiche network is as follows:
    Let block = --> (pad->conv->inst_norm) -->
        res_block = --> (block->relu->block) --> + -->
                     |---------------------------^
    the full architecture is: 
    (block->relu)*3 --> (res_block)*5 --> (up_sample->block->relu)*2 --> (block->tanh) --> scaling
  Reflection padding is used. This makes sense since you don't want the output image to loose pattern around the edge.
  "Class targets" is a label index for the selected style. 
  Questions:
  (1) The model uses a tanh in the end -- but why?
  (2) Why does it use relu only once in the res_block?
  (3) The network size is fixed, but it's fully convolutional. What happens when I change the size? 
  """

def examine_loss_network():
  print('Creating pastiche model...')
  class_targets = K.placeholder(shape=(None,), dtype=tf.int32)
  pastiche_net = pastiche_model(256, width_factor=width_factor,
                                nb_classes=nb_classes,
                                targets=class_targets)
  print('Loading loss network...')
  loss_net, outputs_dict, content_targets_dict = get_loss_net(pastiche_net.output, input_tensor=pastiche_net.input)
  """Note:
  The loss network is basically two computational paths of a VGG architecture. 
  loss_net is the VGG. 
  outputs_dict stores all tensors of each VGG layers output for THE PASTICHE IMAGE
  content_targets_dict stores all tensors of each VGG layers for THE CONTENT IMAGE
  """
  
# visualize_pastiche_model()
examine_loss_network()
