<<<<<<< HEAD
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import cv2
import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(tfImage, input_height=299, input_width=299,
        input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  image_np = cv2.resize(tfImage,(input_height,input_width))
  mean_img = 108                    #image_np.mean()
  neg_mask = image_np < mean_img
  pos_mask = image_np >=mean_img
  image_np[neg_mask]=0
  image_np[pos_mask]=1

  #image_array=np.array(openCVImage)[:, :, 0:3]
  #image_reader=file_name
  image_reader = np.expand_dims(image_np, axis=0)

  #print(image_reader.shape)
  #print(image_reader)
  float_caster = tf.cast(image_reader, tf.float32)
  
  #dims_expander = tf.expand_dims(float_caster, 0);
  #resized = tf.image.resize_bilinear(float_caster, [input_height, input_width])
  #normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  
  
  sess = tf.Session()
  result = sess.run(float_caster)
  #print(sess.eval(result))
  #print(result)
  #print(result==image_reader)
  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  #file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Mul"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  #if args.image:
  #  file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  import cv2
  cap = cv2.VideoCapture(cv2.CAP_DSHOW)

  graph = load_graph(model_file)
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);
  labels = load_labels(label_file)
  ret = True

  with graph.as_default():
    with tf.Session(graph=graph) as sess:
      while (ret):
        ret,tfImage = cap.read()
        cv2.imshow('image',cv2.resize(tfImage,(800,600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          break
        t = read_tensor_from_image_file(tfImage,
                                        input_height=input_height,
                                        input_width=input_width,
                                        input_mean=input_mean,
                                        input_std=input_std)

        #with tf.Session(graph=graph) as sess:
          #start = time.time()
        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})
          #end=time.time()
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]

        #print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
        template = "{} (score={:0.5f})"
        #for i in top_k:
        i=top_k[0]
        print(template.format(labels[i], results[i]))
        
  print("See you again!")
  '''
  if(results[i]<0.97000):
          labels[i]='fist'
          results[i]=results[top_k[1]]
  '''
=======
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import cv2
import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(tfImage, input_height=299, input_width=299,
        input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  image_np = cv2.resize(tfImage,(input_height,input_width))
  mean_img = 108                    #image_np.mean()
  neg_mask = image_np < mean_img
  pos_mask = image_np >=mean_img
  image_np[neg_mask]=0
  image_np[pos_mask]=1

  #image_array=np.array(openCVImage)[:, :, 0:3]
  #image_reader=file_name
  image_reader = np.expand_dims(image_np, axis=0)

  #print(image_reader.shape)
  #print(image_reader)
  float_caster = tf.cast(image_reader, tf.float32)
  
  #dims_expander = tf.expand_dims(float_caster, 0);
  #resized = tf.image.resize_bilinear(float_caster, [input_height, input_width])
  #normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  
  
  sess = tf.Session()
  result = sess.run(float_caster)
  #print(sess.eval(result))
  #print(result)
  #print(result==image_reader)
  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  #file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Mul"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  #if args.image:
  #  file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  import cv2
  cap = cv2.VideoCapture(cv2.CAP_DSHOW)

  graph = load_graph(model_file)
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);
  labels = load_labels(label_file)
  ret = True

  with graph.as_default():
    with tf.Session(graph=graph) as sess:
      while (ret):
        ret,tfImage = cap.read()
        cv2.imshow('image',cv2.resize(tfImage,(800,600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          break
        t = read_tensor_from_image_file(tfImage,
                                        input_height=input_height,
                                        input_width=input_width,
                                        input_mean=input_mean,
                                        input_std=input_std)

        #with tf.Session(graph=graph) as sess:
          #start = time.time()
        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})
          #end=time.time()
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]

        #print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
        template = "{} (score={:0.5f})"
        #for i in top_k:
        i=top_k[0]
        print(template.format(labels[i], results[i]))
        
  print("See you again!")
  '''
  if(results[i]<0.97000):
          labels[i]='fist'
          results[i]=results[top_k[1]]
  '''
>>>>>>> 7f807316aa2366b4fa08e23f19d08388ab45ca5b
