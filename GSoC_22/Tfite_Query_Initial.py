import tensorflow as tf
import numpy as np 
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-i','--imagepath',type=str,metavar='',help='path of the image') 
parser.add_argument('-m','--modelpath',type=str,metavar='',help='path of the model') 
parser.add_argument('-j','--jsonpath',type=str,metavar='',help='path of json file') 

args = parser.parse_args()

def query_gmodel(image_path,model_path,json_path):
    
    image_vis = Image.open(image_path)
    meta_vis = json.load(open(json_path))

    
    lx, ly, lw, lh = meta_vis['leye_x'], meta_vis['leye_y'], meta_vis['leye_w'], meta_vis['leye_h']
    rx, ry, rw, rh = meta_vis['reye_x'], meta_vis['reye_y'], meta_vis['reye_w'], meta_vis['reye_h']
    l_eye = image_vis.crop((max(0, lx), max(0, ly), max(0, lx+lw), max(0, ly+lh)))
    r_eye = image_vis.crop((max(0, rx), max(0, ry), max(0, rx+rw), max(0, ry+rh)))
    
    
    #flipcode
    l_eye = l_eye.transpose(Image.FLIP_LEFT_RIGHT)

    
    left_bb = np.array([[lx, ly, lw, lh]])
    left_bb = tf.convert_to_tensor(left_bb,tf.float32) #left bounding box
    right_bb = np.array([[rx, ry, rw, rh]])
    right_bb = tf.convert_to_tensor(right_bb,tf.float32) #right bounding box 
    
    r_eye = np.asarray(r_eye)
    l_eye = np.asarray(l_eye)
    
    r_eye = tf.convert_to_tensor(r_eye,tf.float32)
    r_eye_resized = tf.image.resize(r_eye, [128,128])    
    l_eye = tf.convert_to_tensor(l_eye,tf.float32)
    l_eye_resized = tf.image.resize(l_eye, [128,128])    
    
    r_eye_i = tf.cast(r_eye_resized,tf.float32)
    l_eye_i = tf.cast(l_eye_resized,tf.float32)    
    
    l_eye_final = tf.expand_dims(l_eye_i, 0) #final left eye as tensor - [1,128,128,3]  
    r_eye_final = tf.expand_dims(r_eye_i, 0)  #final right eye as tensor - [1,128,128,3]  
    
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], right_bb)
    interpreter.set_tensor(input_details[1]['index'], left_bb)
    interpreter.set_tensor(input_details[2]['index'], r_eye_final)
    interpreter.set_tensor(input_details[3]['index'], l_eye_final)
    interpreter.invoke()
    

    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data


if __name__=='__main__':
    IMAGE_LOCATION = args.imagepath
    MODEL_LOCATION = args.modelpath
    JSON_LOCATION = args.jsonpath
    
    print('IMAGE_LOCATION : ' , IMAGE_LOCATION)
    print('MODEL_LOCATION : ' , MODEL_LOCATION)
    print('JSON_LOCATION : ' , JSON_LOCATION)
    
    
    out_test_1 = query_gmodel(IMAGE_LOCATION,MODEL_LOCATION,JSON_LOCATION)
    
    print('---Output---')
    print(out_test_1)








