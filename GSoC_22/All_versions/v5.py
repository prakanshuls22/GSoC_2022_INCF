import tensorflow as tf
import numpy as np 
import json
from PIL import Image
import pandas as pd
import argparse 

parser = argparse.ArgumentParser() 
parser.add_argument('-m','--modelpath',type=str,metavar='',help='path of the model') 

args = parser.parse_args()


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "device": tf.io.FixedLenFeature([], tf.string),
        "screen_h": tf.io.FixedLenFeature([], tf.int64),
        "screen_w": tf.io.FixedLenFeature([], tf.int64),
        "face_valid": tf.io.FixedLenFeature([], tf.int64),
        "face_x": tf.io.FixedLenFeature([], tf.int64),
        "face_y": tf.io.FixedLenFeature([], tf.int64),
        "face_w": tf.io.FixedLenFeature([], tf.int64),
        "face_h": tf.io.FixedLenFeature([], tf.int64),
        "leye_x": tf.io.FixedLenFeature([], tf.int64),
        "leye_y": tf.io.FixedLenFeature([], tf.int64),
        "leye_w": tf.io.FixedLenFeature([], tf.int64),
        "leye_h": tf.io.FixedLenFeature([], tf.int64),
        "reye_x": tf.io.FixedLenFeature([], tf.int64),
        "reye_y": tf.io.FixedLenFeature([], tf.int64),
        "reye_w": tf.io.FixedLenFeature([], tf.int64),
        "reye_h": tf.io.FixedLenFeature([], tf.int64),
        "dot_xcam": tf.io.FixedLenFeature([], tf.float32),
        "dot_y_cam": tf.io.FixedLenFeature([], tf.float32),
        "dot_x_pix": tf.io.FixedLenFeature([], tf.float32),
        "dot_y_pix": tf.io.FixedLenFeature([], tf.float32),
        "reye_x1": tf.io.FixedLenFeature([], tf.int64),
        "reye_y1": tf.io.FixedLenFeature([], tf.int64),
        "reye_x2": tf.io.FixedLenFeature([], tf.int64),
        "reye_y2": tf.io.FixedLenFeature([], tf.int64),
        "leye_x1": tf.io.FixedLenFeature([], tf.int64),
        "leye_y1": tf.io.FixedLenFeature([], tf.int64),
        "leye_x2": tf.io.FixedLenFeature([], tf.int64),
        "leye_y2": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    return example

def normalize_bbco(bbox,fw,fh):
    den = float(max(fw,fh))
    x1 = (float(bbox[0]) - 0.5 * fw) / den
    y1 = (float(bbox[1]) - 0.5 * fh) / den
    x2 = (float(bbox[2])  - 0.5 * fw) / den
    y2 = (float(bbox[3])  - 0.5 * fh) / den
    return np.asarray([[x1, y1, x2, y2]])

if __name__=='__main__':
    
    model_path = args.modelpath

    raw_dataset = tf.data.TFRecordDataset("./combined_individuals.tfrec")
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)


    ## calculating total size of dataset 
    print("\n------Calculating Total Size------")
    total_count=0

    for x in parsed_dataset.as_numpy_iterator():
        total_count = total_count + 1

    print('\n---Total Size---')    
    print(total_count)

    data = []

    print("\n------Collecting Output------")

    p_count = 0

    
    
    for features in parsed_dataset.take(total_count): 

        #Extracting the ID of the image
        path = features['path']
        path_str = path.numpy().decode('utf-8')
        temp_pathv = path_str.split('/')
        imgid = temp_pathv[-1].split('.')[0] #ID of the image to keep track of the images
        p_count = p_count + 1

        #Extracting the ground truth value for that certain image
        gt_value = np.asarray([[features['dot_xcam'].numpy(),features['dot_y_cam'].numpy()]]) #ground truth value

        temp_img = features['image'].numpy()

        image_vis = Image.fromarray(temp_img)

        fw,fh = image_vis.size

        lx1, ly1, lx2, ly2 = int(features['leye_x1']), int(features['leye_y1']), int(features['leye_x2']), int(features['leye_y2'])
        rx1, ry1, rx2, ry2 = int(features['reye_x1']), int(features['reye_y1']), int(features['reye_x2']), int(features['reye_y2'])


        # calculating eye centres
        lx_c_point = (lx1 + lx2)//2
        ly_c_point = (ly1 + ly2)//2

        rx_c_point = (rx1 + rx2)//2
        ry_c_point = (ry1 + ry2)//2

        eye_dist = (np.linalg.norm(np.array((lx_c_point,ly_c_point))-np.array((rx_c_point,ry_c_point))))  ##undivide later if neccessary
        width = eye_dist  ### divide by 2, check with suresh
        a = width//2

        lx = lx_c_point-a
        ly = ly_c_point-a
        lw1 = lx_c_point+a
        lh1 = ly_c_point+a

        rx = rx_c_point-a
        ry = ry_c_point-a
        rw1 = rx_c_point+a
        rh1 = ry_c_point+a

        width = a*2

        #left eye crop
        # l_eye = image_vis.crop((lx_c_point-a, ly_c_point-a, lx_c_point+a, ly_c_point+a))
        l_eye = image_vis.crop((lx, ly, lw1, lh1))
        l_eye = l_eye.transpose(Image.FLIP_LEFT_RIGHT)

        #right eye crop
        r_eye = image_vis.crop((rx, ry, rw1, rh1))

        lw,lh=l_eye.size
        rw,rh=r_eye.size

    #     #normalizing and formatting bb coordinates according to google 
         ##bottom left version
    #    ly = ly + lh
    #    ry = ry + rh

        left_bbx = normalize_bbco([lx1, ly1, lx2, ly2],fw,fh)
        right_bbx = normalize_bbco([rx1, ry1, rx2, ry2],fw,fh)        

    #     #extracting bb coordinates as tensors
        left_bb = tf.convert_to_tensor(left_bbx,tf.float32) #left bounding box
        right_bb = tf.convert_to_tensor(right_bbx,tf.float32) #right bounding box 


    #     # image based modifications
        r_eye = np.asarray(r_eye)
        l_eye = np.asarray(l_eye)

    #     #resizing to 128x128
        r_eye = tf.convert_to_tensor(r_eye,tf.float32)
        r_eye_resized = tf.image.resize(r_eye, [128,128])    
        l_eye = tf.convert_to_tensor(l_eye,tf.float32)
        l_eye_resized = tf.image.resize(l_eye, [128,128]) 

    #     #normailizing the image channel wise
        r_eye_res_np = np.asarray(r_eye_resized)
        l_eye_res_np = np.asarray(l_eye_resized)    

    #     #right eye image normalization
        mean_one = np.mean(r_eye_res_np[:,:,0])
        mean_two = np.mean(r_eye_res_np[:,:,1])
        mean_three = np.mean(r_eye_res_np[:,:,2])

        std_one = np.std(r_eye_res_np[:,:,0])
        std_two = np.std(r_eye_res_np[:,:,1])
        std_three = np.std(r_eye_res_np[:,:,2])    

        x_a = r_eye_res_np[:,:,0] - mean_one
        x_b = r_eye_res_np[:,:,1] - mean_two
        x_c = r_eye_res_np[:,:,2] - mean_three    

        x_fin_a = x_a/std_one
        x_fin_b = x_b/std_two
        x_fin_c = x_c/std_three

        r_eye_final_np = np.stack((x_fin_a,x_fin_b,x_fin_c),axis=2) #final numpy array of right eye after resizing and normalization

    #     print('mean and std dev first channel right eye')
    #     print(np.mean(x_fin_a))
    #     print(np.std(x_fin_a))


    #     #left eye image normalization
        mean_one = np.mean(l_eye_res_np[:,:,0])
        mean_two = np.mean(l_eye_res_np[:,:,1])
        mean_three = np.mean(l_eye_res_np[:,:,2])

        std_one = np.std(l_eye_res_np[:,:,0])
        std_two = np.std(l_eye_res_np[:,:,1])
        std_three = np.std(l_eye_res_np[:,:,2])    

        x_a = l_eye_res_np[:,:,0] - mean_one
        x_b = l_eye_res_np[:,:,1] - mean_two
        x_c = l_eye_res_np[:,:,2] - mean_three    

        x_fin_a = x_a/std_one
        x_fin_b = x_b/std_two
        x_fin_c = x_c/std_three

        l_eye_final_np = np.stack((x_fin_a,x_fin_b,x_fin_c),axis=2) #final numpy array of left eye after resizing and normalization

        # print('\nmean and std dev first channel left eye')
        # print(np.mean(x_fin_a))
        # print(np.std(x_fin_a))




        r_eye_t = tf.convert_to_tensor(r_eye_final_np,tf.float32)
        l_eye_t = tf.convert_to_tensor(l_eye_final_np,tf.float32)  

        l_eye_final = tf.expand_dims(l_eye_t, 0) #final left eye as tensor - [1,128,128,3]  
        r_eye_final = tf.expand_dims(r_eye_t, 0)  #final right eye as tensor - [1,128,128,3]  


        # print('\nfinal data types and shapes')
        # print(r_eye_final.shape)
        # print(l_eye_final.dtype)
        # print(right_bb.dtype)
        # print(left_bb.dtype)
        # print(left_bb)


        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    #     print('---Output Details---')
    #     print(output_details)
    #     print('------')

        interpreter.set_tensor(input_details[0]['index'], right_bb)
        interpreter.set_tensor(input_details[1]['index'], left_bb)
        interpreter.set_tensor(input_details[2]['index'], r_eye_final)
        interpreter.set_tensor(input_details[3]['index'], l_eye_final)
        interpreter.invoke()


        output_data = interpreter.get_tensor(output_details[0]['index'])

        mod_output = np.asarray(output_data)

        data.append([imgid,[mod_output],[gt_value]])

        if(p_count%5000==0):
            print(p_count,' outputs obtained')


    print("\nData Collection Complete")
    print("\nCreating CSV and storing data in it")
    df=pd.DataFrame(data,columns=['Image_ID','Model_Output','GT_Value'])
    path_csv = '/home/skrishna/projects/def-skrishna/shared/v5_10individuals_data.csv'
    df.to_csv(path_csv, index = False)
