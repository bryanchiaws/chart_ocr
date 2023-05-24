import pandas as pd
import argparse
import pdb
import os
import numpy as np
import glob
import tqdm
from PIL import Image
import numpy as np
import cv2
import imutils
import torch
import shutil
import json
import io
import sys
import nltk
import ast
nltk.download('words')
from nltk.corpus import words
correct_words = set(words.words())
from nltk.metrics.distance  import edit_distance
import re

import gc
gc.collect()
torch.cuda.empty_cache()

from DeepRule.test_pipe_type_cloud_mod import DeepRule as DR
from CRAFT_pytorch.test_craft_mod import CRAFTPytorch as CR
from CRAFT_pytorch.file_utils import saveResult
from CRAFT_pytorch.imgproc import loadImage

#Code adapted from https://towardsdatascience.com/pytorch-scene-text-detection-and-recognition-by-craft-and-a-four-stage-network-ec814d39db05

def straighten_manual(bbox, img):
    #Code from https://theailearner.com/tag/cv2-warpperspective/

    pt_A, pt_D, pt_C, pt_B = bbox
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],[0, maxHeight - 1],[maxWidth - 1, maxHeight - 1], [maxWidth - 1, 0]]) 
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out = cv2.warpPerspective(img, M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)

    if maxHeight > maxWidth*1.3: #Need to rotate 90 degrees
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    return out

def subset_inside(inner_box, outer_box):
    #Both boxes are a list: [[top left x, top left y], [bottom right x, bottom right y]]

    midpoint_x = (inner_box[0][0] + inner_box[1][0])/2
    midpoint_y = (inner_box[0][1] + inner_box[1][1])/2

    return (midpoint_x>=outer_box[0][0])*(midpoint_y>=outer_box[0][1])*\
    (midpoint_x<=outer_box[1][0])*(midpoint_y<=outer_box[1][1]) 

def generate_words(image, bboxes, image_folder, chart_type):

    if chart_type == "horizontal":
        image = imutils.rotate_bound(image, -90)

    for key in bboxes.keys():
        pts = np.array(bboxes[key]).astype(int)
        if np.all(pts) > 0:
            #Straighten out image and crop
            word = straighten_manual(bboxes[key], image)            
            bboxes[key] = {'bbox': bboxes[key], 'path': os.path.join(image_folder, f"{key}.png"), 'img': word}
            cv2.imwrite(os.path.join(image_folder, f"{key}.png"), word)

    return bboxes

def bar_math(data, cls_info, pred_dict, bboxes, chart_type, exclusions, threshold = 0.8):
    #Get max value for plot area
    plot_area = cls_info[5]
    all_area = cls_info[4]
    plot_xmin, plot_ymin, plot_xmax, plot_ymax, _ = plot_area

    # Old Method: Find max y value
    # mindist = 1e9
    # minkey = None
    # for i in range(len(bboxes.keys())):
    #     topright = bboxes[str(i)]['bbox'][2]
    #     bottomright = bboxes[str(i)]['bbox'][3]
        
    #     xval = xmin if chart_type == "vertical" else xmax
    #     #Calculate word with min dist
    #     dist = (topright[0] - xval)**2 + (topright[1] - ymax)**2 + (bottomright[0] - xval)**2 + (bottomright[1] - ymax)**2
    #     if dist < mindist:
    #         minkey = i
    #         mindist = dist

    # ymax_val = float("".join([x for x in pred_dict[str(minkey)]['pred'] if x.isdigit() or x == "."]))

    #New Method: Find y value with the most confidence
    plotbox = [[plot_area[0], plot_area[1]], [plot_area[2], plot_area[3]]]
    mainbox = [[all_area[0], all_area[1]], [all_area[2], all_area[3]]]

    max_conf = 0
    ymax_val = None
    data = sorted(data, key = lambda x: x[0])

    for i in range(len(pred_dict.keys())):

        try:
            candidate = float("".join([x for x in pred_dict[str(i)]['pred'] if x.isdigit() or x == "."]))
        except:
            continue

        #First look at axes
        if not subset_inside([bboxes[str(i)]['bbox'][0], bboxes[str(i)]['bbox'][2]], plotbox):
            if subset_inside([bboxes[str(i)]['bbox'][0], bboxes[str(i)]['bbox'][2]], mainbox):
                #If vertical only look at left of plot, if horizontal only look at right of plot
                topleft, _, bottomright, _ = bboxes[str(i)]['bbox'].tolist()
                predx, predy = (topleft[0] + bottomright[0]) /2, (topleft[1] + bottomright[1]) /2
                if chart_type == "vertical" and predx > plot_xmin:
                    continue
                elif chart_type == "horizontal" and predx < plot_xmax:
                    continue
                if pred_dict[str(i)]['conf'] > max_conf:
                    max_conf = pred_dict[str(i)]['conf']
                    ymax_val = candidate
                    ymin, ymax = predy, plot_ymax

        else:
            if pred_dict[str(i)]['conf'] > max_conf:
                max_conf = pred_dict[str(i)]['conf']
                ymax_val = candidate
                #Find bar with minimum distance to word
                topleft, _, bottomright, _ = bboxes[str(i)]['bbox'].tolist()
                predx, predy = (topleft[0] + bottomright[0]) /2, (topleft[1] + bottomright[1]) /2
                mindist = 1e9
                ymin, ymax = None, None
                for bar in data:
                    bar_xmin, bar_ymin, bar_xmax, bar_ymax = bar
                    dist = (bar_xmin - predx)**2 + (bar_ymin - predy)**2 + (bar_xmax - predx)**2 + (bar_ymax - predy)**2
                    if dist < mindist:
                        ymin, ymax = bar_ymin, bar_ymax
                        mindist = dist

    bar_values = {}
    for i in range(len(data)):
        #Data: [xmin, ymin, xmax, ymax]
        bar_values[i] = {"bbox": data[i], "value": round(ymax_val*(data[i][3]-data[i][1])/(ymax-ymin), 2)}

    #Can choose main box as the area between plot and vizualization area
    #Both boxes are a list: [[top left x, top left y], [bottom right x, bottom right y]]
    #mainbox = [[all_area[0], plot_area[3]], [all_area[2], all_area[3]]]
    #Or just choose main box as entire visualization area

    #Filter to words in between plot areas
    labels = {}
    context = {}
    for i in range(len(pred_dict.keys())):
        # [[top left x, top left y], [bottom right x, bottom right y]]
        if subset_inside([bboxes[str(i)]['bbox'][0], bboxes[str(i)]['bbox'][2]], mainbox):
            if not subset_inside([bboxes[str(i)]['bbox'][0], bboxes[str(i)]['bbox'][2]], plotbox):
                topleft, topright, bottomright, bottomleft = bboxes[str(i)]['bbox'].tolist()
                if topright[1] < plot_ymax:
                    continue
                bbox = bboxes[str(i)]['bbox'].tolist()
                labels[str(i)] = {'bbox': bbox, 'word': pred_dict[str(i)]['pred'], 'conf': pred_dict[str(i)]['conf']}
        else:
            if not sum([bool(re.search(x, pred_dict[str(i)]['pred'].lower())) for x in exclusions]):
                bbox = bboxes[str(i)]['bbox'].tolist()
                context[str(i)] = {'bbox': bbox, 'word': pred_dict[str(i)]['pred'], 'conf': pred_dict[str(i)]['conf']}
                
    return labels, bar_values, context

def word_corrector(word):
    temp = [(edit_distance(word, w),w) for w in correct_words if w[0]==word[0]]
    return sorted(temp, key = lambda val:val[0])[0][1]

if __name__ == '__main__':
    # usage: main.py [data visualizations folder][output_folder] [chart classification file]
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs_folder')
    parser.add_argument('output_path')
    parser.add_argument('chart_class')
    parser.add_argument('--pytesseract', action = 'store_false')
    parser.add_argument('--exclusions', default = "['statista 2021', 'additional information', 'show source']")

    #CRAFT Arguments
    parser.add_argument('--trained_model', default='CRAFT_pytorch/weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.3, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.3, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.5, type=float, help='link confidence threshold')
    parser.add_argument('--poly', action='store_true', help='enable polygon type')
    parser.add_argument('--refine', action='store_false', help='enable link refiner')
    parser.add_argument('--refiner_model', default='CRAFT_pytorch/weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--show_time', action='store_false', help='show processing time')
    
    #Deep Text Recognition Args
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default="TPS", help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default ="ResNet", help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default="Attn", help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    args = parser.parse_args()
    deeprule_pipe = DR()
    craft_pipe = CR(args.trained_model, args.refine, args.refiner_model, args.show_time)

    if not args.pytesseract:
        from deep_text_recognition_benchmark.demo_mod import TextRecognition as DTR
        dtr_pipe = DTR(args.workers, args.batch_size, args.batch_max_length, args.imgH, args.imgW,\
        args.rgb, args.character, args.sensitive, args.PAD, args.Transformation, args.FeatureExtraction,\
        args.SequenceModeling, args.Prediction, args.num_fiducial, args.input_channel, \
        args.output_channel, args.hidden_size, args)
    else:
        import pytesseract

    if not os.path.exists(os.path.join(args.output_path, 'kp_extract')):
        os.makedirs(os.path.join(args.output_path, 'kp_extract'))

    images = sorted(os.listdir(args.inputs_folder))

    chart_class = pd.read_csv(args.chart_class)
    chart_class_dict = {k.split("/")[-1]: v for k, v in zip(chart_class["filename"], chart_class["pred"])}

    #Create logging file
    logfile = open(args.output_path + '/logging_file.txt', 'w')
    kp_files = os.listdir(os.path.join(args.output_path, 'kp_extract'))

    # See if it has already been processed
    #images = [x for x in images if x not in kp_files]
    images = [x for x in images if "mc_" not in x]
    images = [x for x in images if "pew_" not in x]

    for image_path in tqdm.tqdm(images):

        path = os.path.join(args.inputs_folder, image_path)

        chart_type = chart_class_dict[image_path]
        if chart_type != "horizontal" and chart_type != "vertical":
            continue

        # Keypoint Detection
        try:
            rotation = 90 if chart_type == "horizontal" else 0
            text_trap = io.StringIO()
            sys.stdout = text_trap
            kp_image, data, cls_info = deeprule_pipe.test(path, rotation)
            sys.stdout = sys.__stdout__
            kp_image.save(os.path.join(args.output_path, 'kp_extract', image_path))
        except Exception as e:
            logfile.write(f'{image_path}: Keypoint detection issue ({e})')
            continue

        # Word detection
        try:
            kp_image = loadImage(os.path.join(args.output_path, 'kp_extract', image_path))
            bboxes, polys, score_text = craft_pipe.test_net(kp_image, args.text_threshold, args.link_threshold, args.low_text, args.poly, args.canvas_size, args.mag_ratio)
            bboxes = {str(i): x for i, x in enumerate(bboxes)}
        
            # Save score text from word detection
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            if os.path.exists(os.path.join(args.output_path, 'word_extract', filename)):
                shutil.rmtree(os.path.join(args.output_path, 'word_extract', filename))
            os.makedirs(os.path.join(args.output_path, 'word_extract', filename))

            result_folder = os.path.join(args.output_path, 'word_extract', filename)
            saveResult(image_path, kp_image[:,:,::-1], polys, dirname=result_folder + "/")

            #Crop images and save
            cropped_image = cv2.imread(os.path.join(args.inputs_folder, filename+file_ext))
            if not os.path.exists(os.path.join(args.output_path, 'word_extract', filename, 'cropped')):
                os.makedirs(os.path.join(args.output_path, 'word_extract', filename, 'cropped'))
            image_folder = os.path.join(args.output_path, 'word_extract', filename, 'cropped')
            
            bboxes_final = generate_words(cropped_image, bboxes, image_folder, chart_type)

        except Exception as e:
            logfile.write(f'{image_path}: Word detection issue ({e})')
            continue

        #Recognize cropped images
        if args.pytesseract:
            whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz.%$()\\'*/,-& "
            pred_dict = {}

            log = open(f'{"/".join(image_folder.split("/")[:-1])}/_log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            for k, v in bboxes_final.items():
                try:
                    db = cv2.resize(v["img"], None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
                    word = pytesseract.image_to_data(db, config=f'-c tessedit_char_whitelist="{whitelist}" --psm 6', output_type='dict')

                    word_str = " ".join([x for x in word['text'] if x != ''])
                    try:
                        sys.stdout = text_trap
                        confs = np.array([x for x in word['conf'] if x != "-1"]).mean()
                        sys.stdout = sys.__stdout__
                    except:
                        confs = 0

                    pred_dict[str(k)] = {"img_name": v["path"], "pred": word_str.strip(), "conf": confs/100}
                    log.write(f'{v["path"]:25s}\t{word_str:25s}\t{confs/100:0.4f}\n')
                except:
                    pred_dict[str(k)] = {"img_name": v["path"], "pred": "", "conf": 0}
                    continue
            
            log.close()

        else:
            try:
                pred_dict = dtr_pipe.demo(image_folder)
            except Exception as e:
                logfile.write(f'{image_path}: Word recognition and cropping issue ({e})')
                continue

        #Get label values
        try:
            exclusions = set(ast.literal_eval(args.exclusions))
            labels, bar_values, context = bar_math(data, cls_info, pred_dict, bboxes_final, chart_type, exclusions)
            if not os.path.exists(os.path.join(args.output_path, 'labels', filename)):
                os.makedirs(os.path.join(args.output_path, 'labels', filename))
            label_folder = os.path.join(args.output_path, 'labels', filename)

            with open(label_folder + "/label.json", "w") as fp:
                json.dump(labels,fp) 

            with open(label_folder + "/values.json", "w") as fp:
                json.dump(bar_values,fp)

            with open(label_folder + "/context.json", "w") as fp:
                json.dump(context,fp)

        except Exception as e:
            logfile.write(f'{image_path}: Value determination/bar math issue ({e})')
            continue
    
    logfile.close()
    print("ALL DONE") 