import argparse
import os
import time
import h5py

import torch
import torch.backends.cudnn as cudnn
import models
from torch import nn
from tqdm import tqdm
import csv

import torchvision.transforms as transforms
import flow_transforms
from imageio import imread
import numpy as np
import sys
from pathlib import Path

subset = 'mediaEval16_test_set'

folder = sys.argv[1]
RGB_stack_size = int(sys.argv[2])
del sys.argv[2]
del sys.argv[1]

current_directory = Path(os.path.abspath(os.path.join(__file__, os.pardir)))
vimotion = Path(current_directory).parent.parent

extension = "" if RGB_stack_size >= 64 else "_extension"

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

current_directory = os.path.abspath(os.path.join(__file__, os.pardir))
vimotion = Path(current_directory).parent.parent

file_path = os.path.join(vimotion, 'static/frames' + folder)

parser = argparse.ArgumentParser(description='PyTorch FlowNet inference on a folder of img pairs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', default=file_path, help='path to images folder')
parser.add_argument('--csvpath', default=file_path)
parser.add_argument('--pretrained', metavar='PTH', default = os.path.join(current_directory, 'flownets_EPE1.951.pth.tar'),
                    help='path to pre-trained model')
parser.add_argument('--output', metavar='DIR', default=file_path, help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--bidirectional', action='store_true', help='if set, will output invert flow (from 1 to 0) along with regular flow')

print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()

    data_dir = args.data
    print("=> fetching img pairs in '{}'".format(data_dir))
    if args.output is None:
        save_path = args.data/'flow_'+subset
    else:
        save_path = args.output # Path(args.output)
    print('=> will save everything to {}'.format(save_path))

    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    # create model
    network_data = torch.load(args.pretrained, map_location=device)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).to(device)
    model.eval()
    #
    # Use encoder (from the first conv layer to conv6_1) of FlowNetS as a feature extractor # added by Thao
    for param in model.parameters():
        param.requires_grad = False
    modules = list(model.children())[0:9]
    feature_extractor = nn.Sequential(*modules)
    #
    cudnn.benchmark = True

    # List of RGB images
    img_files = []
    csv_file = 'frames.csv' if extension == "" else 'extension.csv'
    with open(os.path.join(args.csvpath, csv_file),'r') as csvfile:
        csvReader = csv.reader(csvfile)
        for row in csvReader:
            temp = row[0]
            img_files.append(temp)
    num_files =len(img_files)
    num_stacks = int(num_files/RGB_stack_size)
    print('Num_stacks: ', num_stacks)

    start_time = time.time()
    if num_files < 2:
        print('Check the image folder')
    else:
        all_features = []
        num_movie = 0
        for stackID in range(0, num_stacks):

            clip_all_features = []

            count = 0
            start_time = time.time()
            for k in range(int(stackID * RGB_stack_size), int((stackID + 1) * RGB_stack_size) - 1):
                im1_fn = img_files[k]
                im2_fn = img_files[k + 1]

                # Extract OF features
                img1 = input_transform(imread(os.path.join(data_dir, im1_fn)))
                img2 = input_transform(imread(os.path.join(data_dir, im2_fn)))
                input_var = torch.cat([img1, img2]).unsqueeze(0)
                #
                if args.bidirectional:
                    # feed inverted pair along with normal pair
                    inverted_input_var = torch.cat([img2, img1]).unsqueeze(0)
                    input_var = torch.cat([input_var, inverted_input_var])
                #
                input_var = input_var.to(device)
                # compute feature map
                output = feature_extractor(input_var)

                # output is a feature map of [1, 1024, 6 ,8] => apply AvgPooling2D to get [1, 1024]-feature vector
                create_AP2d = nn.AvgPool2d((output.size(2), output.size(3)), stride=None)
                feature_vector = create_AP2d(output).view(output.size(0), output.size(1))  # feature_vector is a [1, 1024] vector
                # from tensor to numpy
                feature_vector = np.float32(feature_vector.cpu().detach().numpy())

                #
                clip_all_features.append(feature_vector)
                #

                #im1_fn = im2_fn
                #
                count += 1
                if (k == int((stackID + 1) * RGB_stack_size) - 2):
                    clip_all_features.append(feature_vector)
                    num_movie += 1
                    print('Processed', num_movie, ' movie excerpts')
                    print('Running time for a movie excerpt: ', time.time()-start_time)
                    start_time = time.time()
                    break

            temp = np.stack(clip_all_features, axis=0)
            temp = temp.squeeze(1)
            avg_clip_features_8_seg = np.mean(temp, axis=0)

            # Create a list of feature vectors
            all_features.append(avg_clip_features_8_seg)

        all_features = np.stack(all_features, axis=0)

        print('Number of feature vectors: ', all_features.shape)

        # save i3d_features in a .h5 file
        start_time = time.time()
        filename = os.path.join(args.output, 'FlowNetS_features' + extension + '.h5')
        h5file = h5py.File(filename, mode='w')
        h5file.create_dataset('data', data=np.array(all_features, dtype=np.float32))
        h5file.close()
        print("Time for writing feature vectors in .h5 file: ", time.time() - start_time)

if __name__ == '__main__':
    print("#####Extracting features with FlowNetS#####")
    main()
