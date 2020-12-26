import os
import argparse
from pytorch_i3d_N_C import InceptionI3d
import torch
from torchvision import transforms
import gc
import subprocess
import numpy as np
from keyframes_data_loader_RGB import centerKeyframeDataset
import h5py
import time
import sys
from pathlib import Path

# Memory check
def memoryCheck():
    ps = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv'], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    print(ps.communicate(), '\n')
    os.system("free -m")

# Free memory
def freeCacheMemory():
    torch.cuda.empty_cache()
    gc.collect()

def run_pretrained_I3D():
    # Device configuration
    use_cuda = torch.cuda.is_available()
    # Manual seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device: ', device)

    # Image preprocessing: data augmentation, normalization for the pretrained i3d
    Transform_RGB = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Call the dataset
    img_datasets = centerKeyframeDataset(args.img_csvfile, args.img_folder, Transform_RGB)

    # Build data loader
    img_dataloader = torch.utils.data.DataLoader(dataset=img_datasets, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)  # Note: shuffle=False
    # num_worker = 8

    # Model
    i3d_rgb = InceptionI3d(400, in_channels=3)
    i3d_rgb.load_state_dict(torch.load(args.load_model))
    i3d_rgb.to(device)

    img_features = []
    count = 0
    print('Extracting features....')
    for images in img_dataloader: # images are a stack of keyframes
        # Set mini-batch dataset
        inputs = images.to(device)
        #print(inputs.shape)

        # get the inputs of the size of batch x C x T x H x W
        inputs = inputs.unsqueeze(2)
        inputs = inputs.permute(2,1,0,3,4)

        # extract feature vector with the size of 1x 1024 from a stack of images using i3d pretrained model
        stack_features = i3d_rgb.extract_features(inputs)  #torch.squeeze(i3d_rgb.extract_features(inputs))
        # Tensor to array
        stack_features = np.float32(stack_features.cpu().detach().numpy())

        img_features.append(stack_features.squeeze(0))  # list of feature vectors
        count += 1
        print('Number of processed movies: ', count)

        # Free memory - Delete images
        del inputs, stack_features
        freeCacheMemory()

    # Convert a list of tubes into an array
    img_features = np.asarray(img_features)

    return img_features

if __name__ == '__main__':
    print("#####Extracting features with I3D-RGB#####")
    folder = sys.argv[1]
    bz = int(sys.argv[2])
    del sys.argv[2]
    del sys.argv[1]

    current_directory = Path(os.path.abspath(os.path.join(__file__, os.pardir)))
    vimotion = Path(current_directory).parent.parent

    extension = "" if bz >= 64 else "_extension"
    print('Extract features from stacks of RGB images')
    file_path = os.path.join(vimotion, 'static/frames' + folder)
    pretrained_path = os.path.join(current_directory, 'models') # path to pretrained model

    parser = argparse.ArgumentParser()
    csv_file = 'frames.csv' if extension == "" else 'extension.csv'
    parser.add_argument('--img_csvfile', type=str, default=os.path.join(file_path, csv_file), help='the .csv file containing image file names')
    parser.add_argument('--img_folder', type=str, default=file_path, help='the folder containing images')
    #parser.add_argument('--crop_size', type=int, default=224, help='crop size of each image')
    parser.add_argument('--batch_size', type=int, default=bz, help='batch size is also number of frames per stack')
    parser.add_argument('--root', type=str, default=file_path,
                        help='directory containing image files and .csv file of file names')
    parser.add_argument('--load_model', type=str, default=os.path.join(pretrained_path, 'rgb_imagenet.pt'),
                        help='.pt file containing pretrained weights of i3d model')
    parser.add_argument('--save_dir', type=str, default=file_path, help='directory for output saving')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()

    start_time = time.time()
    i3d_features = run_pretrained_I3D()
    print("Running time for feature extraction: ", time.time()-start_time)

    start_time = time.time()
    # save i3d_features in a .h5 file
    filename = os.path.join(file_path, 'I3D_RGB_features' + extension + '.h5')
    h5file = h5py.File(filename,mode='w')
    h5file.create_dataset('data', data=np.array(i3d_features, dtype=np.float32))
    h5file.close()
    print("Time for writing feature vectors in .h5 file: ", time.time()-start_time)
