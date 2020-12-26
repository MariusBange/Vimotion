import subprocess
import numpy as np
import argparse
import torch, torch.utils.data
from torchvision import transforms
from keyframes_data_loader import centerKeyframeDataset
from pretrained_ResNet50 import preResNet50
import os
import time
import gc
from collections.abc import Mapping, Container
from sys import getsizeof
import sys
import h5py
from pathlib import Path

def deep_getsizeof(o, ids):
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, np.unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r

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

# Extract features from all movies
def extractFeature(featureModel, img_dataloaders, device):
    img_features = []   # features extracted from images from each movie
    #
    count = 0
    print('Extract features from frames.....')
    for images in img_dataloaders:
        # Set mini-batch dataset (inputs and targets)
        images = images.to(device)
        #
        all_segments = featureModel(images)
        all_segments = all_segments.mean(dim=0)
        all_segments = all_segments.unsqueeze(0)
        img_features.append(all_segments)
        count += 1
        print('Number of processed movies: ', count)
        # Free memory - Delete images
        del images
        freeCacheMemory()

    output = np.asarray(torch.cat(img_features, dim=0).to('cpu'))
    return output

# Main
def main(args):
    # Device configuration
    use_cuda =  torch.cuda.is_available()
    print('Check cuda: ', use_cuda)
    # Manual seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device: ', device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Image preprocessing: data augmentation, normalization for the pretrained resnet
    Transform_RGB = transforms.Compose([
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    # Call the dataset
    img_datasets = centerKeyframeDataset(img_csvfiles, file_path, Transform_RGB)

    # Build data loader
    img_dataloader = torch.utils.data.DataLoader(dataset=img_datasets, batch_size=args.batch_size,
                                                     shuffle=False, **kwargs)  # Note: shuffle=False
    #
    # memoryCheck()
    model = preResNet50().to(device)

    # Extract features
    start_time = time.time()
    img_features_all = extractFeature(model, img_dataloader, device)
    print('Running time for feature extraction for all movie clips:', time.time() - start_time, ' seconds')

    # Save extracted features, valence list, arousal list
    save_file_name = 'ResNet50_RGB_features' + extension + '.h5'
    h5file = h5py.File(os.path.join(vimotion, 'static/frames' + folder, save_file_name), mode='w')
    h5file.create_dataset("data", data=img_features_all, dtype=np.float32)
    h5file.close()


    # Read h5py file
    h5file = h5py.File(os.path.join(vimotion, 'static/frames' + folder, save_file_name), 'r')
    getData = h5file.get("data")
    dataArray = np.array(getData)
    features = torch.from_numpy(dataArray)  # .to(device)  # Convert numpy arrays to tensors on gpu
    h5file.close()


if __name__ == "__main__":
    print("#####Extracting features with ResNet50#####")
    folder = sys.argv[1]
    num_keyframes = int(sys.argv[2])
    del sys.argv[2]
    del sys.argv[1]

    current_directory = Path(os.path.abspath(os.path.join(__file__, os.pardir)))
    vimotion = Path(current_directory).parent.parent

    extension = "" if num_keyframes >= 64 else "_extension"
    dir_path = current_directory.parent
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224, help='size for cropping images')
    parser.add_argument('--batch_size', type=int, default=num_keyframes, help = 'number of frames used for feature extraction each time')
    #parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training') # Note: torch version 1.1.0
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()
    print(args)

    # ------------------------------------------------------------------------------------------------------------------
    file_path = os.path.join(vimotion, 'static/frames' + folder)  # folders of keyframes 64_keyframes_
    img_csvfiles = os.path.join(file_path, "frames.csv") if extension == "" else os.path.join(file_path, "extension.csv") # .csv files containing image names

    #-------------------------------------------------------------------------------------------------------------------
    main_start_time = time.time()
    main(args)
    print('Total running time: {:.5f} seconds' .format(time.time() - main_start_time))
