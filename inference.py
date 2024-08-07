import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()

parser.add_argument('--features_dim', default='2048', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)


parser.add_argument('--num_f_maps', default='64', type=int)

# Need input
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_layers_PG', type=int)
parser.add_argument('--num_layers_R', type=int)
parser.add_argument('--num_R', type=int)

args = parser.parse_args()

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

# use the full temporal resolution @ 15fps
sample_rate = 1

mapping_file = "./data/gtea/mapping.txt"

model_dir = "./models/"+"gtea"+"/split_4"
results_dir = "./results/"+"gtea" +"/split_4"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
    
num_classes = len(actions_dict)
trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, "Inference", "Inference")

video_path = "./test/color.mp4"

trainer.inference(model_dir, results_dir, video_path, 100 ,actions_dict, device, sample_rate)
