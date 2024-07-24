import argparse, glob, os, torch, warnings, time
from tools import *
from trainer import *
from dataLoader import *

parser = argparse.ArgumentParser(description = "FAME")

### Training setting
parser.add_argument('--max_epoch',  type=int,   default=100,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=400,      help='Batch size')
parser.add_argument('--frame_len',  type=int,   default=300,      help='Input length of utterance')
parser.add_argument('--n_cpu',      type=int,   default=12,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,        help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,    help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,     help='Learning rate decay every [test_step] epochs')

### Data path
parser.add_argument('--train_list', type=str,   default="",       help='The path of the training list')
parser.add_argument('--test_list_English',  type=str,   default="lists/English_test_final.txt", help='The path of the evaluation list, English')
parser.add_argument('--test_list_Other',  type=str,   default="lists/Urdu_test_final.txt",   help='The path of the evaluation list, U or H')
parser.add_argument('--test_path',    type=str,   default="/home/user/data08/FAME/V1-Test",       help='The path of the evaluation data')
parser.add_argument('--save_path',    type=str,    default="", help='Path to save the clean list')

parser.add_argument('--pretrain_s',    type=str,    default="ecapa", help='Path to save the clean list')
parser.add_argument('--pretrain_f',    type=str,    default="face18", help='Path to save the clean list')

### Initial modal path
parser.add_argument('--initial_model',  type=str,   default="",  help='Path of the initial_model')
parser.add_argument('--EU',  type=str,   default="EU",  help='Path of the initial_model')
parser.add_argument('--eval',    dest='eval', action='store_true', help='Do evaluation')

## Init folders
args = init_system(parser.parse_args())
## Init loader
args = init_loader(args)
## Init trainer
s = init_trainer(args)

## Evaluate only
if args.eval == True:
	if 'E' in args.EU:
		s.test_network(args, type = 'English')
	if 'U' in args.EU:
		s.test_network(args, type = 'Other')
	quit()

## Train only
while args.epoch < args.max_epoch:
	args = init_loader(args)
	s.train_network(args)
	if args.epoch % args.test_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%args.epoch)
		if 'E' in args.EU:
			s.test_network(args, type = 'English')
		if 'U' in args.EU:
			s.test_network(args, type = 'Other')
	args.epoch += 1
quit()