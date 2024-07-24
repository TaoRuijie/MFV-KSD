import torch, sys, os, tqdm, numpy, soundfile, time, pickle, cv2, glob, random, scipy
import torch.nn as nn
from tools import *
from loss import *
from model.audio_ecapa import *
from model.visual_res import *
from model.fusion import *
from collections import defaultdict, OrderedDict
from torch.cuda.amp import autocast,GradScaler

def init_trainer(args):
	s = trainer(args)
	args.epoch = 1
	if args.initial_model != '':
		print("Model %s loaded from previous state!"%(args.initial_model))
		s.load_parameters(args.initial_model)
	elif len(args.modelfiles) >= 1:
		print("Model %s loaded from previous state!"%args.modelfiles[-1])
		args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles[-1]))[0][6:]) + 1
		s.load_parameters(args.modelfiles[-1])
	return s

class trainer(nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()

		self.speaker_encoder = ECAPA_TDNN().cuda()
		if args.pretrain_s != "":
			loadedState = torch.load('pretrain/%s.pt'%(args.pretrain_s), map_location="cuda")		
			
			selfState = self.speaker_encoder.state_dict()
			for name, param in loadedState.items():
				origName = name
				if name not in selfState:
					name = name.replace("speaker_encoder.", "")
					if name not in selfState:
						print("%s is not in the model."%origName)
						continue
				if selfState[name].size() != loadedState[origName].size():
					sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s\n"%(origName, selfState[name].size(), loadedState[origName].size()))
					continue
				selfState[name].copy_(param)

		self.face_encoder    = IResNet().cuda()	
		if args.pretrain_f != "":
			loadedState = torch.load('pretrain/%s.pt'%(args.pretrain_f), map_location="cuda")
			selfState = self.face_encoder.state_dict()
			for name, param in loadedState.items():
				origName = name
				if name not in selfState:
					print("%s is not in the model."%origName)
					continue
				if selfState[name].size() != loadedState[origName].size():
					sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
					continue
				selfState[name].copy_(param)

		self.fusion_module   = Fusion().cuda()
		self.loss            = Softmax(nOut = 256).cuda()		
		self.optim           = torch.optim.Adam(self.parameters(), lr = args.lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.test_step, gamma = args.lr_decay)
		print(" Speech model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1e6))
		print(" Face model para number = %.2f"%(sum(param.numel() for param in self.face_encoder.parameters()) / 1e6))
		

	def train_network(self, args):
		self.train()
		scaler = GradScaler()
		self.scheduler.step(args.epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		time_start = time.time()

		for num, (speech, face, labels) in enumerate(args.trainLoader, start = 1):
			self.zero_grad()
			labels      = torch.LongTensor(labels).cuda()	
			face        = face.div_(255).sub_(0.5).div_(0.5)
			with autocast():
				a_embedding   = self.speaker_encoder.forward(speech.cuda(), aug = False)	
				v_embedding   = self.face_encoder.forward(face.cuda())
				av_embedding  = self.fusion_module.forward(a_embedding, v_embedding)
				avloss, prec  = self.loss.forward(av_embedding, labels)	
			scaler.scale(avloss).backward()
			scaler.step(self.optim)
			scaler.update()

			index += len(labels)
			loss += avloss.detach().cpu().numpy()
			top1 += prec
			time_used = time.time() - time_start
			sys.stderr.write(" [%2d] %.2f%% (est %.1f mins) Lr: %5f, Loss: %.5f, ACC: %2.2f%% \r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, lr, loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")

		args.score_file.write("%d epoch, LR %f, LOSS %f ACC: %2.2f%% \n"%(args.epoch, lr, loss/num, top1/index*len(labels)))
		args.score_file.flush()
		return
	
	def test_network(self, args, type):
		self.eval()
		scores, labels = [], []
		if type == 'English':
			lines = open(args.test_list_English).read().splitlines()	
		elif type == 'Other':
			lines = open(args.test_list_Other).read().splitlines()	
		a = 1
		for line in tqdm.tqdm(lines):			
			with torch.no_grad():
				index, label, voice_path, face_path = line.split()[0], int(line.split()[1]), line.split()[2], line.split()[3]
				frame = cv2.imread(os.path.join(args.test_path, face_path))		
				face = cv2.resize(frame, (112, 112))
				face = torch.FloatTensor(numpy.transpose(face, (2, 0, 1))).unsqueeze(0)
				face = face.div_(255).sub_(0.5).div_(0.5)
				# Load voice
				path = os.path.join(args.test_path, voice_path).replace('voices', 'voices_key')
				speech, _ = soundfile.read(path)
				if len(speech.shape) == 2:
					speech = numpy.mean(speech, axis = 1)
				
				audio = torch.FloatTensor(numpy.array(speech)).unsqueeze(0)
				a_embedding   = self.speaker_encoder.forward(audio.cuda())
				v_embedding   = self.face_encoder.forward(face.cuda())
				av_embedding  = self.fusion_module.forward(a_embedding, v_embedding)
				score_1  = self.loss.forward(av_embedding)[0]

				for duration in [2]:
					max_audio = (duration+1) * 16000
					if speech.shape[0] <= max_audio:
						shortage = max_audio - audio.shape[0]
						speech = numpy.pad(speech, (0, shortage), 'wrap')
					feats = []
					for asf in range((len(speech) - (duration * 16000)) // 16000):
						feats.append(speech[asf * 16000: (asf + duration) * 16000])
					feats = numpy.stack(feats, axis = 0)#.astype(numpy.float)
					audio = torch.FloatTensor(feats).cuda()
					a_embedding   = self.speaker_encoder.forward(audio.cuda())				
					score_d = []
					for i in range(a_embedding.shape[0]):					
						av_embedding  = self.fusion_module.forward(a_embedding[i:i+1,:], v_embedding)
						score_d.append(self.loss.forward(av_embedding)[0])
					score_d = sum(score_d) / len(score_d)
					score_1 += score_d
					score = score_1 / 2
					scores.append(score)
					labels.append(label)

		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]		
		print('[%s] EER: %2.4f \n'%(type, EER))
		args.score_file.write("[%s] EER: %2.4f\n"%(type, EER))
		args.score_file.flush()
		return 

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		selfState = self.state_dict()
		loadedState = torch.load(path)
		for name, param in loadedState.items():
			origName = name
			if name not in selfState:
				print("%s is not in the model."%origName)
				continue
			if selfState[name].size() != loadedState[origName].size():
				sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
				continue
			
			selfState[name].copy_(param)