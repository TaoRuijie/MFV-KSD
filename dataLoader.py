import glob, numpy, os, random, soundfile, torch, cv2, wave
from scipy import signal
import torchvision.transforms as transforms
from collections import defaultdict

def init_loader(args):
	trainloader = train_loader(**vars(args))
	args.trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
	return args

class train_loader(object):
	def __init__(self, train_list, frame_len, **kwargs):
		self.frame_len = frame_len * 160 + 240

		self.voice_dic = defaultdict(list)
		lines = open(train_list).read().splitlines()
		self.face_list = []
		for line in lines:
			data = line.split()
			Modality, id, path = data
			if Modality == 'Voice':
				self.voice_dic[id].append(path)
			elif Modality == 'Face':
				self.face_list.append([id, path])

	def __getitem__(self, index):
		face_id, face_path = self.face_list[index]
		type = random.choice(['Pos', 'Neg'])
		if type == 'Neg':
			voice_id = random.choice(list(set(self.voice_dic.keys()) - set(face_id)))
			voice_path = random.choice(self.voice_dic[voice_id])
			label = 0
		elif type == 'Pos':
			voice_id = face_id
			voice_path = random.choice(self.voice_dic[voice_id])
			label = 1
		segments = self.load_wav(file = voice_path)
		segments = torch.FloatTensor(numpy.array(segments))
		faces    = self.load_face(file = face_path)
		faces = torch.FloatTensor(numpy.array(faces))
		return segments, faces, label

	def load_wav(self, file):
		utterance, _ = soundfile.read(file.replace('voices', 'voices_key'))
		if utterance.shape[0] < 16000:
			utterance, _ = soundfile.read(file)
		if len(utterance.shape) == 2:
			utterance = numpy.mean(utterance, axis = 1)
		if utterance.shape[0] <= self.frame_len + 1:
			shortage = self.frame_len + 1 - utterance.shape[0]
			utterance = numpy.pad(utterance, (0, shortage), 'wrap')
		startframe = random.choice(range(0, utterance.shape[0] - (self.frame_len)))
		segment = numpy.expand_dims(numpy.array(utterance[int(startframe):int(startframe)+self.frame_len]), axis = 0)
		return segment[0]

	def load_face(self, file):
		frame = file
		frame = cv2.imread(frame)			
		face = cv2.resize(frame, (112, 112))	
		face = numpy.transpose(face, (2, 0, 1))
		return face

	def __len__(self):
		return len(self.face_list)