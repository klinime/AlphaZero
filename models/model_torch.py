
import torch
from torch import nn, optim
from torchsummary import summary

class Residual(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size):
		super(Residual, self).__init__()
		self.upscale = None if in_planes == out_planes else \
			nn.Conv2d(in_planes, out_planes, 1, bias=False)
		self.conv1 = nn.Conv2d(
			in_planes, out_planes, kernel_size,
			padding=kernel_size//2, bias=False)
		self.relu1 = nn.ReLU(inplace=True)
		self.bn1 = nn.BatchNorm2d(out_planes)
		self.conv2 = nn.Conv2d(
			out_planes, out_planes, kernel_size,
			padding=kernel_size//2, bias=False)
		self.relu2 = nn.ReLU(inplace=True)
		self.bn2 = nn.BatchNorm2d(out_planes)

	def forward(self, x):
		out = self.bn1(self.relu1(self.conv1(x)))
		if self.upscale:
			x = self.upscale(x)
		return self.bn2(self.relu2(x + self.conv2(out)))

class Model(nn.Module):
	def __init__(self, n_layers, in_planes, out_planes,
				 head_planes, height, width, ac_dim):
		super(Model, self).__init__()
		self.res_layers = nn.Sequential(*([Residual(in_planes, out_planes, 3)] + \
			[Residual(out_planes, out_planes, 3) for _ in range(1, n_layers)]))
		self.p_conv = nn.Conv2d(out_planes, head_planes, 3, padding=1, bias=False)
		self.p_relu = nn.ReLU(inplace=True)
		self.p_bn = nn.BatchNorm2d(head_planes)
		self.p_flatten = nn.Flatten()
		self.policy = nn.Linear(height * width * head_planes, ac_dim)
		self.p_out = nn.Softmax(dim=1)
		self.v_conv = nn.Conv2d(out_planes, head_planes * 2, 3, padding=1, bias=False)
		self.v_relu = nn.ReLU(inplace=True)
		self.v_bn = nn.BatchNorm2d(head_planes * 2)
		self.v_flatten = nn.Flatten()
		self.v_linear = nn.Linear(height * width * head_planes * 2, 256)
		self.v_linrelu = nn.ReLU(inplace=True)
		self.value = nn.Linear(256, 1)
		self.v_out = nn.Tanh()

	def forward(self, x):
		x = self.res_layers(x)
		p = self.p_flatten(self.p_bn(self.p_relu(self.p_conv(x))))
		p = self.p_out(self.policy(p))
		v = self.v_flatten(self.v_bn(self.v_relu(self.v_conv(x))))
		v = self.v_out(self.value(self.v_linrelu(self.v_linear(v))))
		return p, v

class Agent():
	def __init__(self, path, n_layers, filters, head_filters, c,
				 height, width, ac_dim, depth, lr, td, device):
		self.path = path
		self.nnet = Model(
			n_layers, depth, filters, head_filters,
			height, width, ac_dim).to(device)
		self.p_loss = nn.CrossEntropyLoss()
		self.v_loss = nn.MSELoss()
		self.opt = optim.Adam(self.nnet.parameters(), lr=lr, weight_decay=c)
		self.device = device
		if device.type == 'cuda':
			self.scaler = torch.cuda.amp.GradScaler()
		self.td = td
		summary(self.nnet, input_size=(depth, height, width))

	def forward(self, s):
		ps, vs = self.nnet(torch.from_numpy(s).to(self.device))
		return ps.data.cpu().numpy().flatten(), vs.data.cpu().numpy().flatten()

	def update(self, s, pi, z):
		self.opt.zero_grad()
		ps, vs = self.nnet(torch.from_numpy(s.astype(np.float32)).to(self.device))
		p_loss = self.p_loss(ps, torch.from_numpy(pi).to(self.device))
		v_loss = self.v_loss(vs, torch.from_numpy(z.astype(np.float32)).to(self.device))
		if self.device.type == 'cuda':
			self.scaler.scale(p_loss).backward(retain_graph=True)
			self.scaler.scale(v_loss).backward()
			self.scaler.step(self.opt)
			self.scaler.update()
		else:
			p_loss.backward()
			v_loss.backward()
			self.opt.step()
		return np.mean([p_loss.item(), v_loss.item()])

	def save(self, i):
		file = '{}/{:03d}/model{}.pth'.format(self.path, i,
			'_td' if self.td else '')
		torch.save({
			'model': self.nnet.state_dict(),
			'opt': self.opt.state_dict(),
		}, file)
		print('Model saved.')
		
	def load(self, i):
		file = '{}/{:03d}/model{}.pth'.format(self.path, i,
			'_td' if self.td else '')
		checkpoint = torch.load(file)
		self.nnet.load_state_dict(checkpoint['model'])
		self.opt.load_state_dict(checkpoint['opt'])
		print('Model loaded.')
