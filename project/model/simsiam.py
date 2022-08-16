import torch.nn as nn
import torchaudio
import torchvision.models as models
import torch



class Siamusic(nn.Module):
    def __init__(self, backbone, augmentation='basic', dim=2048, pred_dim=512, sr=22050, n_fft=512, f_min=0.0, f_max=8000.0, n_mels=48):  
        '''               
        backbone에 원하는 backbone 입력 ex) 'resnet' 
        dim: projected featrue의 dimension
        pred_dim: predictor의 hidden dimension  
        '''    
        super(Siamusic, self).__init__()
        assert backbone in ['resnet50','resnet101','resnet152']
        assert augmentation in ['basic','image']
        self.backbone = backbone
        self.augmentation = augmentation
        
        if augmentation == 'basic':
            self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, f_min=f_min, f_max=f_max, n_mels=n_mels)
            self.to_db = torchaudio.transforms.AmplitudeToDB()
            self.spec_bn = nn.BatchNorm2d(1)
        
        
        self.encoder = models.__dict__[backbone](zero_init_residual=True, pretrained=False, num_classes=dim) # encoder: backbone + projector
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False) #audio data는 채널이 1개이므로 input channel 변경

        # encoder의 projector를 3layer MLP로 구성
        prev_dim = self.encoder.fc.weight.shape[-1] # [num_classes, 2048]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # First Layer
                                            
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # Second Layer
                                            
                                        self.encoder.fc,                                               
                                        nn.BatchNorm1d(dim, affine=False) # Output Layer
                                        )

        self.encoder.fc[6].bias.requires_grad = False
        
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim) # ouput layer
                                       )

    def forward(self, x1, x2):
        if self.augmentation == 'basic': # x1, x2 : [B, 22050*5]
            x1, x2 = self.spec(x1), self.spec(x2) # [B, 48, 431]
            x1, x2 = self.to_db(x1), self.to_db(x2)     # [B, 48, 431]
            x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)   #[B, 1, 48, 431]
            x1, x2 = self.spec_bn(x1), self.spec_bn(x2) #[B, 1, 48, 431]
        # else: x1, x2 : [B, 1, 48, 431]

        z1 = self.encoder(x1)  # x1이 stop_gradient 쪽에 투입되는 경우 # [B, dim]
        z2 = self.encoder(x2)  # x2가 stop_gradient 쪽에 투입되는 경우 # [B, dim]
        
        p1 = self.predictor(z1) # x1이 predictor 쪽에 투입되는 경우 # [B, dim]
        p2 = self.predictor(z2) # x2가 predictor 쪽에 투입되는 경우 # [B, dim]
        return p1, z2.detach(), p2, z1.detach() # z1, z2는 stop_gradient


class Evaluator(nn.Module):
    def __init__(self, encoder, num_classes, augmentation='basic', dim=2048,
                 sample_rate=16000, n_fft=512, f_min=0.0, f_max=8000.0, n_mels=80):
        '''
        encoder : pretrained encoder
        num_classes : target task의 분류할 class 수
        augmentation : pretraining에서 사용된 augmentation
        dim : encoder의 output dimension
        '''
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, 
                                                        f_min=f_min, f_max=f_max, n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # 논문보니까 linear evaluation은 설계하기 나름인듯
        self.encoder = encoder
        self.augmentation = augmentation
        self.num_classes = num_classes
        self.evaluator = nn.Sequential(nn.Linear(dim,1024),
                                       nn.ReLU(),
                                       nn.Linear(1024,512),
                                       nn.ReLU(),
                                       nn.Linear(512,128),
                                       nn.ReLU(),
                                       nn.Linear(128,num_classes)
                                       )
    
    def forward(self,audio):
        # audio shape: [B, 1, 48000]
        
        audio = self.spec(audio) #[B, 80, 862]
        audio = self.to_db(audio) #[B, 80, 862]
        audio = audio.unsqueeze(1) #[B, 1, 80, 862]
        audio = self.spec_bn(audio) #[B, 1, 80, 862]
    
        logit = self.encoder(audio)
        logit = self.evaluator(logit)
        
        return logit




if __name__ == '__main__':
    x = torch.zeros((10,22050*5))
    model = Siamusic('resnet50')
    p1,z2,p2,z1 = model(x,x)