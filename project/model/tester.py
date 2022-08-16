import torch
import torch.nn as nn
import torchaudio
import torchvision.models as models



class TestEncoder(nn.Module):
    def __init__(self, backbone, dim=2048, pred_dim=512, sample_rate=16000, n_fft=512, f_min=0.0, f_max=8000.0, n_mels=96):                 
        '''
        #--소개--#
        Encoder의 출력만 필요한 tSNE에서 사용함.
        기존 모델은 z,p,z,p 4개를 출력하므로 부적합
        #--파라미터--#
        dim: projection의 hidden fc dimension
        pred_dim: predictor의 hidden dimension
        '''
        super().__init__()
        self.backbone = backbone

        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, 
                                                        f_min=f_min, f_max=f_max, n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)
        
        if backbone in ['resnet50','resnet101','resnet152']:
            self.encoder = models.__dict__[backbone](zero_init_residual=True,pretrained=False,num_classes=dim) # encoder: backbone + projector
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

            # encoder의 projector를 3layer MLP로 구성
            prev_dim = self.encoder.fc.weight.shape[-1] # [num_classes, 2048]
            self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                                nn.BatchNorm1d(prev_dim),
                                                nn.ReLU(inplace=True), #First Layer
                                                
                                                nn.Linear(prev_dim, prev_dim, bias=False),
                                                nn.BatchNorm1d(prev_dim),
                                                nn.ReLU(inplace=True), #Second Layer
                                                
                                                self.encoder.fc,
                                                #    nn.Linear(prev_dim, dim, bias=False),
                                                #    nn.BatchNorm1d(prev_dim),
                                                #    nn.ReLU(inplace=True), #Third Layer
                                                
                                                nn.BatchNorm1d(dim, affine=False) #Output Layer
                                                )
            self.encoder.fc[6].bias.requires_grad = False

        
    def forward(self, x):
        # x shape: [B, 1, 48000]
        x = self.spec(x)   #[B, 1, 96, 188] # 왜인지몰라도 tuple로 나온다
        x = x[0]
        x = self.to_db(x)   #[B, 1, 96, 188]
        x = self.spec_bn(x) #[B, 1, 96, 188]
   
        z = self.encoder(x) # [B, dim]
        return z