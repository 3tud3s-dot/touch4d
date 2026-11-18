import torch
import torch.nn as nn
from mmcv.runner import BaseModule, ModuleList


def computeBlendingRT(B0, R, T, W):
    '''
    :param B0: [numV, numB, 3]
    :param R: [numV, numB, 3, 3]
    :param T: [numV, numB, 3]
    :param W: [numV, numB, 1]
    :return:
    '''
    GR = W.unsqueeze(-1) * R
    GR = torch.sum(GR, 1)  # [numV, 3, 3]

    I = torch.eye(3).to(R.get_device())
    GT = I.unsqueeze(0).unsqueeze(0) - R  # [numV, numB, 3, 3]
    GT = torch.matmul(GT, B0.unsqueeze(-1))  # [numV, numB, 3, 1]
    GT = T + GT.squeeze(-1)  # [numV, numB, 3]
    GT = W * GT  # [numV, numB, 3]
    GT = torch.sum(GT, 1)  # [numV, 3]

    return GR, GT


def transformGX(x, R, T):
    '''
    :param x: [numV, 3]
    :param R: [numV, 3, 3]
    :param T: [numV, 3]
    :return:
    '''
    x = torch.matmul(R, x.unsqueeze(-1)).squeeze(-1) # [numV, 3]
    x = x + T
    return x


def batch_DeformGeo(CG, B0, BR, BT, BW, bsize, numV):
    out = torch.zeros(bsize, numV, 3).cuda()
    for b in range(bsize):
        G = CG[b, :, :]
        R = BR[b, :, :, :]
        R = R.unsqueeze(0).repeat(numV, 1, 1, 1)  # [numV, numB, 3, 3]
        T = BT[b, :, :]
        T = T.unsqueeze(0).repeat(numV, 1, 1)  # [numV, numB, 3]
        W = BW[b, :, :]
        W = W.unsqueeze(-1)

        GR, GT = computeBlendingRT(B0[b], R, T, W)
        iG = transformGX(G, GR, GT)

        out[b, :, :] = iG

    return out


def Global_GeoDeform(DD, B0, BR, BT, GW, bsize, numV):
    GG = batch_DeformGeo(CG=DD, B0=B0, BR=BR, BT=BT, BW=GW, bsize=bsize, numV=numV)
    return GG
    
def calcW(pa, radiuseSigma):
    '''
    :param pa: [b, N, k]
    :param radiuseSigma: [b, k]
    :return: [b, N, k]
    '''

    rr = radiuseSigma.unsqueeze(1)  # [bs, 1, k]
    ra = -(pa**2)/(2.*rr**2)
    W = torch.softmax(ra, dim=-1)

    return W

def projectD(GTV, GBV, GNV, D):
    dx = D[:, 0].unsqueeze(-1)  # [numV, 1]
    dy = D[:, 1].unsqueeze(-1)
    dz = D[:, 2].unsqueeze(-1)
    Dp = dx * GTV + dy * GBV + dz * GNV # [numV, 3]
    return Dp


def batchProjD(GTV, GBV, GNV, D, bsize):
    dd = torch.zeros_like(D).cuda()
    for b in range(bsize):
        d = D[b, :, :]
        tv = GTV[b, :, :]
        bv = GBV[b, :, :]
        nv = GNV[b, :, :]
        dd[b, :, :] = projectD(tv, bv, nv, d)
    return dd


def Local_GeoDeform(G0, GD, G0TV, G0BV, G0NV, bsize):
    resDD = batchProjD(GTV=G0TV, GBV=G0BV, GNV=G0NV, D=GD, bsize=bsize)
    DD = G0 + resDD
    return DD

def SampleMap(MapImg, MIndex, MEffi):
    '''
    MapImg [H, W, dim]
    MIndex [numV, 4, 2]
    MEffi [numV, 4]
    '''
    p0 = MapImg[MIndex[:, 0, 1], MIndex[:, 0, 0], :].unsqueeze(1)
    p1 = MapImg[MIndex[:, 1, 1], MIndex[:, 1, 0], :].unsqueeze(1)
    p2 = MapImg[MIndex[:, 2, 1], MIndex[:, 2, 0], :].unsqueeze(1)
    p3 = MapImg[MIndex[:, 3, 1], MIndex[:, 3, 0], :].unsqueeze(1)

    pp = torch.cat([p0, p1, p2, p3], dim=1)

    p = MEffi.unsqueeze(-1) * pp
    p = torch.sum(p, 1)
    return p

class LinearLayer(BaseModule):
    def __init__(self, in_ch, out_ch, k_relu=0.01, ifNorm=False):
        super(LinearLayer, self).__init__()
        if k_relu > 0:
            self.FC = nn.Sequential(nn.Linear(in_ch, out_ch), nn.LeakyReLU(k_relu, inplace=True))
        else:
            self.FC = nn.Sequential(nn.Linear(in_ch, out_ch), nn.ReLU(inplace=True))

        if ifNorm:
            self.midlayerNorm = nn.LayerNorm(out_ch, elementwise_affine=False)
        self.ifNorm = ifNorm

    def forward(self, x, ifLinear = False):
        if ifLinear:
            x = self.FC[0](x)
        else:
            x = self.FC(x)
        if self.ifNorm:
            x = self.midlayerNorm(x)

        return x

class CNN2dLayer(BaseModule):
    def __init__(self, in_ch, out_ch, k_size, stride, padding, k_relu=0.2):
        super(CNN2dLayer, self).__init__()
        if k_relu < 0:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, k_size, stride=stride, padding=padding),
                                      nn.InstanceNorm2d(out_ch),
                                      nn.ReLU())
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, k_size, stride=stride, padding=padding),
                                      nn.InstanceNorm2d(out_ch),
                                      nn.LeakyReLU(k_relu, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class CNNTrans2dLayer(BaseModule):
    def __init__(self, in_ch, out_ch, k_size, stride, padding, k_relu=0.2):
        super(CNNTrans2dLayer, self).__init__()
        if k_relu < 0:
            self.transConv = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, k_size, stride=stride, padding=padding),
                                           nn.InstanceNorm2d(out_ch),
                                           nn.ReLU())
        else:
            self.transConv = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, k_size, stride=stride, padding=padding),
                                           nn.InstanceNorm2d(out_ch),
                                           nn.LeakyReLU(k_relu, inplace=True))

    def forward(self, x):
        x = self.transConv(x)
        return x

class Res_MLP(BaseModule):
    def __init__(self, in_ch):
        super(Res_MLP, self).__init__()
        self.MLP = LinearLayer(in_ch, in_ch)

    def forward(self, x):
        x = x + self.MLP(x)
        return x

class Dis_MLP(BaseModule):
    def __init__(self, in_ch, out_ch):
        super(Dis_MLP, self).__init__()

        self.MLP_B = nn.Sequential(LinearLayer(in_ch, 64),
                                   Res_MLP(64), Res_MLP(64), Res_MLP(64),
                                   Res_MLP(64))
        self.LinearGeo = nn.Sequential(LinearLayer(64+in_ch, 64), nn.Linear(64, out_ch))

    def forward(self, x):
        z = self.MLP_B(x)
        vp = self.LinearGeo(torch.cat([z, x], dim=-1))
        return vp
        
class RelativePosEncoder(BaseModule):
    def __init__(self, in_ch, out_ch=128):
        super(RelativePosEncoder, self).__init__()
        self.encoder_1 = nn.Sequential(CNN2dLayer(in_ch, 512, 3, 1, 1),
                                       CNN2dLayer(512, 256, 3, 1, 1),
                                       CNN2dLayer(256, 128, 3, 1, 1),
                                       CNN2dLayer(128, out_ch, 3, 1, 1))  # 128 --> 128

    def forward(self, x):
        x = self.encoder_1(x)
        return x

class DisEncoder(BaseModule):
    def __init__(self, in_ch, out_ch=1024, midSize=64):
        super(DisEncoder, self).__init__()
        self.encoder_1 = nn.Sequential(CNN2dLayer(in_ch, 128, 3, 2, 1),  # 64
                                       CNN2dLayer(128, 256, 3, 2, 1),  # 32
                                       CNN2dLayer(256, 512, 3, 2, 1),  # 16
                                       CNN2dLayer(512, 1024, 3, 2, 1),  # 8
                                       CNN2dLayer(1024, 1024, 3, 2, 1),  # 4
                                       CNN2dLayer(1024, out_ch, 3, 2, 1))  # 2

        self.in_ZLinear = nn.Linear(4*out_ch, midSize)

    def forward(self, x):
        x = self.encoder_1(x)

        b, c, h, w = x.size()
        z = torch.flatten(x, 1, -1)  # [b, 4*1024]
        z = self.in_ZLinear(z)

        return z, b, c, h, w

class DisDecoder(BaseModule):
    def __init__(self, in_ch, out_ch):
        super(DisDecoder, self).__init__()
        self.decoder1 = nn.Sequential(CNNTrans2dLayer(in_ch, 1024, 4, 2, 1),  # 4
                                      CNNTrans2dLayer(1024, 1024, 4, 2, 1),  # 8
                                      CNNTrans2dLayer(1024, 512, 4, 2, 1),  # 16
                                      CNNTrans2dLayer(512, 256, 4, 2, 1),  # 32
                                      CNNTrans2dLayer(256, 128, 4, 2, 1),  # 64
                                      CNNTrans2dLayer(128, 64, 4, 2, 1),  # 128
                                      CNNTrans2dLayer(64, out_ch, 4, 2, 1))  # 256

    def forward(self, x):
        x = self.decoder1(x)
        return x

class Pred_decoder(BaseModule):
    def __init__(self, in_ch, midSize, uv_ch, out_ch):
        super(Pred_decoder, self).__init__()
        #midSize = 512
        #midSize = 64
        #self.in_ZLinear = nn.Linear(in_ch * 4, midSize)
        self.out_ZLinear = LinearLayer(midSize, in_ch * 4)

        self.decoder = DisDecoder(in_ch, 64)

        self.uvencoder = nn.Sequential(LinearLayer(uv_ch, 64), LinearLayer(64, 64),
                                       LinearLayer(64, 64), LinearLayer(64, 64))

        self.MLP_pred = Dis_MLP(128, out_ch)
        self.out_ch = out_ch

    def forward(self, midz, uv, gMIndex, gMEffi, b, c, h, w):
        z = self.out_ZLinear(midz)
        # # b, 1024, 2, 2
        z = z.view(b, c, h, w)

        z = self.decoder(z)
        # H*W=256*256
        z = z.permute(0, 2, 3, 1)  # [bsize, H, W, dim]

        fuv = self.uvencoder(uv)

        out_vp = []
        for bi in range(b):
            bz = SampleMap(z[bi, :, :, :], gMIndex[bi], gMEffi[bi])
            # # bz is the decode concat to uv
            bz = torch.cat([bz, fuv[bi]], dim=-1)
            vp = self.MLP_pred(bz)
            out_vp.append(vp.unsqueeze(0))

        out_vp = torch.cat(out_vp, dim=0)
        return out_vp

class DynamicDeltaEncoder(BaseModule):
    def __init__(self, in_ch, v_ch, out_ch=128):
        super(DynamicDeltaEncoder, self).__init__()
        self.encoder_1 = nn.Sequential(CNN2dLayer(in_ch, 512, 3, 1, 1),
                                       CNN2dLayer(512, 256, 3, 1, 1),
                                       CNN2dLayer(256, 128, 3, 1, 1))  # 128 --> 128
        self.encoder_2 = nn.Sequential(CNN2dLayer(128+v_ch, 128, 3, 1, 1),
                                       CNN2dLayer(128, 128, 3, 1, 1),
                                       CNN2dLayer(128, out_ch, 3, 1, 1))

    def forward(self, f, v):
        x = self.encoder_1(f)
        x = self.encoder_2(torch.cat([x, v], dim=1))
        return x
        