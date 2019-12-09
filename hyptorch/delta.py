import torch
import torchvision
from scipy.spatial import distance_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def delta_hyp(dismat):
    """
    computes delta hyperbolicity value from distance matrix
    """
    
    p = 0
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)
    
    maxmin = np.max(np.minimum(XY_p[:,:,None], 
                               XY_p[None,:,:]), axis=1)
    return np.max(maxmin - XY_p)



class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        B = x.shape[0]
        return x.view(B, -1)
    
    
def get_delta(loader):
    """
    computes delta value for image data by extracting features using VGG network;
    input -- data loader for images
    """
    vgg = torchvision.models.vgg16(pretrained=True);
    vgg_feats = vgg.features
    vgg_classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
    
    vgg_part = nn.Sequential(vgg_feats, 
                             Flatten(),
                             vgg_classifier).to(device)
    vgg_part.eval()
    
    all_features = []
    for i, (batch, _) in enumerate(loader):
        with torch.no_grad():
            batch = batch.to(device)
            all_features.append(vgg_part(batch).detach().cpu().numpy())
    
    all_features = np.concatenate(all_features)
    idx = np.random.choice(len(all_features), 1500)
    all_features_small = all_features[idx]
    
    dists = distance_matrix(all_features_small, all_features_small)
    delta = delta_hyp(dists)
    diam = np.max(dists)
    return delta, diam    
