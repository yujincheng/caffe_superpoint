import torch
import torchvision
import numpy as np
import cv2

class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
    # self.unsample = torch.nn.UpsamplingBilinear2d(scale_factor=8)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    # desc_all = torch.nn.functional.upsample_bilinear(desc,scale_factor=8)
    # dna = torch.norm(desc_all, p=2, dim=1) # Compute the norm.
    # desc_all = desc_all.div(torch.unsqueeze(dna, 1))
    dense = torch.nn.functional.softmax(semi,dim=1)
    nodust = dense[:,:-1,:,:]
    Hc,Wc = nodust.shape[2:4]
    nodust = torch.transpose(nodust,1, 2)
    nodust = torch.transpose(nodust,2, 3)
    heatmap = torch.reshape(nodust, (1,Hc, Wc, 8, 8) )
    heatmap = torch.transpose(heatmap, 2, 3)
    heatmap = torch.reshape(heatmap, (1,1,Hc*8, Wc*8) )
    desc_reshape = torch.reshape(desc, (1,-1,Hc*8, Wc*8) )
    Cout = torch.cat((heatmap,desc_reshape),1)
    return Cout


def nms_fast(in_corners, H, W, dist_thresh):
  """
  Run a faster approximate Non-Max-Suppression on numpy corners shaped:
    3xN [x_i,y_i,conf_i]^T

  Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
  are zeros. Iterate through all the 1's and convert them either to -1 or 0.
  Suppress points by setting nearby values to 0.

  Grid Value Legend:
  -1 : Kept.
    0 : Empty or suppressed.
    1 : To be processed (converted to either kept or supressed).

  NOTE: The NMS first rounds points to integers, so NMS distance might not
  be exactly dist_thresh. It also assumes points are within image boundaries.

  Inputs
    in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
    H - Image height.
    W - Image width.
    dist_thresh - Distance to suppress, measured as an infinty norm distance.
  Returns
    nmsed_corners - 3xN numpy matrix with surviving corners.
    nmsed_inds - N length numpy vector with surviving corner indices.
  """
  grid = np.zeros((H, W)).astype(int) # Track NMS data.
  inds = np.zeros((H, W)).astype(int) # Store indices of points.
  # Sort by confidence and round to nearest int.
  inds1 = np.argsort(-in_corners[2,:])
  corners = in_corners[:,inds1]
  rcorners = corners[:2,:].round().astype(int) # Rounded corners.
  # Check for edge case of 0 or 1 corners.
  if rcorners.shape[1] == 0:
    return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
  if rcorners.shape[1] == 1:
    out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
    return out, np.zeros((1)).astype(int)
  # Initialize the grid.
  for i, rc in enumerate(rcorners.T):
    grid[rcorners[1,i], rcorners[0,i]] = 1
    inds[rcorners[1,i], rcorners[0,i]] = i
  # Pad the border of the grid, so that we can NMS points near the border.
  pad = dist_thresh
  grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
  # Iterate through points, highest to lowest conf, suppress neighborhood.
  count = 0
  for i, rc in enumerate(rcorners.T):
    # Account for top and left padding.
    pt = (rc[0]+pad, rc[1]+pad)
    if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
      grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
      grid[pt[1], pt[0]] = -1
      count += 1
  # Get all surviving -1's and return sorted array of remaining corners.
  keepy, keepx = np.where(grid==-1)
  keepy, keepx = keepy - pad, keepx - pad
  inds_keep = inds[keepy, keepx]
  out = corners[:, inds_keep]
  values = out[-1, :]
  inds2 = np.argsort(-values)
  out = out[:, inds2]
  out_inds = inds1[inds_keep[inds2]]
  return out, out_inds

if __name__ == '__main__':

  grayim = cv2.imread('14.png',0)
  ori_size = grayim.shape
  interp = cv2.INTER_AREA
  print(ori_size)
  grayim = cv2.resize(grayim, ( 640 , 480 ), interpolation=interp)
  print(grayim.shape)
  sizeratio = (ori_size[0]/480 , ori_size[1]/640)
  # sizeratio = (1,1)
  print(sizeratio)
  grayim = (grayim.astype('float32') / 255.)
  grayim.tofile("14.qwe.1")
  assert grayim.ndim == 2, 'Image must be grayscale.'
  assert grayim.dtype == np.float32, 'Image must be float32.'
  H, W = grayim.shape[0], grayim.shape[1]
  print('{H}, {W}'.format(H=H,W=W))
  inp = grayim.copy()
  inp = (inp.reshape(1, H, W))
  inp = torch.from_numpy(inp)
  inp = torch.autograd.Variable(inp).view(1, 1, H, W)
  # fe.net.forward(grayim)
  print(inp)
  inp.numpy().tofile("inp.qwe")
  # inp.numpy().dump("inp.bin")
