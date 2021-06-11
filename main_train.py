from torch import Tensor
from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
from PIL import Image

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = [] #CJ : noise
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        segmap = Image.open("Input/Images/doge_father_seg.bmp").convert('L')
        segmap = np.asarray(segmap)
        #print(segmap.shape)

        segmap = torch.Tensor(segmap).cuda()
        #torch.Size([1, 3, 166, 250])[b,c,H,W]
        #print(segmap.size())
       
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp, segmap)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt, segmap)
