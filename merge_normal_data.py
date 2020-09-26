import os, fnmatch, shutil, subprocess
import regex as re
import io
import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
import pydicom as dicom
from scipy.ndimage import rotate
from skimage import exposure
import cv2
import glob

SAX_SERIES = {
    # challenge training
    "SC-HF-I-1": "0004",
    "SC-HF-I-2": "0106",
    "SC-HF-I-4": "0116",
    "SC-HF-I-10": "0024",
    "SC-HF-I-40": "0134",
    "SC-HF-NI-3": "0379",
    "SC-HF-NI-4": "0501",
    "SC-HF-NI-34": "0446",
    "SC-HF-NI-36": "0474",
    "SC-HYP-1": "0550",
    "SC-HYP-3": "0650",
    "SC-HYP-38": "0734",
    "SC-HYP-40": "0755",
    "SC-N-2": "0898",
    "SC-N-3": "0915",
    "SC-N-40": "0944",
    "SC-HF-I-5": "0156",
    "SC-HF-I-6": "0180",
    "SC-HF-I-7": "0209",
    "SC-HF-I-8": "0226",
    "SC-HF-NI-11": "0270",
    "SC-HF-NI-31": "0401",
    "SC-HF-NI-33":"0424",
    "SC-HF-NI-7": "0523",
    "SC-HYP-37": "0702",
    "SC-HYP-6": "0767",
    "SC-HYP-7": "0007",
    "SC-HYP-8": "0796",
    "SC-N-5": "0963",
    "SC-N-6": "0981",
    "SC-N-7": "1009",
    "SC-HF-I-11": "0043",
    "SC-HF-I-12": "0062",
    "SC-HF-I-9": "0241",
    "SC-HF-NI-12": "0286",
    "SC-HF-NI-13": "0304",
    "SC-HF-NI-14": "0331",
    "SC-HF-NI-15": "0359",
    "SC-HYP-10": "0579",
    "SC-HYP-11": "0601",
    "SC-HYP-12": "0629",
    "SC-HYP-9": "0003",
    "SC-N-10": "0851",
    "SC-N-11": "0878",
    "SC-N-9": "1031"
}
SUNNYBROOK_ROOT_PATH = "D:\\shu\\project\\data\\Sunnybrook_data\\"

TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            "Sunnybrook Cardiac MR Database ContoursPart3",
                            "TrainingDataContours")
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            "Sunnybrook Cardiac MR Database DICOMPart3",
                            'TrainingDataDICOM')

ONLINE_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            "Sunnybrook Cardiac MR Database ContoursPart1",
                            "OnlineDataContours")
VAL_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            "Sunnybrook Cardiac MR Database ContoursPart2",
                            "ValidationDataContours")

VAL_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        "Sunnybrook Cardiac MR Database DICOMPart2",
                        "ValidationDataDICOM")
ONLINE_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        "Sunnybrook Cardiac MR Database DICOMPart1", 
                        "OnlineDataDICOM")


def shrink_case(case):
    toks = case.split("-")
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt).rjust(2, '0')
        except ValueError:
            return x
    return "-".join([shrink_if_number(t) for t in toks])

class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r"\\([^\\]*)\\contours-manual\\IRCCI-expert\\IM-0001-(\d{4})-icontour-manual.txt", ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))
    
    def __str__(self):
        return "<Contour for case %s, image %d>" % (self.case, self.img_no)
    
    __repr__ = __str__

def convert_to_grayscale_with_increase_brightness_fast(im,rate):
    """
    im=image(2d),don't normalize
    incr=scale number
    """
    nmin = np.min(im.astype(float))
    nmax = np.max(im.astype(float))
    if (nmin==nmax) :
        out = im
    else:
        out = (im-nmin)/(nmax-nmin)
    out_mean = np.sum(out)/(np.sum((out>0).astype(float))) #chemical potential
    out[out <= (out_mean*rate)] = out_mean #Energy Alignment
    out=1 /( 1 + np.exp(out_mean-out) )
    ans=((out - np.min(out[out > 0])) / (np.max(out[out > 0])-np.min(out[out > 0])))
    ans=(ans*255).astype(np.uint8)
    return ans

def get_square_crop(img,lab):
    lx,ly = img.shape
    xm,ym = lx//2,ly//2
    xm = int(np.mean(xm))
    ym = int(np.mean(ym))
    delta = 128;#cut middle 256*256 from manual data
    img = img[xm-delta:xm+delta,ym-delta:ym+delta]
    lab = lab[xm-delta:xm+delta,ym-delta:ym+delta]
    return img,lab

def getAlignImg(t,label = None):#!!!notice, only take uint8 type for the imrotate function!!!
    rows= t.Rows
    col = t.Columns
    f = lambda x:np.asarray([float(a) for a in x])
    o = f(t.ImageOrientationPatient)
    o1 = o[:3]
    o2 = o[3:]
    oh = np.cross(o1,o2)
    or1 = np.asarray([0.6,0.6,-0.2])
    o2new = np.cross(oh,or1)
    theta = np.arccos(np.dot(o2,o2new)/np.sqrt(np.sum(o2**2)*np.sum(o2new**2)))*180/3.1416
    theta = theta * np.sign(np.dot(oh,np.cross(o2,o2new)))
    res = np.array(t.pixel_array,dtype=np.float)
    res = rotate(res,theta,reshape=False)
    if label is None:
        return res
    else:
        lab = rotate(label,theta,reshape=False)
        return res,lab

def load_contour(contour, img_path,save_dir):
    filename = "IM-0001-%04d.dcm" % (contour.img_no)
    full_path = os.path.join(img_path, contour.case,'DICOM', filename)
    f = dicom.read_file(full_path)
    ds = f.pixel_array.astype(np.int)
    ctrs = np.loadtxt(contour.ctr_path, delimiter=" ").astype(np.int)
    label = np.zeros_like(ds, dtype="uint8")
    cv2.fillPoly(label, [ctrs], 255)
    img,lab = getAlignImg(f,label)  
    img=convert_to_grayscale_with_increase_brightness_fast(img,1)
    img=cv2.resize(img,(176,176))
    lab=cv2.resize(lab,(176,176))
    SAVE_DIR_img=os.path.join(save_dir,'img\\')
    SAVE_DIR_gt=os.path.join(save_dir,'gt\\')
    if not os.path.exists(SAVE_DIR_img):
        os.makedirs(SAVE_DIR_img)
    if not os.path.exists(SAVE_DIR_gt):
        os.makedirs(SAVE_DIR_gt)
    cv2.imwrite(SAVE_DIR_img + filename.replace(".dcm", ".png"),img)
    cv2.imwrite(SAVE_DIR_gt +filename.replace(".dcm", "_gt.png"),lab)

def get_all_contours(contour_path):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt')]
    print("Shuffle data")
    np.random.shuffle(contours)
    print("Number of examples: {:d}".format(len(contours)))
    return contours

def change_countour(countour_path):
    gt=cv2.imread(countour_path,cv2.IMREAD_GRAYSCALE)
    gt = (np.where(gt>0,0,255)).astype('uint8')
    return gt


def load_manual_data(fdir):
    im_dir=fdir + '\\images\\*.jpg'
    g_dir=fdir + '\\contours\\*.jpg'
    im_list=glob.glob(im_dir)
    g_list=glob.glob(g_dir)
    indice=np.arange(len(im_list))
    np.random.shuffle(indice)
    split_indice=len(im_list)*0.8
    test=indice[int(split_indice):]
    train=indice[:int(split_indice)]
    for i in test:
        img=cv2.imread(im_list[i],cv2.IMREAD_GRAYSCALE)
        g=change_countour(g_list[i])
        img=convert_to_grayscale_with_increase_brightness_fast(img,1)
        g=cv2.resize(g,(176,176))
        img=cv2.resize(img,(176,176))
        cv2.imwrite(r'D:\\shu\\project\\data\\test\\gt\\'+str(i)+'_gt.png',g)
        cv2.imwrite(r'D:\\shu\\project\\data\\test\\img\\'+str(i)+'.png',img)
    
    for i in train:
        img=cv2.imread(im_list[i],cv2.IMREAD_GRAYSCALE)
        g=change_countour(g_list[i])
        img=convert_to_grayscale_with_increase_brightness_fast(img,1)
        g=cv2.resize(g,(176,176))
        img=cv2.resize(img,(176,176))
        cv2.imwrite(r'D:\\shu\\project\\data\\train\\gt\\'+str(i)+'_gt.png',g)
        cv2.imwrite(r'D:\\shu\\project\\data\\train\\img\\'+str(i)+'.png',img)

def load_manual_julia_data(img,gt):
    indice=np.arange(len(img))
    np.random.shuffle(indice)
    split_indice=len(img)*0.8
    split_indice1=len(img)*0.9
    test=indice[int(split_indice1):]
    val=indice[int(split_indice):int(split_indice1)]
    train=indice[:int(split_indice)]
    for i in test:
        filename=(os.path.basename(img[i])).replace(".png", "")
        im=convert_to_grayscale_with_increase_brightness_fast(cv2.imread(img[i],cv2.IMREAD_GRAYSCALE),1)
        im=cv2.resize(im,(176,176))
        g=cv2.imread(gt[i],cv2.IMREAD_GRAYSCALE)
        g = (np.where(g>0,255,0)).astype('uint8')
        g=cv2.resize(g,(176,176))
        cv2.imwrite(r'D:\\shu\\project\\data\\test\\gt\\'+filename+'_gt.png',g)
        cv2.imwrite(r'D:\\shu\\project\\data\\test\\img\\'+filename+'.png',im)
    for i in train:
        filename=(os.path.basename(img[i])).replace(".png", "")
        im=convert_to_grayscale_with_increase_brightness_fast(cv2.imread(img[i],cv2.IMREAD_GRAYSCALE),1)
        im=cv2.resize(im,(176,176))
        g=cv2.imread(gt[i],cv2.IMREAD_GRAYSCALE)
        g = (np.where(g>0,255,0)).astype('uint8')
        g=cv2.resize(g,(176,176))
        cv2.imwrite(r'D:\\shu\\project\\data\\train\\gt\\'+filename+'_gt.png',g)
        cv2.imwrite(r'D:\\shu\\project\\data\\train\\img\\'+filename+'.png',im)
    for i in val:
        filename=(os.path.basename(img[i])).replace(".png", "")
        im=convert_to_grayscale_with_increase_brightness_fast(cv2.imread(img[i],cv2.IMREAD_GRAYSCALE),1)
        im=cv2.resize(im,(176,176))
        g=cv2.imread(gt[i],cv2.IMREAD_GRAYSCALE)
        g = (np.where(g>0,255,0)).astype('uint8')
        g=cv2.resize(g,(176,176))
        cv2.imwrite(r'D:\\shu\\project\\data\\validate\\gt\\'+filename+'_gt.png',g)
        cv2.imwrite(r'D:\\shu\\project\\data\\validate\\img\\'+filename+'.png',im)

def read_data_fullpath(target_dir, search_pattern):
    """
    read images and labels's full_dir (in list)
    """
    img=[]
    gt=[]
    #open list
    path=glob.glob(target_dir + search_pattern) 
    for index in range(len(path)-1):
        if index %2==0:
            img.append(path[index])
        else:
            gt.append(path[index])     
    return img, gt

if __name__ == "__main__":
    print("Mapping ground truth contours to images...")
    foldertype=['train',"validate", "test"]
    save_dirs=r'D:\\shu\\project\\data\\'
    for folder in foldertype:
        save_dir=save_dirs + folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if folder == 'test':
            gt_dir = ONLINE_CONTOUR_PATH
            img_dir = ONLINE_IMG_PATH
        elif folder == 'train':
            gt_dir = TRAIN_CONTOUR_PATH
            img_dir = TRAIN_IMG_PATH
        else:
            gt_dir = VAL_CONTOUR_PATH
            img_dir = VAL_IMG_PATH

        ctrs = get_all_contours(gt_dir)
        for ctr in ctrs:
            con=Contour(ctr)
            load_contour(con, img_dir,save_dir)
    mdirs=r'D:\\shu\\project\\data\\manual_data\\'
    foldertype=['manual_contours','manual_contours_ch4']
    for mdir in foldertype:
        fdir = mdirs + mdir
        load_manual_data(fdir)

    print("write julia manual images...")
    mdirs=r'D:\\shu\\project\\data\\data_segmenter_trainset\\'
    img,gt = read_data_fullpath(mdirs,search_pattern='*.png')
    load_manual_julia_data(img,gt)
