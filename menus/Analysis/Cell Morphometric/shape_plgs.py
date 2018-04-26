from .static import *
from imagepy.core.engine import Tool
from imagepy import IPy
from imagepy.core.engine import Simple, Filter
from imagepy.core.manager import WindowsManager
from imagepy.core.roi.pointroi import PointRoi
from imagepy import ImagePlus
import time
class Mark:
    def __init__(self, data):
        self.data = data

    def draw(self, dc, f, **key):
        dc.SetPen(wx.Pen((255,255,0), width=1, style=wx.SOLID))
        dc.SetTextForeground((255,255,0))
        font = wx.Font(8, wx.FONTFAMILY_DEFAULT, 
                       wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False)
        
        dc.SetFont(font)
        data = self.data[0 if len(self.data)==1 else key['cur']]
        for i in range(len(data)):
            pos = f(*(data[i][0][1], data[i][0][0]))
            dc.DrawCircle(pos[0], pos[1], 2)
            dc.DrawText('id={}'.format(i), pos[0], pos[1])
            if data[i][1]==None:continue
            k1, k2, a = data[i][1]
            aixs = np.array([[-np.sin(a), np.cos(a)],
                             [np.cos(a), np.sin(a)]])*[k1/2, k2/2]
            ar = np.linspace(0, np.pi*2,25)
            xy = np.vstack((np.cos(ar), np.sin(ar)))
            arr = np.dot(aixs, xy).T+data[i][0]
            dc.DrawLines([f(*i) for i in arr[:,::-1]])

class IndexShow(Tool):
    title = 'Index Show'
    view = [(int, (0,30), 0,  u'width', 'width', 'pix')]
    para = {'width':1}
    
    def __init__(self):
        self.sta = 0
        self.cursor = wx.CURSOR_CROSS
        
    def mouse_down(self, ips, x, y, btn, **key):
        self.sta = 1
        ips.snapshot()
    
    def mouse_up(self, ips, x, y, btn, **key):
        self.sta = 0
        print(x,y)
        gray=ips.img[int(y),int(x)]
        print(gray)
        labels=measure.label(ips.img.copy())
        gray=labels[int(y),int(x)]
        print(gray)        
        props = measure.regionprops(labels)

        l=max(props[gray-1].image.shape)+10
        img1=np.zeros((l,l))
        i=props[gray-1]
        img1[int((l-i.image.shape[0])/2):int((l-i.image.shape[0])/2)+i.image.shape[0],int((l-i.image.shape[1])/2):int((l-i.image.shape[1])/2)+i.image.shape[1]]=i.image
        
        ipsd = ImagePlus([img1], "original")
        ipsd = ImagePlus([ShowPlt(img1).plt_img], "show")
        ipsd.backmode = ipsd.backmode
        IPy.show_ips(ipsd)

class RegionShape(Simple):
    title = 'Cell Morphometric'
    note = ['8-bit', '16-bit']
    
    para = {'con':'8-connect', 'slice':False, 'compactness':True, 'convex_hull_area_ratio':True,'convex_hull_perimeter_ratio':False,
            'elliptic_cpmpactness':True, 'feret_ratio':False,'radial_distance_mean':False,'radial_distance_sd':False, 'radial_distance_area_radtio':False,
            'zero_crossings':True, 'entropy':False}
    
    view = [(list, ['4-connect', '8-connect'], str, 'conection', 'con', 'pix'),
            (bool, 'slice', 'slice'),
            ('lab','=========  indecate  ========='),
            (bool, 'compactness', 'compactness'),
            (bool, 'convex_hull_area_ratio', 'convex_hull_area_ratio'),
            (bool, 'convex_hull_perimeter_ratio', 'convex_hull_perimeter_ratio'),
            (bool, 'elliptic_cpmpactness', 'elliptic_cpmpactness'),
            (bool, 'feret_ratio', 'feret_ratio'),
            (bool, 'radial_distance_mean', 'radial_distance_mean'),
            (bool, 'radial_distance_sd', 'radial_distance_sd'),
            (bool, 'radial_distance_area_radtio', 'radial_distance_area_radtio'),
            (bool, 'zero_crossings', 'zero_crossings'),
            (bool, 'entropy', 'entropy')
            ]
    def run(self, ips, imgs, para = None):
        if not para['slice']: 
            imgs = [ips.img]
        else: imgs = ips.imgs
        buf = imgs[0].astype(np.uint16)
        strc = ndimage.generate_binary_structure(2, 1 if para['con']=='4-connect' else 2)
        idct = ['compactness','convex_hull_area_ratio','convex_hull_perimeter_ratio','elliptic_cpmpactness','feret_ratio','radial_distance_mean',
                'radial_distance_sd','radial_distance_area_radtio','zero_crossings','entropy']
        key = {'compactness':'compactness',
                'convex_hull_area_ratio':'convex_hull_area_ratio',
                'convex_hull_perimeter_ratio':'convex_hull_perimeter_ratio',
                'elliptic_cpmpactness':'elliptic_cpmpactness',
                'feret_ratio':'feret_ratio',
                'radial_distance_mean':'radial_distance_mean',
                'radial_distance_sd':'radial_distance_sd',
                'radial_distance_area_radtio':'radial_distance_area_radtio',
                'zero_crossings':'zero_crossings',
                'entropy':'entropy',
                }
        idct = [i for i in idct if para[key[i]]]
        titles = ['Slice', 'ID'][0 if para['slice'] else 1:] 
        titles.extend(idct)
        k = ips.unit[0]
        data, mark = [], []
        for i in range(len(imgs)):
            n = ndimage.label(imgs[i], strc, output=buf)
            index = range(1, n+1)
            dt = []
            if para['slice']:dt.append([i]*n)
            #print([i for i in range(n)])
            dt.append([i for i in range(n)])
            #print(dt)
            shape=Shapes(imgs[i])
            t=time.time()
            for i in idct:
                if para[i]:dt.append(np.array(shape.para[i]).round(2))
                print(dt)
            #print('#########',time.time()-t)
            centroids = [i.centroid for i in shape.props]
            cvs = [(i.major_axis_length, i.minor_axis_length, i.orientation) for i in shape.props]
            mark.append([(center, cov) for center,cov in zip(centroids, cvs)])
            data.extend(list(zip(*dt)))

        # print(dt)
        IPy.table(ips.title+'-region statistic', data, titles)
        ips.mark = Mark(mark)
        ips.update = True
        ips.tool = IndexShow()

plgs = [RegionShape]

if __name__=='__main__':
    img_gray = imread('test1.png',True)
    img_gray=np.array((img_gray>128)*255)
    shape=Shape(img_gray)
    fig1=plt.figure('图')
    plt.subplot(251)
    plt.title('origina')
    plt.imshow(shape.img,cmap='gray')
    plt.subplot(252)
    plt.title('convex')
    plt.imshow(shape.labels,cmap='gray')
    plt.show()