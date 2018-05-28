from .morphometric import *
from imagepy.core.engine import Tool
from imagepy import IPy
from imagepy.core.engine import Simple, Filter
from imagepy.core.manager import WindowsManager
from imagepy.core.roi.pointroi import PointRoi
# from imagepy import ImagePlus
import time
from skimage.measure import regionprops
from skimage import measure
from imagepy import IPy, wx
from shapely import geometry
from shapely.geometry import Point
from wx.lib.pubsub import pub
import pandas as pd
pub.subscribe(show_plt, 'show_plt')
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
    view = [(int, 'width',(0,30), 0,  u'width',  'pix')]
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
        msk=ips.get_msk()
        contours = measure.find_contours(msk, msk.max()/2)
        for cont in contours:
            polygon = geometry.Polygon(cont)
            if polygon.contains(Point(y,x)):
                # l, a, (cx,cy), (ax1,ax2), m =
                cont=np.array(cont)
                cont[:,0]=cont[:,0].max()-cont[:,0]
                wx.CallAfter(pub.sendMessage,'show_plt',para=feature2d(cont)+(cont,))

class RegionShape(Simple):

    title = 'Cell Morphometric'
    note = ['8-bit', '16-bit']
    
    para = {'con':'8-connect', 'slice':False,'center':True, 'area':True,'perimeter':True, 'compactness':True, 'convex_hull_area_ratio':True,'convex_hull_perimeter_ratio':False,
            'elliptic_cpmpactness':True, 'feret_ratio':False,'radial_distance_mean':False,'radial_distance_sd':False, 'radial_distance_area_radtio':False,
            'zero_crossings':True, 'entropy':False}
    
    view = [(list,  'con',['4-connect', '8-connect'], str, 'conection', 'pix'),
            (bool, 'slice', 'slice'),
            ('lab', None,'=========  base  ========='),
            (bool, 'center', 'center'),
            (bool, 'area', 'area'),
            (bool, 'perimeter', 'perimeter'),

            ('lab', None,'=========  advance  ========='),
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
        print('ok123')
        if not para['slice']: 
            imgs = [ips.img]
        else: imgs = ips.imgs

        buf = imgs[0].astype(np.uint16)
        # strc = ndimage.generate_binary_structure(2, 1 if para['con']=='4-connect' else 2)
        idct = ['compactness','convex_hull_area_ratio','convex_hull_perimeter_ratio','elliptic_cpmpactness','feret_ratio','radial_distance_mean',
                'radial_distance_sd','radial_distance_area_radtio','zero_crossings','entropy']
        mor = {
                'center0':[],
                'center1':[],
                'area':[],
                'perimeter':[],
                'compactness':[],
                'convex_hull_area_ratio':[],
                'convex_hull_perimeter_ratio':[],
                'elliptic_cpmpactness':[],
                'feret_ratio':[],
                'radial_distance_mean':[],
                'radial_distance_sd':[],
                'radial_distance_area_radtio':[],
                'zero_crossings':[],
                'entropy':[],
                'center':[],
                'cov':[],
                }
        titles = ['Slice', 'ID'][0 if para['slice'] else 1:] 
        if para['center']:titles.extend(['Center-X','Center-Y'])
        if para['area']:titles.append('Area')
        if para['perimeter']:titles.append('perimeter')

        titles.extend([i for i in idct if para[i]])
        print('############')

        k = ips.unit[0]
        data, mark = [], []
        for i in range(len(imgs)):
            if ips.get_msk() is None:img=imgs[i]
            else: img=ips.get_msk()
            contours = measure.find_contours(img, img.max()/2)
            # contours = measure.find_contours(msk, msk.max()/2)
            for cont in contours:
                l, a, (cx,cy), (ax1,ax2), m =feature2d(cont)
                hull = ConvexHull(cont)
                hull_cont=np.array([cont[hull.vertices,1], cont[hull.vertices,0]]).T
                h_l, h_a, (h_cx,h_cy), (h_ax1,h_ax2), h_m =feature2d(hull_cont)
                mor['center0'].append(cx)
                mor['center1'].append(cy)
                mor['area'].append(a)
                mor['perimeter'].append(l)
                mor['compactness'].append(compactness(a,l))
                mor['convex_hull_area_ratio'].append(convex_hull_area_ratio(a,h_a))
                mor['convex_hull_perimeter_ratio'].append(convex_hull_perimeter_ratio(l,h_l))
                mor['elliptic_cpmpactness'].append(elliptic_cpmpactness(ax1,ax2,l))
                mor['feret_ratio'].append(feret_ratio(cont))
                mor['radial_distance_mean'].append(radial_distance_mean((cx,cy),cont))
                mor['radial_distance_sd'].append(radial_distance_sd((cx,cy),cont))
                mor['radial_distance_area_radtio'].append(radial_distance_area_radtio((cx,cy),cont))
                mor['zero_crossings'].append(zero_crossings((cx,cy),cont))
                mor['entropy'].append(entropys((cx,cy),cont))
                mor['center'].append((cx,cy))
                mor['cov'].append((ax1,ax2,m))
            dt = []
            n=len(contours)
            print('########',n)
            if para['slice']:dt.append([i]*n)
            dt.append([i for i in range(n)])
            if para['center']:
                dt.append(np.round(mor['center0']*k,2))
                dt.append(np.round(mor['center1']*k,2))
            if para['area']:dt.append(np.round(mor['area'],2)*k**2)
            if para['perimeter']:dt.append(np.round(mor['perimeter'],2)*k)
            if para['compactness']:dt.append(np.round(mor['compactness'],2))
            if para['convex_hull_area_ratio']:dt.append(np.round(mor['convex_hull_area_ratio'],2))
            if para['convex_hull_perimeter_ratio']:dt.append(np.round(mor['convex_hull_perimeter_ratio'],2))
            if para['elliptic_cpmpactness']:dt.append(np.round(mor['elliptic_cpmpactness'],2))
            if para['feret_ratio']:dt.append(np.round(mor['feret_ratio'],2))
            if para['radial_distance_mean']:dt.append(np.round(mor['radial_distance_mean'],2))
            if para['radial_distance_sd']:dt.append(np.round(mor['radial_distance_sd'],2))
            if para['radial_distance_area_radtio']:dt.append(np.round(mor['radial_distance_area_radtio'],2))
            if para['zero_crossings']:dt.append(np.round(mor['zero_crossings'],2))
            if para['entropy']:dt.append(np.round(mor['entropy'],2))

            centroids = mor['center']
            cvs =  mor['cov']
            mark.append([(center, cov) for center,cov in zip(centroids, cvs)])
            data.extend(list(zip(*dt)))
            print(titles)
            print(data)
        # IPy.table(ips.title+'-region statistic', data, titles)
        IPy.show_table(pd.DataFrame(data, columns=titles), ips.title+'-region')
        print('123')
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