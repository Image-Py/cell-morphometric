from scipy.misc import imread
import matplotlib.pyplot as plt
from numpy.linalg import norm, eig
import numpy as np
from scipy.stats import entropy
from scipy.spatial import ConvexHull
from shapely import geometry
from skimage import measure
from matplotlib.lines import Line2D
def area(cont):
    return np.cross(cont[1:], cont[:-1]).sum()/2

def length(cont):
    return norm(cont[1:]-cont[:-1], axis=1).sum()

def centroid(cont):
    ws = np.cross(cont[1:], cont[:-1])
    cs = (cont[1:] + cont[:-1])/3
    return (cs.T*ws).sum(axis=1)/ws.sum()

def feature2d(cont):
    p1, p2 = cont[1:], cont[:-1]
    l = norm(p1-p2, axis=1).sum()
    dxy, p12 = np.cross(p1, p2), p1 + p2

    a00 = dxy.sum()/2
    a10, a01 = (dxy*p12.T).sum(axis=1)/a00/6
    a20, a02 = (dxy * (p1 * p12 + p2**2).T).sum(axis=1)/12
    a11 = (dxy * (p1[:,0] * (p12[:,1] + p1[:,1])
        + p2[:,0] * (p12[:,1] + p2[:,1]))).sum()/24
    cov = np.array([[a20, a11],[a11, a02]])/a00
    dcov = np.dot([[a10],[a01]],[[a10,a01]])
    eigvalue, eigvector = eig((cov-dcov)*16)
    return l, a00, (a10, a01), np.sqrt(eigvalue), eigvector

def compactness(area,perimeter):return (area*4*np.pi)/(perimeter**2)

def convex_hull_area_ratio(area,h_area): return area/h_area

def convex_hull_perimeter_ratio(perimeter,h_perimeter):return h_perimeter/perimeter


def elliptic_cpmpactness(major,minor,perimeter):
    '''
    elliptic_cpmpactness=fitting ellipse perime/tumor perime

    np.pi*b+2*(a-b) is the formula ellipse perime

    '''

    elp_p =np.pi*minor+2*(major-minor)
    return elp_p/perimeter

def feret_ratio(contours):
    '''
    calculate the feret ratio
    feret ratio=minimum feret ratio/maxmum feret ratio

    '''
    tri = np.array(contours)
    r = np.linspace(0, np.pi*2, 361)
    v = np.array([np.cos(r), np.sin(r)])
    rst = np.dot(tri, v).ptp(axis=0)
    return min(rst)/max(rst)

def radial_distance_mean(centr,contours):
    dist=(contours-np.array(centr))
    return np.sum(np.array([np.linalg.norm(dist,axis=1)]))/len(dist)/np.max(dist)

def radial_distance_mean1(centr,contours):
    dist=(contours-np.array(centr))
    return np.sum(np.array([np.linalg.norm(dist,axis=1)]))/len(dist)
def radial_distance_sd(centr,contours):
    dist=np.array([np.linalg.norm((contours-centr),axis=1)])
    dist/=np.max(dist)
    return np.std(dist)

def radial_distance_area_radtio(centr,contours):

        # polygon = geometry.Polygon(cont)
        # cir=geometry.Point((cx,cy)).buffer(radial_distance_mean1((cx,cy),cont))
        # boundary=polygon.difference(cir).boundary

    polygon = geometry.Polygon(contours)
    cir=geometry.Point(centr).buffer(radial_distance_mean1(centr,contours))
    return polygon.difference(cir).area

def zero_crossings(centr,contours):
    dist=np.array([np.linalg.norm((contours-centr),axis=1)])
    a= dist-radial_distance_mean1(centr,contours)
    return len(np.where(np.diff(np.sign(a)))[0])

def entropys(centr,contours):
    dist=np.array([np.linalg.norm(i) for i in (contours-centr)])
    dist=dist*100//(int(np.max(dist)))
    hist=list(np.histogram(dist,bins=100,range=(0,100)))
    return entropy(hist[0])

def draw_feret_ratio(ax,contours):
    tri = np.array(contours)
    line_long=(np.max(tri)-np.min(tri))*1.8
    r = np.linspace(0, np.pi, 181)
    v = np.array([np.cos(r), np.sin(r)])
    rst = np.dot(tri, v).ptp(axis=0)
    ma = np.dot(tri, np.array([np.cos(r[np.argmax(rst)]), np.sin(r[np.argmax(rst)])]))
    mi = np.dot(tri, np.array([np.cos(r[np.argmin(rst)]), np.sin(r[np.argmin(rst)])]))
    #长线
    drawline(irotate(np.array([line_long,min(ma)]),r[np.argmax(rst)]),irotate(np.array([-line_long,min(ma)]),r[np.argmax(rst)]),clr='red',ax1=ax)
    drawline(irotate(np.array([line_long,max(ma)]),r[np.argmax(rst)]),irotate(np.array([-line_long,max(ma)]),r[np.argmax(rst)]),clr='red',ax1=ax)
    # #短线
    drawline(irotate(np.array([line_long,min(mi)]),r[np.argmin(rst)]),irotate(np.array([-line_long,min(mi)]),r[np.argmin(rst)]),clr='red',ax1=ax)
    drawline(irotate(np.array([line_long,max(mi)]),r[np.argmin(rst)]),irotate(np.array([-line_long,max(mi)]),r[np.argmin(rst)]),clr='red',ax1=ax)
def irotate(ary,thi):
    return np.dot(ary,np.array([[np.cos(thi), -np.sin(thi)],
                               [np.sin(thi), np.cos(thi)]]))
def drawline(x1,x2,clr,ax1):
    line = [tuple(x1),tuple(x2)]
    (line_x, line_y) = zip(*line)
    ax1.add_line(Line2D(line_x,line_y, linewidth=1, color=clr))

def show_plt(para):

        l, a, (cx,cy),(ax1,ax2), m,cont=para
        # cont[:]=cont[::-1,:]
        #画边界
        print(cont.shape)
        # cont[:]=np.array([cont[:,0]-np.min(cont[:,0]),cont[:,1]-np.min(cont[:,1])]).T
        a = np.linspace(0, np.pi*2,100)
        xys = np.array([np.cos(a), np.sin(a)])
        M = m * (ax1, ax2)/2
        xs, ys =np.dot(M, xys)

        plt.figure('图')        
        plt.subplot(231)
        plt.title('origina')
        plt.plot(cont[:,1], cont[:,0], 'blue')

        hull = ConvexHull(cont)

        plt.subplot(232)
        plt.title('hull image')
        plt.plot(cont[:,1], cont[:,0], 'blue')
        plt.plot(cont[hull.vertices[0],1], cont[hull.vertices[0],0], 'ro')
        plt.plot(cont[hull.vertices,1], cont[hull.vertices,0], 'r--', lw=2)
                
        plt.subplot(233)
        plt.title('radial area')
        plt.plot(cont[:,1], cont[:,0], 'blue')
        #画radial_distance_area_radtio
        polygon = geometry.Polygon(cont)
        cir=geometry.Point((cx,cy)).buffer(radial_distance_mean1((cx,cy),cont))
        boundary=polygon.difference(cir).boundary
        for i in boundary:
            plt.plot(np.array(i)[:,1], np.array(i)[:,0], 'red')

        #画等效椭圆
        plt.subplot(234)
        plt.title('elliptic')
        plt.plot(cont[:,1], cont[:,0], 'blue')
        plt.plot(ys+cy, xs+cx, 'red')

        #画feret
        ax=plt.subplot(235)
        plt.title('feret')
        draw_feret_ratio(ax,cont)
        plt.plot(cont[:,1], cont[:,0], 'blue')
        # plt.imshow(img)

        
        plt.subplot(236)
        plt.title('distant_hist')
        dist=np.array([np.linalg.norm((cont-np.array((cx,cy))),axis=1)])[0]
        hist1=[i for i in range(len(dist))]
        plt.plot(hist1,dist,'r')
        y=[radial_distance_mean1((cx,cy),cont)]*len(dist)
        hist1=[i for i in range(len(dist))]
        plt.plot(hist1,y,'b')
        plt.show()
if __name__ == '__main__':
    img = imread('test.png')
    contours = measure.find_contours(img, img.max()/2)
    index=0
    for cont in contours:
        # print(cont[:,0])
        cont=np.array([cont[:,0]-cont[:,0].min(),cont[:,1]-cont[:,1].min()]).T
        index+=1

        l, a, (cx,cy), (ax1,ax2), m =feature2d(cont)
        print('area', a)
        print('perimeter:', l)
        print('centroid ', cx, cy)
        print('major:%.4f  minor:%.4f'%(ax1, ax2))
        print('compactness:',compactness(a,l))
        print('feret_ratio:',feret_ratio(cont))
        print('radial_distance_mean:',radial_distance_mean((cx,cy),cont))
        print('radial_distance_sd:',radial_distance_sd((cx,cy),cont))
        print('zero_crossings:',zero_crossings((cx,cy),cont))
        print('entropys:',entropys((cx,cy),cont))

        print('#########')
        hull = ConvexHull(cont)
        hull_cont=np.array([cont[hull.vertices,1], cont[hull.vertices,0]]).T
        # print(hull_cont)
        h_l, h_a, (h_cx,h_cy), (h_ax1,h_ax2), h_m =feature2d(hull_cont)
        print('h_area', h_a)
        print('h_perimeter:', h_l)
        print('h_centroid ', h_cx, h_cy)
        print('h_major:%.4f  h_minor:%.4f'%(h_ax1, h_ax2))

        print('convex_hull_area_ratio',convex_hull_area_ratio(a,h_a))
        print('convex_hull_perimeter_ratio',convex_hull_perimeter_ratio(l,h_l))
        print('elliptic_cpmpactness',elliptic_cpmpactness(ax1,ax2,l))
        print('radial_distance_area_radtio',radial_distance_area_radtio((cx,cy),cont))
        print('#########')
        show_plt(l, a, cx,cy, ax1,ax2, m,cont)


    plt.show()
