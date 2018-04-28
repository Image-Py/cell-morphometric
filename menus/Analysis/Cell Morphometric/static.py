
from skimage import data,color,morphology,measure,draw
import numpy as np
from scipy.misc import imread,imsave
from scipy.ndimage import binary_fill_holes
from scipy.stats import entropy
import math
from scipy import ndimage
import wx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os,time

class ShowPlt:
    def __init__(self, img):
        l=int(max(img.shape)*1.5)
        img1=np.zeros((l,l))
        img1[int((l-img.shape[0])/2):int((l-img.shape[0])/2)+img.shape[0],int((l-img.shape[1])/2):int((l-img.shape[1])/2)+img.shape[1]]=img
        self.img = img1
        self.plt_img=self.show_plt()
    def show_plt(self):
        img_gray=self.img
        shape=Shape(img_gray,1)
        self.contours=shape.contours
        fig1=plt.figure('图')
        ax =plt.subplot(241)
        plt.title('origina')
        plt.imshow(shape.img,cmap='gray')
        plt.subplot(242)
        plt.title('convex')
        plt.imshow(shape.get_chull_img(),cmap='gray')

        ax=plt.subplot(243)
        plt.title('cricle')    
        center=shape.props.centroid
        circle1 = plt.Circle((center[1], center[0]), shape.radial_distance_mean(),fill=False, color='r')
        ax.add_artist(circle1)
        plt.imshow(shape.img,cmap='gray')

        ax=plt.subplot(244)
        plt.title('distance mean cricle')
        dist=np.array([np.linalg.norm(i) for i in (shape.contours-shape.props.centroid)])
        hist1=[i for i in range(len(dist))]
        plt.plot(hist1,dist,'r')
        y=[shape.radial_distance_mean()]*len(dist)
        hist1=[i for i in range(len(dist))]
        plt.plot(hist1,y,'b')

        ax=plt.subplot(245)
        plt.title('without circle')
        img1=img_gray.copy()
        rr, cc = draw.circle(center[0], center[1], shape.radial_distance_mean())
        img1[rr, cc] = 0

        plt.imshow(img1,cmap='gray')
        ax=plt.subplot(246)
        plt.title('ellipse')
        elp_o=shape.props.orientation*180/np.pi
        ellipse = mpatches.Ellipse((center[1],center[0]), shape.props.major_axis_length, shape.props.minor_axis_length,-elp_o, fill=False,edgecolor='red',linewidth=1)  
        ax.add_patch(ellipse)
        plt.imshow(shape.img,cmap='gray')
        axes=plt.subplot(247)
        plt.title('feret_ratio')
        self.feret_ratio(axes)
        plt.imshow(shape.img,cmap='gray')
        plt.savefig ( "./my_img.png" )
        plt.clf()
        img = imread('my_img.png',False)
        return img
    def feret_ratio(self,ax):
        tri = np.array(self.contours)
        r = np.linspace(0, np.pi, 181)
        v = np.array([np.cos(r), np.sin(r)])
        rst = np.dot(tri, v).ptp(axis=0)
        ma = np.dot(tri, np.array([np.cos(r[np.argmax(rst)]), np.sin(r[np.argmax(rst)])]))
        mi = np.dot(tri, np.array([np.cos(r[np.argmin(rst)]), np.sin(r[np.argmin(rst)])]))
        #长线
        self.drawline(self.irotate(np.array([500,min(ma)]),r[np.argmax(rst)]),self.irotate(np.array([-500,min(ma)]),r[np.argmax(rst)]),clr='b',ax1=ax)
        self.drawline(self.irotate(np.array([500,max(ma)]),r[np.argmax(rst)]),self.irotate(np.array([-500,max(ma)]),r[np.argmax(rst)]),clr='b',ax1=ax)
        # #短线
        self.drawline(self.irotate(np.array([500,min(mi)]),r[np.argmin(rst)]),self.irotate(np.array([-500,min(mi)]),r[np.argmin(rst)]),clr='b',ax1=ax)
        self.drawline(self.irotate(np.array([500,max(mi)]),r[np.argmin(rst)]),self.irotate(np.array([-500,max(mi)]),r[np.argmin(rst)]),clr='b',ax1=ax)
    def irotate(self,ary,thi):
        return np.dot(ary,np.array([[np.cos(thi), -np.sin(thi)],
                                   [np.sin(thi), np.cos(thi)]]))
    def drawline(self,x1,x2,clr,ax1):
        line = [tuple(x1),tuple(x2)]
        (line_x, line_y) = zip(*line)
        ax1.add_line(Line2D(line_x,line_y, linewidth=1, color=clr))


def compactness(props):return (props.area*4*np.pi)/(props.perimeter**2)

def convex_hull_area_ratio(props): return props.solidity

def convex_hull_perimeter_ratio(props,chull_img):
    labels_cvx = measure.label(chull_img,connectivity=2)
    self.props_cvx = measure.regionprops(labels_cvx)[0]
    return self.props_cvx.perimeter/self.props.perimeter

def elliptic_cpmpactness(props):

    elp_ma = props.major_axis_length
    elp_mi = props.minor_axis_length
    elp_o = props.orientation*180/np.pi
    elp_p =math.pi * ( 3*(elp_ma/2+elp_mi/2) - math.sqrt( (3*elp_ma/2 + elp_mi/2) * (elp_ma/2 + 3*elp_mi/2)))
    return elp_p/props.perimeter


def feret_ratio(contours):
    tri = np.array(contours)
    r = np.linspace(0, np.pi*2, 361)
    v = np.array([np.cos(r), np.sin(r)])
    rst = np.dot(tri, v).ptp(axis=0)
    return min(rst)/max(rst)

def radial_distance_mean(props,contours,k):
    centr = np.array(list(props.centroid))
    dist=(contours-centr)*k
    return np.sum(np.array([np.linalg.norm(dist,axis=1)]))/len(dist)


#can use np.std?
def radial_distance_sd(props,contours,k):
    u=radial_distance_mean(props,contours,k)
    centr = np.array(list(props.centroid))
    dist=np.array([np.linalg.norm(self.contours-centr,axis=1)])*k
    return np.sqrt(np.sum(np.square(dist-u))/len(dist))

class Shape:
    def __init__(self, img,k):
        img=np.array(img)
        self.k=k
        self.img=img
        self.contours = measure.find_contours(self.img, 0)[0]
        self.labels=measure.label(self.img)
        self.props = measure.regionprops(self.labels)[0]
    def compactness(self):return (self.props.area*4*np.pi)/(self.props.perimeter**2)
    def convex_hull_area_ratio(self): return self.props.solidity
    def convex_hull_perimeter_ratio(self):
        labels_cvx = measure.label(self.get_chull_img(),connectivity=2)
        self.props_cvx = measure.regionprops(labels_cvx)[0]
        return self.props_cvx.perimeter/self.props.perimeter
    def elliptic_cpmpactness(self):
        elp_ma = self.props.major_axis_length
        elp_mi = self.props.minor_axis_length
        elp_o = self.props.orientation*180/np.pi
        elp_p =math.pi * ( 3*(elp_ma/2+elp_mi/2) - math.sqrt( (3*elp_ma/2 + elp_mi/2) * (elp_ma/2 + 3*elp_mi/2)))
        return elp_p/self.props.perimeter
    def get_chull_img(self):return  morphology.convex_hull_image(self.img) 
    def feret_ratio(self):
        tri = np.array(self.contours)
        r = np.linspace(0, np.pi*2, 361)
        v = np.array([np.cos(r), np.sin(r)])
        rst = np.dot(tri, v).ptp(axis=0)
        return min(rst)/max(rst)
    def radial_distance_mean(self):
        centr = np.array(list(self.props.centroid))
        dist=(self.contours-centr)*self.k
        return np.sum(np.array([np.linalg.norm(dist,axis=1)]))/len(dist)

    def radial_distance_sd(self):
        u=self.radial_distance_mean()
        centr = np.array(list(self.props.centroid))
        dist=np.array([np.linalg.norm(i) for i in (self.contours-centr)])*self.k
        return np.sqrt(np.sum(np.square(dist-u))/len(dist))
    def radial_distance_area_radtio(self):
        img_temp=np.copy(self.img)
        points=np.array(np.where(img_temp>0)).T
        dist=self.radial_distance_mean()
        points=points-self.props.centroid
        return np.sum([np.linalg.norm(i) for i in points]>dist)
    def zero_crossings(self):
        dist=np.array([np.linalg.norm((self.contours-self.props.centroid),axis=1)])
        a= dist-self.radial_distance_mean()
        return len(np.where(np.diff(np.sign(a)))[0])
    def entropy(self):
        centr = np.array(list(self.props.centroid))
        dist=np.array([np.linalg.norm(i) for i in (self.contours-centr)])*self.k
        max1=np.max(dist)
        dist=dist*100//(int(max1))
        hist=list(np.histogram(dist,bins=100,range=(0,100)))
        return entropy(hist[0])

class Shapes(object):
    """docstring for Shapes"""
    def __init__(self, img,k):
        self.img=img
        self.k=k
        labels=measure.label(self.img)
        self.props = measure.regionprops(labels)
        idct = ['compactness','convex_hull_area_ratio','convex_hull_perimeter_ratio','elliptic_cpmpactness','feret_ratio','radial_distance_mean',
                'radial_distance_sd','radial_distance_area_radtio','zero_crossings','entropy']
        self.para = {
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
        for i in self.props:
            l=max(i.image.shape)*2
            img1=np.zeros((l,l))
            img1[int((l-i.image.shape[0])/2):int((l-i.image.shape[0])/2)+i.image.shape[0],int((l-i.image.shape[1])/2):int((l-i.image.shape[1])/2)+i.image.shape[1]]=i.image
            temp=Shape(img1,self.k)

            self.para['center0'].append(round(temp.props.centroid[1]*self.k,1))
            self.para['center1'].append(round(temp.props.centroid[0]*self.k,1))
            self.para['area'].append(temp.props.area*self.k**2)
            self.para['perimeter'].append(round(temp.props.perimeter*k,1))
            self.para['compactness'].append(temp.compactness())
            self.para['convex_hull_area_ratio'].append(temp.convex_hull_area_ratio())
            self.para['convex_hull_perimeter_ratio'].append(temp.convex_hull_perimeter_ratio())
            self.para['elliptic_cpmpactness'].append(temp.elliptic_cpmpactness())
            self.para['feret_ratio'].append(temp.feret_ratio())
            self.para['radial_distance_mean'].append(temp.radial_distance_mean())
            self.para['radial_distance_sd'].append(temp.radial_distance_sd())
            self.para['radial_distance_area_radtio'].append(temp.radial_distance_area_radtio())
            self.para['zero_crossings'].append(temp.zero_crossings())
            self.para['entropy'].append(temp.entropy())