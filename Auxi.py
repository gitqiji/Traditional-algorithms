import numpy as np
import cv2

class Auxi(object):
    def __init__(self,name='display'):
        self.col = 100
        self.row = 2
        self.name = name
        self.color = {
            -1: (0, 0, 255),
            0: (0, 0, 255),
            1: (0, 255, 0),
            2: (255, 255, 255),
            3: (255,0,0)}
    def savefle(self,x,y):
        with open('data.txt','w') as fdata:
            for i in range(100):
                fdata.writelines(x[:,i])
                fdata.writelines(y[0][i])

    def data0(self):
        x = np.random.randint(10,300,size = (self.row,self.col))
        t = x[0] - x[1]
        y = np.where(t > 0,0,1).reshape(1,self.col)
        line = [[(0,0),(500,500)]]
        return x,y,line

    def data1(self):
        row = 2
        y = np.zeros((1,100))
        y[0,0:50] = 1
        y[0,50:] = 0
        x0 = np.random.randint(10,150,size=(row,50))
        t0 = np.random.randint(150,300,size=(row,1,15))
        t1 = np.random.randint(10,150,size=(row,1,15))
        xtR = np.vstack ((t0[0],t1[1]))#np.append(t0[0],t1[1],axis = 0)
        xtL = np.vstack ((t1[1],t0[0]))#np.append(t0[1],t1[0],axis = 0)
        xt1 = np.hstack((xtR,xtL))#np.append(xtR,xtL,axis = 1)
        xt0 = np.random.randint(150,300,size=(row,20))
        x1 = np.hstack ((xt0,xt1))#np.append (xt0,xt1,axis = 1)
        x = np.append (x0,x1,axis = 1)
        return x,y

    def svmdata(self, tag=None):
        x0 = np.random.randint(10,300,size=(1,100))
        x1 = x0 + 10 + np.random.randint(10,150,size=(1,100)) # x1 - x0 > 10
        x11 = np.random.randint(10,300,size=(1,100)) # x0 - x1 > 10
        x00 = x11 + 10 + np.random.randint(10,150,size=(1,100))
        x_0 = np.hstack((x0,x00))
        x_1 = np.hstack((x1,x11))
        x = np.vstack((x_0,x_1))
        y = np.hstack((0-np.ones_like(x0),np.ones_like(x0)))
        if tag is not None:
            y = np.hstack((1-np.ones_like(x0),np.ones_like(x0)))
        line = [[(0,10),(490,500)],[(10,0),(500,490)]]
        return x,y,line
    def nonline(self,tag=None):
        x0 = np.random.randint(200,300,size=(2,50))
        x11 = np.vstack((np.random.randint(100,400,size=(1,20)), np.random.randint(100,190,size=(1,20))))
        x12 = np.vstack((np.random.randint(100,190,size=(1,20)),np.random.randint(100,400,size=(1,20))))
        x13 = np.vstack((np.random.randint(100,300,size=(1,10)), np.random.randint(310,400,size=(1,10))))
        x1 = np.hstack((x11,x12,x13))
        line = [[(195,195),(400,195)],[(195,195),(195,305)],[(195,305),(300,305)]]
        print np.shape(x0),np.shape(x1)
        x = np.hstack((x1,x0))
        y = np.ones((1,100))
        y[0][0:50] = -1
        if tag is not None:
            y[0][0:50] = 0
        return x,y,line
    def gaussiondata(self,flag=None):
        pos = [160,350]
        sigma = [30,30]
        r = 80;uni_num = 200
        def data1():
            x1 = np.random.randint(200,290,size=(2,uni_num))
            x2 = np.random.randint(310,350,size=(2,uni_num))
            x3 = np.vstack((np.random.randint(310,350,size=(1,uni_num)),np.random.randint(200,290,size=(1,uni_num))))
            x = np.hstack((x1,x2,x3))
            return x
        def data2():
            x1 = np.vstack((np.random.normal(pos[0],sigma[1],size=uni_num),np.random.normal(pos[0],sigma[1],size=uni_num)))
            x2 = np.vstack((np.random.normal(pos[1],sigma[1],size=uni_num),np.random.normal(pos[1],sigma[1],size=uni_num)))
            x3 = np.vstack((np.random.normal(pos[1],sigma[1],size=uni_num),np.random.normal(pos[0],sigma[1],size=uni_num)))
            x = np.hstack((x1,x2,x3))
            #print x
            x = x.astype(int)
            return x
        def data3():
            x1 = np.vstack((np.random.normal(pos[0],sigma[0],size=uni_num),np.random.randint(pos[0]-r,pos[0]+r,size=(1,uni_num))))
            x2 = np.vstack((np.random.normal(pos[1],sigma[0],size=uni_num),np.random.randint(pos[0]-r,pos[0]+r,size=(1,uni_num))))
            x3 = np.vstack((np.random.randint(pos[0]-r,pos[0]+r,size=(1,uni_num)),np.random.normal(pos[1],sigma[0],size=uni_num)))
            x4 = np.vstack((np.random.randint(pos[1]-r,pos[1]+r,size=(1,uni_num)),np.random.normal(pos[1],sigma[0],size=uni_num)))
            x = np.hstack((x1,x2,x3,x4)).astype(int)
            return x
        def data4():
            x1 = np.vstack((np.random.normal(pos[0],sigma[0],size=uni_num),np.linspace(pos[0]-r,pos[0]+r,uni_num)))
            x2 = np.vstack((np.random.normal(pos[1],sigma[0],size=uni_num),np.linspace(pos[0]-r,pos[0]+r,uni_num)))
            x3 = np.vstack((np.linspace(pos[0]-r,pos[0]+r,uni_num),np.random.normal(pos[1],sigma[0],size=uni_num)))
            x4 = np.vstack((np.linspace(pos[1]-r,pos[1]+r,uni_num),np.random.normal(pos[1],sigma[0],size=uni_num)))
            x = np.hstack((x1,x2,x3,x4)).astype(int)
            return x
        data = {1:data1(),2:data2(),3:data3(),4:data4()}
        select = lambda flag: data[flag]
        x = select(flag)
        y1 = np.zeros((1,uni_num))
        y2 = np.hstack((y1,y1+1))
        y3 = np.hstack((y1,y2+1))
        y4 = np.hstack((y1,y3+1))
        y = {1: y3, 2: y3, 3: y4, 4: y4}
        line = [[(0,300),(500,300)],[(300,0),(300,500)]]
        return x,y[flag],line

    def plotinitial(self,x,line):
        m = x.shape[1]
        img = np.zeros((500,500,3),'uint8')
        for pnt in line:
            img = cv2.line(img,pnt[0],pnt[1],self.color[2],1)
        return img,m
    def plotorigin(self,x,line):
        img,m = self.plotinitial(x,line)
        for i in range(m):
            img = cv2.circle(img,(x[0][i],x[1][i]),5,self.color[2],0) # color[y[0][i]]
        self.tagandshow(img)

    def plotline(self,x,line,decline):
        img,m = self.plotinitial(x,line)
        for res in decline:
            img = cv2.line(img,res[0],res[1],self.color[1],1)
        self.tagandshow(img)

    def plotres(self,x,pre,line,decline=None):
        img,m = self.plotinitial(x,line)
        for i in range(m):
            img = cv2.circle(img,(x[0,i],x[1,i]),5,self.color[2],0)
            img = cv2.circle(img,(x[0,i],x[1,i]),4,self.color[pre[i]],-1)
        if decline is not None:
            for res in decline:
                img = cv2.line(img,res[0],res[1],self.color[1],1)
        self.tagandshow(img)

    def tagandshow(self,img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'label: 0', (10, 450), font, 1, self.color[2], 1, cv2.LINE_AA)
        img = cv2.rectangle(img, (150, 430), (160, 450), self.color[0], -1, 8, 0)

        cv2.putText(img, 'label: 1', (300, 30), font, 1, self.color[2], 1, cv2.LINE_AA)
        img = cv2.rectangle(img, (440, 10), (450, 30), self.color[1], -1, 8, 0)

        cv2.imshow(self.name,img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cmd = input()
        if cmd == 1:
            imgname = 'C:\\Users\\qiji\\Documents\\temp\\' + self.name + '.jpg'
            cv2.imwrite(imgname,img)


if __name__ == '__main__':
    auxi = Auxi()
    x,y,line = auxi.gaussiondata(3)
    auxi.plotres(x,y[0],line)
