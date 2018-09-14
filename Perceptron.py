from auxi import *

class Perceptron(object):
    def __init__(self,sample,label):
        print 'This is preceptron'
        self.x = sample
        self.y = label
        self.alpha = 0.01
        self.m,self.n = np.shape(sample)
        self.w = 0
        self.b = 0
        self.decline = []
        self.res = []


    def randomgrad(self):
        w = np.random.random((1,self.m))
        b = np.random.random((1,1))
        x = self.x
        item = 100
        for j in range(item):
            err = 0;res = np.zeros_like(self.y)
            for i in range(self.n):
                a = np.dot(w,x[:,i]) + b
                res[0][i] = np.sign(a)
                if y[0][i] == np.sign(a):
                    #continue
                    pass
                else:
                    w = w + self.alpha * y[0][i] * x[:,i].T
                    b = b + self.alpha * y[0][i]
                    err += 1
            self.decline.append([(0, int(-b[0][0]/w[0][1])), (400, int(-(400*w[0][0]+b[0][0])/w[0][1]))])
            self.res.append(res)
            if err == 0:
                print res
                self.w = w
                self.b = b
                break
            else:
                print 'item: ',j,'err: ',err


'''
loss function:
functional margin
sum(-yi(w*xi + b)) s.t.sign(w*xi+b) != yi
dw = sum(yi*xi)
db = sum(yi)
line:
w1x1 + w2x2 + b = 0
[0,-b/w2],[500,-(500w1+b)/w2]
'''

if __name__ == '__main__':

    auxi = Auxi()
    x,y,line = auxi.svmdata()
    pcpt = Perceptron(x,y)
    pcpt.randomgrad()
    dline = pcpt.decline
    print dline,type(dline)
    for l in dline:
        print l[0],l[1]
    print type(line)
    for l in line:
        print l[0],l[1]
    auxi.plotorigin(x,line)
    for i in range(len(dline)):
        print pcpt.res[i]
        auxi.plotres(x,pcpt.res[i][0],line,[dline[i]])
