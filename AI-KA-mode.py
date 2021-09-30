from PIL import Image
from pytesseract import image_to_string
from pytesseract import image_to_data, Output
import time,os 
#import cv2
from scipy import stats
import math
import numpy as np
import cv2 as cv


def get_numbers(fname):
    #img = cv.imread(fname,0)
    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #cl1 = clahe.apply(img)
    #print(cl1)
    #try:
    #cv.imwrite('Clahe-'+fname,cl1)
    #img = Image.fromarray(cl1)
    a = image_to_data(Image.open(fname),output_type=Output.DICT,lang='fin+swe', config='--psm 3' )
    #a = image_to_data(cl1,output_type=Output.DICT,lang='fin+swe', config='--psm 12' )
    #a = image_to_data(fname,output_type=Output.DICT)
    b = a['conf']
    #print(' Zeros ',b.count(0))
    #print('Boxes ', b.count('-1'))
    count = 0 
    count_low = 0 
    count_high = 0
    sum  = 0 
    bb = []
    for i in range(0,len(b)):
        if b[i] != '-1':
            bb.append(b[i])
            count = count + 1
            sum = sum + b[i]
            if b[i] < 50: 
                count_low = count_low + 1 
            elif b[i] > 49 :
                count_high = count_high +1
        #print(b[i])
#print(a) 
    smode = 0 
    mini = 0 
    maxi = 0 
    var = 0 
    if len(bb)>0:
        smode = stats.mode(bb)[0][0]
        minmax = stats.describe(bb)[1]
        mini = minmax[0]
        maxi = minmax[1]
        var = 0
        print(smode)
        print(mini,maxi)
        if math.isnan(stats.describe(bb)[3]) == False:
            var =  stats.describe(bb)[3]
            print(var)
        
    if count > 0 : 
        ave = sum/count      
        #print( ' Average ', ave)
        low = 100* count_low/count
        high = 100*count_high/count
    else:
        ave = 0 
        low = 0 
        high = 0 
        #print('Average ', ave) 
    return   ave, b.count(0),b.count('-1'),low,count_low,high,count_high,smode,mini,maxi,var

fname = '0027.jpg'
dirpath = os.getcwd()
#dirpath = dirpath + '/lajitellut'
start = time.time()
w_file = open('model-psm-3-original-data.csv','w')
#w_file.write('Conf_ave'+ ','+ 'Zeros ,' +'Boxes ,' + 'Low ,' + 'High ,'   'dirname , '+ 'filename' + '\n')
content = ''
for root,dirs,files in os.walk(dirpath):
    ##w_file.write(fname +','+foldername+ '\n')
    #print(dirs)
    #print(len(dirs), len(files))
    for onedir in dirs:
        oneFullDir = os.path.join(root,onedir)
        print(oneFullDir)
        FilesInDir = os.listdir(oneFullDir)
        fullDirContentCount = len(FilesInDir)
        start = time.time()
        for i in range(0,fullDirContentCount):
            start = time.time()
            FileToRead = oneFullDir + '/' + FilesInDir[i]
            #image = cv2.imread(FileToRead)
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# check to see if we should apply thresholding to preprocess the
# image
#if args["preprocess"] == "thresh":
# 
            #gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            #print(FilesInDir[i], onedir)
            if os.path.isfile(FileToRead)==True:
                aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk = get_numbers(FileToRead)
                a = str(aa)
                b = str(bb)
                c = str(cc)
                d = str(dd)
                e = str(ee)
                f = str(ff)
                g = str(gg)
                h = str(hh)
                i = str(ii)
                j = str(jj)
                k = str(kk)
                w_file.write(a + ',' + b + ',' + c  + ',' + d + ',' + e + ','+ f + ','+ g + ','+ h + ','+ i + ','+ k + ','+ j + ','  + onedir + ',' + FileToRead + '\n')
            end = time.time()
            print( 'Wall clock time ', end- start)
        #content+="{}, {}\n".format(oneFullDir.lstrip(dirpath), fullDirContentCount)
#aa,bb,cc = get_numbers(fname)
end = time.time()
print( 'Wall clock time ', end- start)
#print('Ave ', aa,' Zeros', bb,'Boxes ', cc)


