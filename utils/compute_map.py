import numpy as np
import sys
import os
import cv2
import shutil
saveTop9 = 'TOP9-Retrival-Results'#保存检索结果的地址
imgDatapath = 'source_edge'#存放待检索图像的地址
queryImgDataPath = '330sketches'#存放检索图像的地址

queryImg = '330sketches.txt'#所有检索的图像名称以及对应的类别以及相关联图像的数目
allImg = 'train.txt'#存放待检索图像的名称以及类别

queryImgFeature = 'sketch-crow-features'#存放检索图像的特征向量
allImgFeature = 'source-crow-features'#存放待检索图像的特征向量

N = 14660#待检索图像的数目
#按行提取txt文档中的信息并保存到两个list中
fid = open(queryImg)
queryImgs = []
for line in fid.readlines():
    queryImgs.append(line.split(" "))
#print(len(queryImgs),type(queryImgs[1][1]))
fid.close()
fid= open(allImg)
allImgs=[]
for line in fid.readlines():
    allImgs.append(line.split(" "))
fid.close()
#待检索图像以及检索图像的数目
queryImgNum = len(queryImgs)
allImgNum = len(allImgs)
#用来存放每一个检索图像的AP
ap = np.zeros((queryImgNum))

#对每一个检索图像求AP
for i in range(queryImgNum):
    #为检索结果创建对应的文件夹
    queryPath = queryImgs[i][0]#10/9.jpg
    imgPath = os.path.join(queryImgDataPath, queryPath)
    img = cv2.imread(imgPath)
    queryPathParts = queryPath.split("/")
    queryFile = queryPathParts[0]#10
    queryNamePath = queryPathParts[1]#9.jpg
    loc = queryNamePath.split(".")[0]#9
    queryName = loc

    saveTop9Path = os.path.join(saveTop9+queryFile, queryName)
    if os.path.exists(saveTop9Path):
        shutil.rmtree(saveTop9Path)#如果存在则删除已有的检索结果
        os.makedirs(saveTop9Path)
    else:
        os.makedirs(saveTop9Path)

    queryClass = queryImgs[i][1]
    queryCorrNum = int(queryImgs[i][2])
    queryImgFeaturePath = os.path.join(queryImgFeature,queryFile,queryName+'.npy')
    queryFeature = np.load(queryImgFeaturePath)
    
    dist = np.zeros((allImgNum))#保存检索图像到每一副检索图像的距离
    #二重循环，求出检索图像到每一幅待检索图像的距离
    for j in range(allImgNum):
        allPath = allImgs[j][0]
        allPathParts = allPath.split('/')
        allFile = allPathParts[0]
        allNamePath = allPathParts[1]
        allName = allNamePath.split('.')[0]
        allClass = allImgs[i][1]
        allImgFeaturePath = os.path.join(allImgFeature, allFile,allName+'.npy')
        allFeature = np.load(allImgFeaturePath)
        dist[j]=((allFeature-queryFeature)**2).sum()
    #对距离进行排序，从小到大
    rank = np.argsort(dist)
    similarTerm = 0
    precision = np.zeros((N))
    #print(dist[rank[0:9]])
    #下面的循环用来求每一个召回率下对应的准确率
    for k in range(N):
        topkClass = allImgs[rank[k]][1]
        if queryClass == topkClass:
            similarTerm = similarTerm + 1.
            precision[k]= similarTerm/(k+1)
    #保存检索结果的前9个
    for k in range(9):
        topkClass = allImgs[rank[k]][1]
        imPath = os.path.join(imgDatapath, allImgs[rank[k]][0])
       # print(imPath)
        im = cv2.imread(imPath)
        filePath = allImgs[rank[k]][0].split("/")
        saveTop9ImgPath = os.path.join(saveTop9Path,str(k)+'_'+filePath[0]+'_'+filePath[1])
        print(saveTop9ImgPath)
        cv2.imwrite(saveTop9ImgPath, im)
        saveQueryImgPath = os.path.join(saveTop9Path, 'query'+'.jpg')
        cv2.imwrite(saveQueryImgPath, img)
    #计算AP
    sumPre = precision.sum()
    ap[i] = (sumPre/queryCorrNum) * 100

    print(queryName, ap[i])
#计算maP
maP = ap.sum() / queryImgNum

print("map is %f" % maP)


















