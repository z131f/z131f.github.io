---
title: python实战_用镜子实现触屏功能
date: '2024-08-13 00:41:01'
updated: '2024-08-13 00:45:53'
tags:
  - python
  - 项目
permalink: /post/python-actual-combat-fmla5.html
comments: true
toc: true
---

# python实战_用镜子实现触屏功能

github上的[sistine](https://github.com/bijection/sistine?tab=readme-ov-file)项目用一面小镜子和一美元硬币实现了触屏功能。本文尝试用一些比较常见的方法达到类似效果。

**本文代码会逐行解析**

**面向读者**：刚掌握python基本语法，想用所学知识做一些有意思的项目

# 原理分析

如下图所示，在摄像头上方放置一面镜子

​![Opencv实战：用一面镜子实现触屏功能（1）项目原理分析_image_1](assets/Opencv实战：用一面镜子实现触屏功能（1）项目原理分析_image_1-20240422182919-u9jpnud.png)​

​![Opencv实战：用一面镜子实现触屏功能（1）项目原理分析_image_2](assets/Opencv实战：用一面镜子实现触屏功能（1）项目原理分析_image_2-20240422182928-pfhf8t7.jpg)​

用手指触碰屏幕

​![Opencv实战：用一面镜子实现触屏功能（1）项目原理分析_image_3](assets/Opencv实战：用一面镜子实现触屏功能（1）项目原理分析_image_3-20240422182932-oi4n3r7.jpg)​

可以发现，手指会在屏幕上产生倒影，手指没有接触屏幕时，真实的手指和倒影之间会有一段距离；手指接触到屏幕时，倒影也会和手指碰到一起。这样，我们就可以通过检测手指是否和倒影接触来得到我们需要的信息。

程序拆解如下：

1. 获取当前摄像头中的图像。
2. 图像处理。
3. 查找两个手指轮廓的位置，计算两个极点之间的距离，利用两个点之间的距离得到想要的位置坐标（摄像头拍到的画面中的坐标信息）。
4. 建立起一种摄像头画面中的点到实际屏幕上的点的对应关系，利用这种关系得到要输出的结果坐标 （实际屏幕上的坐标）。
5. 利用这个坐标去做我们想要实现的操作。

# 代码实现

## 1.从摄像头获取图像

```python
import cv2
import numpy as np

camera = cv2.VideoCapture(0)
camera.set(3, 640)  
camera.set(4, 480)

while True:
	ret, frame = camera.read()
	frame_flip = cv2.flip(frame, 1)
	cv2.imshow('frame_flip', frame_flip)
	cv2.waitKey(1)

```

运行代码，可以看到成功获取到了摄像头图像。

### 代码分析

下面我们逐句读这段代码。

```python
import cv2
import numpy as np
```

引入所需包

```python
camera = cv2.VideoCapture(0)
camera.set(3, 640)  
camera.set(4, 480)
```

第一句的意思是调用opencv库中的VideoCapture函数来得到摄像头对象，传入的参数0表示摄像头的编号。我们直接传入0，获取电脑自带的摄像机就好。

之后两行表示设置相机的参数，这里分别设置了相机的宽度和高度。

循环体里的内容：

```python
ret, frame = camera.read()
```

这个函数可以从摄像头里读一次数据，返回两个参数。

* ret：是否成功获取到数据。
* frame：获取到的图像对象，这是一个numpy数组。

再下一行

```python
frame_flip = cv2.flip(frame, 1)
```

这句表示把图像翻转，如何不加这一句，获取的图像和我们实际的动作是左右对称的。  
参数：

* 1：水平翻转
* 0：垂直翻转
* -1：水平垂直翻转  
  如图：  
  ​![Opencv实战：用一面镜子实现触屏功能（2）从摄像头获取图像_image_1](assets/Opencv实战：用一面镜子实现触屏功能（2）从摄像头获取图像_image_1-20240422183046-kug0kvk.png)​

最后两行

```python
cv2.imshow('frame_flip', frame_flip)
cv2.waitKey(1)
```

第一句表示把要显示的图片加载到窗口，第一个参数是窗口的名字。  
第二句的意思是获取按下的按键，它执行了两个操作：

1. 告诉opencv显示图像
2. 监听按下的按键，在用户按下按键或1ms时限到了时返回

每次想要显示图像时，都要调用这个函数。

```python
cv2.waitKey(0)
```

你也可以把参数改为0，表示无限的等待时间。

```python
s = cv2.waitKey(0)
if s == ord('k):
	print('k')
```

这个函数会返回按下的按键，可以用这种方法来检测按下了哪个键。

## 2.图像处理

图像预处理有转换颜色空间、高斯滤波、二值化和腐蚀膨胀四步。

### 转换颜色空间

首先，我们要知道，什么是颜色空间。

参考百度百科：

> 颜色空间也称彩色模型（又称彩色空间或彩色系统）它的用途是在某些标准下用通常可接受的方式对彩色加以说明。

所以说，颜色空间是对色彩的一种说明方式。

举个栗子：最常用的颜色空间是RGB颜色空间。熟悉吧！它通过R，G，B三个分量来描述颜色信息。我们想要使用的LAB颜色空间和RGB不同的是，它的一个分量是亮度，这样我们就可以得到去除亮度这个分量的图片，减弱亮度对我们做图像处理的影响。

LAB颜色空间的三个分量分别是：

* **L**​*代表亮度*
* **a**​*代表从绿色到红色的分量*
* ***b***代表**从蓝色到黄色**的分量

​![Opencv实战：用一面镜子实现触屏功能（3）图像处理_image_1](assets/Opencv实战：用一面镜子实现触屏功能（3）图像处理_image_1-20240422183126-ngozcsn.jpg)  
（图源百度）

在opencv中，我们只用一行语句就可以实现这种变换

```python
img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
```

### 高斯滤波

滤波(blur)操作是一种基于邻域的图像平滑方法。  
当图像噪声只是图像的一小部分时，用某一像素点的邻域进行变换得到的新的像素点可以减小噪声的影响，从而很好的平滑噪声。  
直接对中心点的邻域求算数平均的方法称作**均值滤波**，求中值的方法称为**中值滤波**，而高斯滤波对图像邻域中的点赋予了权重，可以视作对均值滤波的改进。

opencv提供了高斯滤波相关函数：

```python
cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) → dst
```

上面这个是函数原型，我们只需要像下面这样调用就好了：

```python
img = cv2.GaussianBlur(img, (5, 5), 0)
```

这样调用可以用一个5\*5的矩阵（卷积核）对原图像做高斯滤波。

### 二值化

此时我们得到的图像是由L，A，B三张单通道图像组成的，是一个640\*480\*3，我们只需要用到其中与肤色比较相关的一维，所以我们可以用以下函数来分离出单通道图像：

```python
img = cv2.inRange(img[:, :, 2], np.array([50]), np.array([120]))
```

这里的img\[: , : , 2\]意味取出图像的第三个通道，即得到480\*640\*1的图像，这种单通道图像就是灰度图。

inRange函数是什么意思呢？这个函数的意思是根据图像每个像素的值来筛选像素点。这里填入的下限是50，上限是120，所以值处在50和120之间的点会被保留，这样就可以保留与皮肤色调相近的点，去除其他的点，如图：

​![Opencv实战：用一面镜子实现触屏功能（3）图像处理_image_2](assets/Opencv实战：用一面镜子实现触屏功能（3）图像处理_image_2-20240422183137-clur14c.png)​

### 开闭运算

图像的腐蚀，膨胀正如字面上的意思，腐蚀可以认为是给图像“减肥”，而膨胀就是给图像“增肥”。

* 先腐蚀后膨胀： 去除孤立的小点，毛刺
* 先膨胀后腐蚀：填平小孔，弥合小裂缝

可以认为膨胀就是把缺陷填补了，腐蚀就是把毛刺腐蚀掉了，但这样讲并不严谨，只是一种形象的理解，大家明白意思就好。

先腐蚀后膨胀的操作也叫做**开运算**。

```python
kernel = np.ones((4, 4), np.uint8)  # 卷积核
img = cv2.erode(img, kernel)  # 腐蚀
img = cv2.dilate(img, kernel) # 膨胀
```

## 3.轮廓检测

### 轮廓是什么？

引用opencv中文文档：

> 轮廓可以简单地解释为连接具有相同颜色或强度的所有连续点（沿边界）的曲线。

获得轮廓后，我们可以用一些函数得到轮廓的面积，上下顶点等信息，方便做下一步处理。

### 轮廓检测

opencv提供了简单易用的轮廓检测函数，可以快速从图像中分割出各个物体的轮廓。

```python
contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

这个函数可以自动从图像中检测轮廓，来看它的输入输出：

* contours：保存了所有得到的轮廓结果。
* hierarchy：轮廓层级相关，不深入介绍。
* binaryImage：传入的灰度图
* cv2.RETR_TREE：表示为检查到的轮廓建立一个层级树结构。
* cv2.CHAIN_APPROX_SIMPLE：表示仅保存轮廓的拐点信息。

### 检测手指

接下来，我们尝试找到手指的轮廓。

```python
cnt_list = []
for cnt in contours:
	center, size, angle = cv2.minAreaRect(cnt)
	if (35 < angle < 55) or (125 < angle < 145) or size[0] < 13 or size[1] < 25:  
	    continue  
	area = cv2.contourArea(cnt)  
	if area > 50000 or area < 300:  
	    continue
	cnt_list.append((cnt, area))
```

这段代码中，我们首先调用`cv2.minAreaRect(cnt)`​来用一个外接矩形逼近轮廓，得到外接矩形的大小和的朝向信息并进行了一次过滤。

利用`cv2.contourArea(cnt)`​对轮廓的大小作出限制，剔除过小和过大的轮廓。

最后，我们把合法的轮廓和面积大小存入列表备用。

```python
if len(cnt_list) >= 2:  
    cnt_list.sort(key=lambda x: x[1])  
    center1, size1, angle1 = cv2.minAreaRect(cnt_list[-1][0])  
    center2, size2, angle2 = cv2.minAreaRect(cnt_list[-2][0])  
    if center1[1] < center2[1]:  
        return cnt_list[-1][0], cnt_list[-2][0]
    else:  
        return cnt_list[-2][0], cnt_list[-1][0]
```

这里，我们首先要确保得到的轮廓数多于两个，不然检测到一侧的手指轮廓是无法得到坐标信息的。

然后我们调用列表的sort方法进行排序。这里的`lambda x: x[1]`​是一个匿名函数：输入列表，返回列表的第一个元素。

再往下是两次用外接矩形近似。这一段代码我写在一个函数中，所以需要按顺序返回两个轮廓。如何判断哪个是实际的手指，哪个是镜面中的手指呢？我采用了一种简单的方法：坐标靠上的是镜面中的手指，坐标靠下的是实际的手指。

最后，我们可以给轮廓打上不同的颜色，显示在画面上：

```python
cv2.drawContours(frame_flip, finger1, -1, (0, 255, 0), 3)  
cv2.drawContours(frame_flip, finger2, -1, (0, 0, 255), 3)
```

结果如图：![Opencv实战：用一面镜子实现触屏功能（4）轮廓检测_image_1](assets/Opencv实战：用一面镜子实现触屏功能（4）轮廓检测_image_1-20240422183207-1ikpahq.png)​

这种方法是有缺陷的：如果把手指完全贴到屏幕上，opencv就会把两部分手指识别为同一个轮廓！我们需要一种方法来分割轮廓。这里采用分水岭算法来简单解决这个问题。

### 分水岭算法

分水岭算法是一种模拟地理结构的算法。我们可以把灰度图想成一片陆地，每个像素的灰度值就是该点的海拔高度，灰度值较大的点连成的线是山脊，山脊之间会形成山谷。

向这片陆地注入水时，水会逐渐淹没山谷。随着水位的升高，两个山谷中的水会汇集在一起，而我们可以在山脊上修建大坝来阻止这种汇集。这些大坝连成的线，就是分水岭算法得到的图像分割线。

opencv中实现了`watershed()`​函数来实现分水岭算法，但是在使用这个函数之前，我们还要做一些处理。

使用我们之前做过一系列处理之后得到的二值化图像`binaryImage`​：

```python
# 卷积核
kernel = np.ones((3, 3), np.uint8)
# 背景
sure_bg = cv2.dilate(binaryImage, kernel, iterations=3)

dist_transfrom = cv2.distanceTransform(binaryImage, cv2.DIST_L2, 5)  
ret, sure_fg = cv2.threshold(dist_transfrom, 0.7 * dist_transfrom.max(), 255, 0)  
# 前景
sure_fg = np.uint8(sure_fg)  
# 未知区域
unknown = cv2.subtract(sure_bg, sure_fg)

# 得到掩膜
ret, markers = cv2.connectedComponents(sure_fg)  
markers = markers + 1  
markers[unknown == 255] = 0  
color_image = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR) 
# 分水岭算法
markers = cv2.watershed(color_image, markers)  
color_image[markers == -1] = (0, 0, 0)  
binaryImage = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

binaryImage = cv2.erode(binaryImage, kernel, iterations=2)
```

下面是这段代码的解释：

```python
sure_bg = cv2.dilate(binaryImage, kernel, iterations=3)
```

这里我们用膨胀得到图像的背景。膨胀扩大了物体的范围，可以保证得到的都是背景区域。如图：  
​![Opencv实战：用一面镜子实现触屏功能（6）算法优化_image_1](assets/Opencv实战：用一面镜子实现触屏功能（6）算法优化_image_1-20240422183407-xfoouzd.png)​

接下来，我们需要得到手指所在的区域，即图像的前景，而背景和前景之间的区域为边界。分水岭算法可以为我们找到确定的边界。

```python
dist_transfrom = cv2.distanceTransform(binaryImage, cv2.DIST_L2, 5)  
ret, sure_fg = cv2.threshold(dist_transfrom, 0.5 * dist_transfrom.max(), 255, 0)  
# 前景
sure_fg = np.uint8(sure_fg)  
```

​`distanceTransform()`​函数为距离变换函数，可以得到一个和原图像等大的矩阵，其中每个像素的值为其到最近的背景像素的距离。

利用`threshold()`​函数，我们可以过滤出值较大的像素，这样就得到了我们的前景图。如下：  
​![Opencv实战：用一面镜子实现触屏功能（6）算法优化_image_2](assets/Opencv实战：用一面镜子实现触屏功能（6）算法优化_image_2-20240422183416-a9zcc1z.png)​

这一部分就是可以确定是手指的部分。

最后，我们用两个区域相减：

```python
unknown = cv2.subtract(sure_bg, sure_fg)
```

​![Opencv实战：用一面镜子实现触屏功能（6）算法优化_image_3](assets/Opencv实战：用一面镜子实现触屏功能（6）算法优化_image_3-20240422183421-9hoqohb.png)​

这一部分就是我们要用分水岭算法处理的部分。

我们用`connectedComponents()`​来创建一个掩膜。

```python
ret, markers = cv2.connectedComponents(sure_fg)  
markers = markers + 1  
markers[unknown == 255] = 0  
```

connectedComponents将传入的图像的白色区域视作前景，它用0来表示图像的背景，而我们对其加1，用1来表示背景，并用0标记unknow区域。

```python
color_image = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR) 
# 分水岭算法
markers = cv2.watershed(color_image, markers)  
color_image[markers == -1] = (0, 0, 0)  
binaryImage = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
```

​`watershed()`​函数需要传入一个三通道图像，所以我们把灰度图转成RGB图像。分水岭算法完成后，掩膜中值为-1的点就是分割线的位置。我们把这些区域置0，并转回灰度图：  
​![Opencv实战：用一面镜子实现触屏功能（6）算法优化_image_4](assets/Opencv实战：用一面镜子实现触屏功能（6）算法优化_image_4-20240422183426-ygsfy6t.png)​

这个分割线还不够明显，所以我们做两次腐蚀操作：

​![Opencv实战：用一面镜子实现触屏功能（6）算法优化_image_5](assets/Opencv实战：用一面镜子实现触屏功能（6）算法优化_image_5-20240422183430-yxhv28z.png)​

现在再做轮廓检测，就基本不会有轮廓重叠的问题发生了。

## 4.坐标处理

### 从图像中获取坐标

我们利用镜面上轮廓的最低点和实际轮廓的最高点来计算坐标。

就是下图中的top和bottom：  
​![Opencv实战：用一面镜子实现触屏功能（5）坐标处理_image_1](assets/Opencv实战：用一面镜子实现触屏功能（5）坐标处理_image_1-20240422183257-4g4y3vl.png)​

代码如下：

```python
bottom = tuple(finger_mirror[finger_mirror[:, :, 1].argmin()][0])  
top = tuple(finger_real[finger_real[:, :, 1].argmax()][0])
```

从外向里看：

* tuple是强制类型转换，转换成元组。
* argmin和argmax是numpy中的函数，在数学中，它们分别是“使式子最小/最大的取值”的意思，在这里也是类似的。利用这种方法，我们能分别得到轮廓最低点和最高点的坐标。

利用如下方法计算两个点之间的坐标：

```python
distance = math.pow((Point0[0] - PointA[0]), 2) + math.pow((Point0[1] - PointA[1]), 2)  
distance = math.sqrt(distance)
```

然后根据两个点之间的**距离是否小于阈值**来判断是否接触屏幕（我设置了50）。  
**注意**：手指贴在屏幕上时，opencv会把两部分的手指检测成一个轮廓，可以通过把距离设置的大一些，手指不直接接触屏幕来规避这个问题。

如果接触屏幕了，我们可以取这两个点的中点作为最终的位置。

```python
result = [int((top[0] + bottom[0]) / 2), int((top[1] + bottom[1]) / 2)]
```

为了方便后续操作，这里我用int类型存储。

利用上一节中出现过的cycle函数在这个点画一个圆，如下：

​![Opencv实战：用一面镜子实现触屏功能（5）坐标处理_image_2](assets/Opencv实战：用一面镜子实现触屏功能（5）坐标处理_image_2-20240422183304-t8ujsr2.png)​

### 触点抖动问题

测试时可以发现，触点的显示伴随着抖动。手指的触点并不能用一个点来精确定义，我们检测到的点的坐标会在小范围内波动。在摄像头拍到的图片中，因为是俯视角，所以触点在竖坐标上的移动量很小。这样的一段小距离映射到了整个触碰区域上时会被放大，所以我们需要削减这种抖动。

我们可以通过一次记录多个点，并求这些点的中心作为结果坐标来缓解这种抖动。

```python
def get_centers_of_points(points):  
    length = len(points)  
    if length > 0:  
        center = [sum([x[0] for x in points]) / length, sum([x[1] for x in points]) / length]  
    else:  
        center = [-1, -1]  
    return center
```

但是，由于环境光的变化等种种原因，有时我们会检测出一些非常奇怪的点。这些点会对点列的中心产生非常不利的影响，所以我们需要想一种办法来排除这些点。这个问题也叫**离群点检测**问题。

我们引入**Z-score**来解决这个问题。

#### 什么是Z-score？

> z-score 也叫 standard score， 用于评估样本点到总体均值的距离。

z-score的计算公式很简单：

$$
z = \frac{{x - \mu}}{\sigma}
$$

其中$\mu$是平均值，$\sigma$是样本标准差。

##### 代码实现

我们去除z-score大于2.2的点。

```python
import numpy as np

def get_point:
	center = get_centers_of_points(points)  
	distant = []  
	for point in points:  
	    distant.append(get_distance_point(point, center))  
	distant_std = np.std(distant)
	result_point = []  
  
	for i in range(len(points)):  
  
    zscore = distant[i] / distant_std  
    if not zscore > 2.2:  
        result_point.append(points[i])  
  
  
	return get_centers_of_points(result_point)
```

​`get_distance_point()`​是计算两个点之间距离的函数，之前讲过如何实现。  
这里，我们用numpy库提供的`numpy.std()`​方法来计算标准差。

去除掉离群点后，我们重新计算样本中心点作为最后结果。

### 单应性变换

什么是单应性变换呢？不严谨的说，单应性变换可以理解成空间中一个面到另一个面的投影关系。

可以看这篇文章来了解单应性矩阵：  
https://blog.csdn.net/qq_40918859/article/details/123774719

当然，opencv为我们提供了十分方便的函数来计算单应性矩阵，所以不想了解也可以直接跳过。

#### opencv中的单应性变换

```python
Homography, status = cv2.findHomography(points_mirror, points_window)
```

调用这个函数，我们就能轻松得到单应性矩阵，其中：

* Homography是我们最终得到的3\*3的单应性矩阵
* points_mirror是摄像头中获取到的一组点（就是上面计算出的点），格式为numpy数组
* points_window为这组点在屏幕上的对应坐标，也是numpy数组

我们可以制作一个简单的界面来获取points_mirror和points_window这两个点列：

```python
import win32gui  
import win32con  
import win32print
# 这个函数用来获取屏幕分辨率
def get_scree_size():  
    hDC = win32gui.GetDC(0)  
    b = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)  
    a = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)  
    return a, b


cv2.namedWindow('window', cv2.WINDOW_NORMAL)  
# 设置一个窗口
cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# 设置窗口为全屏

a, b = get_scree_size()
window = np.ones((a, b, 3), dtype=np.float32)
points_window = np.array([设置几个点，如[1,1]], dtype=np.int32)
points_mirror = []
for point in points_window:
	cv2.circle(Homography_win, point, 5, (0, 255, 0), -1, cv2.LINE_AA)

	# 这里省略，利用上面提到的方法获取触点坐标

# 计算单应性矩阵
Homography, status = cv2.findHomography(points_mirror, points_window)

```

假设我们已经得到了单应性矩阵，我们先来学习如何把摄像头图像中得到的点转换到屏幕上。

```python
H_inv = np.matrix(Homography)
point = np.array([[point_mirror[0]], [point_mirror[1]], [1]])
result = np.dot(H, point)
result = result * (1.0 / result[2][0])
new_point = [result[0][0], result[1][0]]
```

第一句`np.matrix(Homography)`​是求逆矩阵的意思。

接下来，我们把原来的坐标转换为numpy数组并添加了一维坐标，这是因为单应性矩阵是一个3\*3的矩阵，我们需要用一个三维列向量与它相乘。

​`np.dot()`​就是矩阵乘法的意思。

最后，我们把向量的第三维重新化为1并分离出前两维作为结果。

像上面那样制作一个全白页面，显示出触点，最终效果如下：

​![Opencv实战：用一面镜子实现触屏功能（5）坐标处理_image_3](assets/Opencv实战：用一面镜子实现触屏功能（5）坐标处理_image_3-20240422183317-vj0699s.png)​

## 5.模拟点击

我们用pynupt库来模拟鼠标的点击。

```python
from pynput import mouse
# 获取控制对象
control = mouse.Controller()

# 设置鼠标位置
control.position = (100, 100)

# 模拟左键按下
control.press(mouse.Button.left)  
# 模拟左键松开
control.release(mouse.Button.left)
```
