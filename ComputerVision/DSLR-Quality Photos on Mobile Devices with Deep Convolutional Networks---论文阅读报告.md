# DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks（ICVV 2017)---论文阅读报告

[TOC]

[论文地址](https://arxiv.org/pdf/1704.02470.pdf)  [作者项目地址](http://people.ee.ethz.ch/~ihnatova/#title)

###   一、选题概览

> 提高图像增强算法，我首先联想到的是其在手机摄像方面的应用。尽管内置智能手机相机的质量迅速提高，但它们的物理限制——传感器尺寸小、镜头紧凑以及缺乏特定的硬件——阻碍了它们达到数码单反相机的质量效果。于是，通过软件方法达到图像效果增强尤为重要。而手机照片增强任务复杂多变，难以用传统计算机图形学方法达到泛用解，所以，我选择以深度学习方法，作为选题方向，并选择这篇论文进行深入。

在这篇论文中，作者提出了一种端到端的深度学习方法，以此将普通移动设备照片品质转换为数码单反质量(DSLR-quality)的图像品质，达到图像增强效果。

在这片论文中，同其他深度学习算法一样，最重要的两部分——网络架构与损失函数如下:

- **残差卷积神经网络结构**   

  个人理解：该网络可以看作一个GANs(生成对抗网络)，一边变换生成图片，一边判断。

  <img src="https://s2.loli.net/2024/03/18/lvIWLaoqmRQtxCO.jpg" style="zoom:67%;" />

  

- **复合损失函数：** 在图像增强这一任务中，一般的损失函数(MSE/MAE/Cross Entrop Loss/NLL等)均无法使用。为此，就像YOLO算法损失函数由边界框损失、类别预测损失、目标存在概率损失三部分构成一样，论文作者提出了复合颜色损失(Color loss)、纹理损失(Texture loss)、推算内容损失（Content loss)的损失函数。先分别计算三个loss，

  为了使图像变换更平滑，还引入了一个变换损失(Total variation lo)

  再进行复合，并将复合损失传递给神经网络。

  ![image-20240317162159564](https://s2.loli.net/2024/03/18/WX4tPuZha5esKn3.png)

 

### 二、数据集DSLR Photo Enhancement Dataset

为了解决从智能手机相机拍摄的低质量图像到专业数码单反相机拍摄的高质量图像的图像转换问题，论文引入了一个大规模的现实世界数据集，DSLR Photo Enhancement Dataset(DPED)。

DPED由三部智能手机和一台数码单反相机在野外同步拍摄的照片(为了确保所有相机同时拍摄照片，这些设备被安装在一个三脚架上，并通过无线控制系统远程激活)组成。用于收集数据的设备下表所示，图中示例如下：

<img src="https://s2.loli.net/2024/03/18/9xzSvrDVPRIBwQT.png" alt="image-20240317183116349" style="zoom:67%;" />

<img src="https://s2.loli.net/2024/03/18/CVpnPthTEBjQrgK.png" alt="image-20240317183044059" style="zoom:80%;" />

总共，在3周内收集了22K张照片，其中索尼智能手机4549张照片，iPhone 5727张照片，佳能和黑莓相机6015张照片。这些照片是白天在各种不同的地方、在不同的照明和天气条件下、由自动模式拍摄的。在整个拍摄过程中，所有图片均使用了各相机的默认设置。

**问题：**

**Matching algorithm**

如上图Figure 3所示，由于摄像机的视角和位置不同，同步拍摄的图像并没有完全对齐，图片分辨率也有不同。为解决这个问题，论文执行了额外的非线性变换，得到了一个固定分辨率的图像，作为网络的输入。

*算法如下：*

1. 对于每个(phoneDSLR)图像对，计算并匹配图像上的SIFT关键点。并使用RANSAC（Random Sample Consensus）估计单应性。

2. 裁剪两幅图像到相交部分，并缩小DSLR图像尺寸到手机屏幕尺寸的大小

   效果如下：

   <img src="https://s2.loli.net/2024/03/18/XSPscgriaVZpFRx.png" alt="image-20240317184249819" style="zoom: 67%;" />

在对齐的高分辨率图像上训练CNN是不可行的，因此从这些照片中提取大小为100×100px的patch。据论文作者的初步实验表明，更大的patch大小并不会带来更好的性能，同时需要更多的计算资源。

- 使用非重叠滑动窗口提取patch

- 窗口沿着来自每个手机-DSLR图像对的两个图像平行移动，并且它在手机图像上的位置通过基于相互关联度量的移位和旋转进行额外调整.

- 为了避免明显的位移(导致变换对应的图像不匹配)，数据集中只包含相互关系大于0.9的补丁。大约100张原始图像被保留用于测试，其余的照片用于训练和验证。

- 通过这个过程，黑莓-佳能、iphone -佳能和索尼-佳能对的培训补丁分别为139K、160K和162K，测试补丁分别为2.4-4.3K。

  **It should be emphasized that both training and test patches are precisely matched, the potential shifts do not exceed 5 pixels. In the following we assume that these patches of size 3×100×100 constitute the input data to our CNNs.**

  需要强调的是，训练和测试补丁都是精确匹配的，潜在的位移不超过5个像素。下面假设这些大小为3×100×100的patch构成cnn的输入数据，并开始Mothed介绍



###  三、Mothed 论文方法介绍 

#### 1. task任务说明

给一个低质量照片$I_s$(源图像)，希望通过一种图像增强手段将其转换到类似单反相机拍摄的高质量图片$I_t$

引入一个深度残差CNN网络 $F_W$(模型权重参数为$W$),来学习$I_s$到$I_t$​的转换函数，使得训练集$F_W(I_s),I_t$，两者的复合损失最小

<img src="https://s2.loli.net/2024/03/18/pl9crduTCJV2DeH.png" alt="image-20240317185708026" style="zoom:67%;" />

#### 2. Loss function设计

##### 2.1 Color loss颜色损失

为了测量增强图像和目标图像之间的Color loss，论文建议应用高斯模糊(见图5)并计算得到的表示之间的欧几里得距离。在cnn的使用下，这相当于使用一个额外的卷积层，其中有一个固定的高斯核，后面跟着均方误差(MSE)函数。Color loss可以写成:<img src="https://s2.loli.net/2024/03/18/IDBaPeOR7uovC5j.png" alt="image-20240317190100545" style="zoom:80%;" />

这种损失背后的思路是评估图像之间的亮度，对比度和主要颜色的差异，同时消除Texture和Content的比较。因此，作者通过视觉检测固定一个常数σ作为最小值，以确保纹理和内容被丢弃。

如下图演示了，图像对(X, Y)的MSE和颜色损失，其中Y = X在随机方向上移动了n个像素<img src="https://s2.loli.net/2024/03/18/6fIe1uHUb8pGtzO.png" alt="image-20240317190354989" style="zoom:50%;" />

由此可以观察到，color loss对小的失真(小于等于2像素)几乎不敏感。对更高的位移(3-5px)，它仍然比MSE小5-10倍，而对于更大的位移，它表现出类似的幅度和行为。由此说明color loss函数设计的高效：**迫使增强图像具有与目标图像相同的颜色分布，同时容忍小的像素位置不匹配**。

##### 2.2 Texture Loss纹理损失

Texture Loss不用像Color Loss一样用预先设定的损失函数，而是使用**生成式对抗网络**(GANS)来直接学习测量纹理质量。

训练过程：观察假的(低质图像改进后的图像)和真实的(目标的高质量图像)，来预测输入的图像是否是‘真实的’。

通过训练，GANS最小化交叉熵损失函数，由此可将纹理损失定义为GANS的生成目标:

其中Fw和D分别表示生成器网络参数和鉴别器网络参数。

鉴别器是在{phone, DSLR}图像对上预训练，然后与所提出的生成器联合训练，作者这里还是用了GANs的常规设计方法

<img src="https://s2.loli.net/2024/03/18/MNgq8dI9in1YJQX.png" alt="image-20240317191927588" style="zoom:67%;" />

##### 2.3 Content Loss内容损失

根据预训练的VGG-19网络ReLU层产生的激活图来定义内容损失Content Loss。

这种通过网络激活函数产生的损失不是对图像之间的逐像素测量差异，而是鼓励它们具有相似的特征表示，包括其内容和感知质量的各个方面。

总结：

设ψj()为函数的VGG-19 CNN网络第j层卷积后得到的特征映射，内容损失定义为增强图像的特征表示与目标图像之间的欧氏距离:

![image-20240317192614520](https://s2.loli.net/2024/03/18/MkNdFxOTl614XtQ.png)

#####  2.4 Total variation loss 总共变换损失

除了之前的损失外，作者还添加了总变化(TV)损失，以增强生成图像的空间平滑性：

![image-20240317192923417](https://s2.loli.net/2024/03/18/6vmrqH2S8a5flTu.png)

##### 2.5 Total loss (最终复合，总的损失函数)

最终损失被定义为以前损失的加权和：

![image-20240317193033575](https://s2.loli.net/2024/03/18/pWUfmYyjN1waCnt.png)

### 四、 生成、鉴别CNN网络设计

![image-20240317193140386](https://s2.loli.net/2024/03/18/uiGLEgl63owbSxA.png)

##### 1. 图像增强网络Image enhancement network

这个图像变换网络是完全卷积的，从9×9层开始，然后是四个残差块：每个残差块由两个3×3层组成，与批处理归一化层交替。

在残差块之后使用两个额外的层，其卷积核大小为3×3，另一个层的核大小为9×9。

CNN转换网络中的所有层都有64个通道，并且后面跟着一个ReLU激活函数，除了最后一层用a scaled tanh来输出结果。

##### 2. discriminator CNN鉴别网络

由五个卷积层组成，每个层都有一个LeakyReLU非线性和批处理归一化比率。

第一层、第二层和第五层的卷积步长分别为4、2和2。

最后将sigmoid激活函数应用于包含1024个神经元的最后一层全连接层的输出，并产生输入图像被目标单反相机拍摄的概率。

##### 3. VGG-19网络

输出$Loss_{content}$的网络，为经典VGG-19网络

![VGG-19 Convolutional Neural Network - All about Machine Learning](https://s2.loli.net/2024/03/18/nSQNU97KLZ2k3oF.png)

### 五、训练设计与实验结果

##### 1.训练设计

这个网络在NVidia Titan X GPU上进行了20K次迭代训练，批处理大小为50，采用Adam修正的随机梯度下降法进行优化，学习率为5e-4。对所有所有相机整个网络架构和实验设置相同。

> 由于我在网上能承担的服务器，最多只有P100，且算力有限，我没能完成对整个实验过程的复现。

##### 2. 实验结果

![image-20240317195247570](https://s2.loli.net/2024/03/18/bVuOYI2Te7ioalc.png)

可以看到，论文作者网络(倒数第二张)图与DSLR高质量图间的差异已经较小。

并且，论文作者将该方法与其他图像增强方法比较，达到以下优秀的结果：

![image-20240317195912716](https://s2.loli.net/2024/03/18/45DZpc9fbs7EtiG.png)

同时，将该网络增强后图像、人工增强后图像、与实际高质量DSLR图像混杂一起，请人们判别，发现人们已难以区分其中差别：

![image-20240317200342591](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240317200342591.png)

##### 3.该方法尚存的限制

由于这个网络的增强过程是全自动的，一些缺陷是不可避免的。

1. 颜色偏差（(见图12第一张图像中的地面/山脉，如第二张图的沥青）

2. 噪声放大——GANs网络的通病（可以有效地恢复高频信号，也导致了高频噪声的放大）

   如一下第二、第三张图片

3. 需要以匹配的源/目标训练**图像对**的形式进行强监督，这使得其他摄像机重复该过程变得繁琐。见DPED数据集的创建

   作者对此提出了一种弱监督改进方法：

   > A. Ignatov, N. Kobyshev, R. Timofte, K. Vanhoey, and L. Van Gool. Wespe: Weakly supervised photo enhancer for digital cameras. 2017

<img src="https://s2.loli.net/2024/03/18/FfxoP268EdsAKNh.png" alt="image-20240317200610710" style="zoom:80%;" />

### 六、论文总结

为了有效地将普通智能手机的相机转换为高质量的单反相机，作者提出了一种图像增强解决方案。

这个端到端深度学习方法使用了一个复合的感知误差函数，结合了内容、颜色和纹理损失。

为了训练和评估所提出的方法，引入了DPED——一个大规模数据集，由三种不同的手机和一个高端反射相机拍摄的真实照片组成，并提出了一种有效的校准图像的方法，使其适合于图像到图像的学习。

定量和定性评估表明，增强后的图像质量与DSLR拍摄的照片相当，并且该方法本身可以应用于各种质量级别的相机

### 七、个人阅读报告总结

#### 1. 我学到了什么？

我学到了一种端到端图像增强网络(GANS),由生成网络，鉴定网络，VGG-19三部分组成，并知道了具体的网络细节。

同第一次见YOLO算法一样，我学到了在图像增强这个任务里一种可行的将颜色、纹理、内容、变换四者复合的一种可行的损失函数。

在未来，我能参考以上，自己设计图像增强网络，并根据使用本论文提到的损失函数方法，进行网络训练。

我学到了图像增强数据集的建立过程，了解了其中的困难，并知道作者通过一种数据图像变换的方法，从最初拍摄的图像，转成训练网络所需要的patch。

了解了对图像增强方法好坏的评估过程，同时对论文写作有了进一步了解。

了解了就tensorflow对论文网络的搭建。

####  2. 扩展：源码阅读

我在github上找到了基于tensorflow框架搭建的网络[项目地址](https://github.com/aiff22/DPED)，但没有合适的设备训练。另外，由于该网络使用tensorflow1.x，而我本机和服务器均为tensorflow2.4,不方便使用，所以，目前只停留在源码阅读层面。以后准备自己用比较熟悉的pytorch框架复现。

##### 生成网络

```python
def resnet(input_image):  
    """  
    定义一个简化的ResNet卷积神经网络模型。  
      
    参数:  
        input_image (tf.Tensor): 输入的图像数据，形状为[batch_size, height, width, channels]。  
      
    返回:  
        tf.Tensor: 经过网络处理后的增强图像。  
    """  
      
    # 使用TensorFlow的变量作用域来组织模型变量  
    with tf.compat.v1.variable_scope("generator"):  
          
        # 第一个卷积层  
        W1 = weight_variable([9, 9, 3, 64], name="W1")  # 定义权重变量  
        b1 = bias_variable([64], name="b1")  # 定义偏置变量  
        c1 = tf.nn.relu(conv2d(input_image, W1) + b1)  # 应用卷积和ReLU激活函数  
          
        # 第一个残差块  
        W2 = weight_variable([3, 3, 64, 64], name="W2")  
        b2 = bias_variable([64], name="b2")  
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))  # 卷积后应用实例归一化和ReLU  
          
        W3 = weight_variable([3, 3, 64, 64], name="W3")  
        b3 = bias_variable([64], name="b3")  
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3)) + c1  # 残差连接  
          
        # 第二个残差块  
        W4 = weight_variable([3, 3, 64, 64], name="W4")  
        b4 = bias_variable([64], name="b4")  
        c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))  
          
        W5 = weight_variable([3, 3, 64, 64], name="W5")  
        b5 = bias_variable([64], name="b5")  
        c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5)) + c3  # 残差连接  
          
        # 第三个残差块  
        W6 = weight_variable([3, 3, 64, 64], name="W6")  
        b6 = bias_variable([64], name="b6")  
        c6 = tf.nn.relu(_instance_norm(conv2d(c5, W6) + b6))  
          
        W7 = weight_variable([3, 3, 64, 64], name="W7")  
        b7 = bias_variable([64], name="b7")  
        c7 = tf.nn.relu(_instance_norm(conv2d(c6, W7) + b7)) + c5  # 残差连接  
          
        # 第四个残差块  
        W8 = weight_variable([3, 3, 64, 64], name="W8")  
        b8 = bias_variable([64], name="b8")  
        c8 = tf.nn.relu(_instance_norm(conv2d(c7, W8) + b8))  
          
        W9 = weight_variable([3, 3, 64, 64], name="W9")  
        b9 = bias_variable([64], name="b9")  
        c9 = tf.nn.relu(_instance_norm(conv2d(c8, W9) + b9)) + c7  # 残差连接  
          
        # 额外的卷积层  
        W10 = weight_variable([3, 3, 64, 64], name="W10")  
        b10 = bias_variable([64], name="b10")  
        c10 = tf.nn.relu(conv2d(c9, W10) + b10)  # 归一化和残差连接  
          
        W11 = weight_variable([3, 3, 64, 64], name="W11")  
        b11 = bias_variable([64], name="b11")  
        c11 = tf.nn.relu(conv2d(c10, W11) + b11)  # 归一化和残差连接  
          
        # 输出层  
        W12 = weight_variable([9, 9, 64, 3], name="W12")  # 注意这里的卷积核大小是9x9  
        b12 = bias_variable([3], name="b12")  
        # 使用tanh激活函数，并缩放到[-1, 1]范围，然后偏移到[0, 1]范围（假设输入图像已经归一化到[0, 1]）  
        enhanced = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5  
      
    return enhanced  # 返回增强后的图像  
```

##### 对抗网络

```python
# 定义判别器网络  
def adversarial(image_):  
    # 在"discriminator"这个命名空间下创建变量  
    with tf.compat.v1.variable_scope("discriminator"):  
        # 一系列卷积层，提取输入图像的特征  
        conv1 = _conv_layer(image_, 48, 11, 4, batch_nn=False)  
        conv2 = _conv_layer(conv1, 128, 5, 2)  
        conv3 = _conv_layer(conv2, 192, 3, 1)  
        conv4 = _conv_layer(conv3, 192, 3, 1)  
        conv5 = _conv_layer(conv4, 128, 3, 2)  
          
        # 将最后一层卷积的输出展平  
        flat_size = 128 * 7 * 7  
        conv5_flat = tf.reshape(conv5, [-1, flat_size])  
          
        # 全连接层  
        W_fc = tf.Variable(tf.compat.v1.truncated_normal([flat_size, 1024], stddev=0.01))  
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))  
        fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)  
          
        # 输出层  
        W_out = tf.Variable(tf.compat.v1.truncated_normal([1024, 2], stddev=0.01))  
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))  
        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)  
      
    # 返回对抗网络的输出  
    return adv_out  
```

##### 其他函数

```python
# 权重变量的初始化  
def weight_variable(shape, name):  
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.01)  
    return tf.Variable(initial, name=name)  
  
  
# 偏置变量的初始化  
def bias_variable(shape, name):  
    initial = tf.constant(0.01, shape=shape)  
    return tf.Variable(initial, name=name)  
  
  
# 二维卷积操作  
def conv2d(x, W):  
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  
  
  
# Leaky ReLU激活函数  
def leaky_relu(x, alpha=0.2):  
    return tf.maximum(alpha * x, x)  
  
  
# 定义卷积层，包含卷积操作和激活函数  
def _conv_layer(net, num_filters, filter_size, strides, batch_nn=True):  
    weights_init = _conv_init_vars(net, num_filters, filter_size)  # 初始化权重  
    strides_shape = [1, strides, strides, 1]  # 设置步长  
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))  # 初始化偏置  
  
    # 进行卷积操作并加上偏置  
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME') + bias  
    # 使用Leaky ReLU激活函数  
    net = leaky_relu(net)  
  
    # 可选地，进行实例归一化  
    if batch_nn:  
        net = _instance_norm(net)  
  
    return net  
  
  
# 实例归一化操作  
def _instance_norm(net):  
    # 获取输入的形状信息  
    batch, rows, cols, channels = [i.value for i in net.get_shape()]  
    var_shape = [channels]  
  
    # 计算均值和方差  
    mu, sigma_sq = tf.compat.v1.nn.moments(net, [1, 2], keepdims=True)  
    # 定义可学习的尺度和平移参数  
    shift = tf.Variable(tf.zeros(var_shape))  
    scale = tf.Variable(tf.ones(var_shape))  
  
    epsilon = 1e-3  # 防止除零错误  
    # 执行归一化  
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)  
  
    # 应用尺度和平移  
    return scale * normalized + shift  
  
  
# 初始化卷积层的权重  
def _conv_init_vars(net, out_channels, filter_size, transpose=False):  
    # 获取输入的形状信息  
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]  
  
    # 设置权重的形状  
    if not transpose:  
        weights_shape = [filter_size, filter_size, in_channels, out_channels]  
    else:  
        weights_shape = [filter_size, filter_size, out_channels, in_channels]  
  
    # 初始化权重变量  
    weights_init = tf.Variable(tf.compat.v1.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)  
    return weights_init
```

##### VGG-19

经典网络。本片论文也给了我对经典网络(LeNet、AlexNet、VGGNet、InceptionNet和ResNet)的新认识，以前只会把它们用在简单的图像分类/回归任务中。

```python
def net(path_to_vgg_net, input_image):

    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = scipy.io.loadmat(path_to_vgg_net)
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        layer_type = name[:4]
        if layer_type == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif layer_type == 'relu':
            current = tf.nn.relu(current)
        elif layer_type == 'pool':
            current = _pool_layer(current)
        net[name] = current

    return net

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)

def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

def preprocess(image):
    return image - IMAGE_MEAN
```

##### 损失函数计算

如论文中提到的方法

```python
    # 1) texture (adversarial) loss

    discrim_target = tf.concat([adv_, 1 - adv_], 1)

    loss_discrim = -tf.reduce_sum(discrim_target * tf.compat.v1.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))
    loss_texture = -loss_discrim

    correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
    discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # 2) content loss

    CONTENT_LAYER = 'relu5_4'

    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_image * 255))

    content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size

    # 3) color loss

    enhanced_blur = utils.blur(enhanced)
    dslr_blur = utils.blur(dslr_image)

    loss_color = tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2))/(2 * batch_size)

    # 4) total variation loss

    batch_shape = (batch_size, PATCH_WIDTH, PATCH_HEIGHT, 3)
    tv_y_size = utils._tensor_size(enhanced[:,1:,:,:])
    tv_x_size = utils._tensor_size(enhanced[:,:,1:,:])
    y_tv = tf.nn.l2_loss(enhanced[:,1:,:,:] - enhanced[:,:batch_shape[1]-1,:,:])
    x_tv = tf.nn.l2_loss(enhanced[:,:,1:,:] - enhanced[:,:,:batch_shape[2]-1,:])
    loss_tv = 2 * (x_tv/tv_x_size + y_tv/tv_y_size) / batch_size

    # final loss

    loss_generator = w_content * loss_content + w_texture * loss_texture + w_color * loss_color + w_tv * loss_tv

    # psnr loss

    enhanced_flat = tf.reshape(enhanced, [-1, PATCH_SIZE])

    loss_mse = tf.reduce_sum(tf.pow(dslr_ - enhanced_flat, 2))/(PATCH_SIZE * batch_size)
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))
```

由此用tensorflow amdn进行反向传播,完成训练。

**生成器**

```python
train_step_gen = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)
```

**判别器**

```python
train_step_disc = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_discrim, var_list=discriminator_vars)
```

