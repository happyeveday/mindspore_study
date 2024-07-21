# MindSpore By Yang
## 2024/7/21
### 架构
![1721453862712](https://github.com/user-attachments/assets/e4ce216c-b74d-400b-a5f6-960d9d83cab2) 

由下至上：硬件层->硬件计算加速平台(例如CANN、CUDA、CPU)->编译层MindSpore架构->各个API接口->各方面实际应用 
### 接口学习
①vision.Rescale() 
举例：vision.Rescale(1.0/255.0,0) 
解释：1.0/255.0：即sclae=1.0/255.0，将图像的每个像素值除以255.0；0：shift=0，偏移量为0 
②vision.Normalize() 
举例：vision.Normalize(mean=(0.1307,), std=(0.3081,))
解释：以均值为mean，标准差为std，代入归一化公式将图像进行归一化处理 
③vision.HWC2CHW() 
解释：H为高度(Height)，W为宽度(Width)，C为通道(Channel)，这个API是将图像由OpenCV等图像处理库产生的原始图像格式HWC，转化为适合PyTorch、MindSpore、Tensorflow等机器学习架构处理的CHW图像格式 
### 相关公式
new_pixel_value=(old_pixel_value×scale)+shift，对应vision.Rescale 
归一化公式：normalized_pixel = (pixel - mean) / std，其中pixel为图像的原像素值，在使用归一化前一般将像素值调至(0,1.0)；mean为像素值的均值，std为像素值的标准差
