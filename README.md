面向垃圾池的三维高精度动态建模
====
为了解决传统多视图立体视觉（multi-view stereo， MVS）和基于学习的MVS的重建方法仅能实现表面三维重构，而无法获取堆放垃圾的内部结构、发酵程度等深层次信息的问题，提出了一种融合了三维重建、点云配准以及时空过程演化模拟的垃圾池三维高精度动态建模方法。
概述

在焚烧发电行业中，垃圾池的管理问题是目前待解决的关键问题。高热值的垃圾入炉可以减少设备的调度难度，延长使用寿命，增大燃烧发电效益。但是，高热值的垃圾一直无法做到精细评估。行业内部分企业已对垃圾池进行了技术改造，并建立了垃圾池数字化模拟模型，供管理员复盘垃圾池内的垃圾变化情况。然而研究发现，传统多视图立体视觉（multi-view stereo， MVS）和基于学习的 MVS的重建方法，更关注于物体表面重建，无法获取垃圾池内部结构，无法全面了解垃圾池的状态信息。为弥补这一研究空白，我们选择研究面向垃圾池的三维高精度动态建模作为本文研究的主要问题。
实验
我们的论文是通过pytorch实现完成的，python=3.10.16 CUDA=11.3 requirements请见文件夹Time-space.
实验数据
<img width="458" height="311" alt="image" src="https://github.com/user-attachments/assets/f091eedb-61a0-4fb2-be63-50027f303765" />
多层点云配准结果
<img width="324" height="243" alt="image" src="https://github.com/user-attachments/assets/ca10e648-5b19-4297-a7b5-e8ed9834e7ed" />
时空演化模型质量高度变化曲线
<img width="865" height="307" alt="image" src="https://github.com/user-attachments/assets/913c70fa-0aa6-468a-8096-a462f5b6d2dc" />
夏季时空序列图
<img width="614" height="567" alt="image" src="https://github.com/user-attachments/assets/e5e7fd77-e433-4bc4-a678-37616c9e9143" />

引文
如果您想在工作中使用本文，请引用本文
田冬生，张思庆，陈启丽，等. 面向垃圾池的三维高精度动态建模[J].北京信息科技大学学报(自然科学版), 2025, 40(4): 12-21.TIAN D S, ZHANG S Q, CHEN Q L, et al. High-accuracy 3D dynamic modeling for waste pools[J]. Journal of Beijing Information Science & Technology University (Science and Technology Edition), 2025, 40(4): 12-21.
许可证
北京信息科技大学
