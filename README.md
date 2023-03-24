# FAST-LIVO-CC_Comments

## Introduction

本仓库是 [FAST-LIVO](https://github.com/hku-mars/FAST-LIVO) 的详细中文注释。



## Done

- 整体流程看完
- LiDAR 和 Image 数据时间戳同步看完，并发现一些问题
- VIO 系统看完，包括视觉子地图的选择，当前帧 patch 和地图点 patch 的关联，直接光度对齐的 EsIKF 优化
- LIO 部分添加了数据时间同步的注释，但是没有添加算法部分注释，这部分注释可见 [r2live-CC_Comments](https://github.com/Cc19245/r2live-CC_Comments) 中的 LIO 部分

- 关于 LIO 部分的 EsIKF 公式推导，可见我的系列博客：[Kalman Filter in SLAM 系列文章](https://blog.csdn.net/qq_42731705/article/details/129425086)



## TODO

- [ ] 点云去畸变部分写的很乱，而且发现一些去畸变的时候对时间的处理问题，待查
- [ ] ~~整理 EsIKF 公式推导成电子版并发布~~ See **Done**



## Note

- 本仓库的环境链接库



## Acknowledgements

- [FAST-LIVO](https://github.com/hku-mars/FAST-LIVO) 

- [fast-livo](https://github.com/hr2894235132/fast-livo)

