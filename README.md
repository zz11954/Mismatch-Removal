# Awesome-Mismatch-Removal

[TOC]
>Methods for eliminating mismatches can be broadly classified into manual methods and deep learning-based methods. Manual methods can be further classified into resampling-based methods, non-parametric model-based methods, and relaxed geometric constraint-based methods, each of which relies on different fundamentals.

## Manual Methods

### Resampling-Based Methods

- [RANSAC] Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography, 1981 [[pdf]](https://apps.dtic.mil/dtic/tr/fulltext/u2/a460585.pdf) [[wiki]](https://en.wikipedia.org/wiki/Random_sample_consensus)
- [MLESAC] MLESAC: A new robust estimator with application to estimating image geometry, CVIU'2000 [[pdf]](http://www.academia.edu/download/3436793/torr_mlesac.pdf) [[code_pcl]](https://github.com/PointCloudLibrary/pcl/tree/master/sample_consensus/include/pcl/sample_consensus)
- [NAPSAC]Napsac: High noise, high dimensional robust estimation-it’s in the bag，BMVC'2002[[pdf]](https://d1wqtxts1xzle7.cloudfront.net/51485218/NAPSAC_High_Noise_High_Dimensional_Robus20170123-23393-jxxcl1-libre.pdf?1485199178=&response-content-disposition=inline%3B+filename%3DNapsac_High_Noise_High_Dimensional_Robus.pdf&Expires=1711038934&Signature=Sk4FynbP~i6Ldh3fCffGfi2G-0MayDfLfkGaavAFkFyLhNRaM5VSccpsnqz3yXUos7Phy8cwLwMAjCZo3ye-yfCdEZVS3lQklsK1hdWyzwTgI9YUXHaI3xzYHNrQlIpa-4tSWIem9BP4oBCC7DDKoa5LOKuFXlRso98JBssYx7rz8SI5Ash2zYarn77~llu4YMKVb-dCnPZ6pJaD9wpMqt55JgsNd4Qs49iBVjR5XesQ8eD-V8Ol~Uag~UDac8aHx7TGQzkl7Nn8yHhzeYPwLSCuFN8-VupY94mZQPfkVl9TMhycSiBga45zAovHjGWgP9wRfLUhdbbkRXVeyLmWAg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
- [LO-RANSAC] Locally Optimized RANSAC,[[pdf]],2003,(https://cmp.felk.cvut.cz/~chum/papers/chum-DAGM03.pdf)
- [PROSAC] Matching with PROSAC-progressive sample consensus, CVPR'2005 [[pdf]](https://dspace.cvut.cz/bitstream/handle/10467/9496/2005-Matching-with-PROSAC-progressive-sample-consensus.pdf?sequence=1) [[code_pcl]](https://github.com/PointCloudLibrary/pcl/tree/master/sample_consensus/include/pcl/sample_consensus)
- [Groupsac] GroupSAC: Efficient consensus in the presence of groupings,ICCV'2009 [[pdf]](https://ieeexplore.ieee.org/document/5459241)
- [USAC] A universal framework for random sample consensus.TPAMI'2012,[[pdf]](https://ieeexplore.ieee.org/document/6365642)
- [EVSAC] EVSAC: Accelerating Hypotheses Generation by Modeling Matching Scores with Extreme Value Theory, ICCV'2013 [[pdf]](https://ieeexplore.ieee.org/document/6751418)
- [GC-RANSAC] Graph-Cut RANSAC, CVPR'2018 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Barath_Graph-Cut_RANSAC_CVPR_2018_paper.pdf) [[code]](https://github.com/danini/graph-cut-ransac)
- [MAGSAC] MAGSAC: Marginalizing Sample Consensus, CVPR'2019 [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Barath_MAGSAC_Marginalizing_Sample_Consensus_CVPR_2019_paper.pdf) [[code]](https://github.com/danini/magsac)
- [P-NAPSAC] Progressive NAPSAC: sampling from gradually growing neighborhoods,2019[[pdf]](https://arxiv.org/abs/1906.02295)
- [MAGSAC++] MAGSAC++, a fast, reliable and accurate robust estimator,CVPR'2020[[pdf]](https://arxiv.org/abs/1912.05909)
- [MAGSAC++] Marginalizing Sample Consensus, TPAMI'2022 [[pdf]](https://ieeexplore.ieee.org/document/9511155) [[code]](https://github.com/danini/magsac)
  
### Non-parametric Model-Based Methods
- Fast non-rigid surface detection, registration and realistic augmentation.,IJCV'2007[[pdf]](https://vincentlepetit.github.io/files/papers/comp_pilet_ijcv07.pdf)
- Direct estimation of nonrigid registrations with image-based self-occlusion reasoning.,ICCV'2007,[[pdf]](https://ieeexplore.ieee.org/document/4408989)
- [ICF/SVR] Rejecting mismatches by correspondence function, IJCV'2010 [[pdf]](http://www.nlpr.ia.ac.cn/2010papers/kz/gk24.pdf)
- Regularized vector field learning with sparse approximation for mismatch removal,PR'2013[[pdf]](https://pages.ucsd.edu/~ztu/publication/pr13_robustmatching.pdf)[[code]](https://github.com/jiayi-ma/VFC)
- [RPM-VFC] Robust Point Matching via Vector Field Consensus, TIP'2014 [[pdf]](http://or.nsfc.gov.cn/bitstream/00001903-5/99530/1/1000009269450.pdf) [[code]](https://github.com/jiayi-ma?tab=repositories)
- Robust l2e estimation of transformation for non-rigid registration,TSP'2015,[[pdf]](https://www.researchgate.net/profile/Jiayi_Ma/publication/273176803_Robust_L2E_Estimation_of_Transformation_for_Non-Rigid_Registration/links/552142f10cf2f9c130512304.pdf) [[code]](https://github.com/jiayi-ma?tab=repositories)
- [LLT] Robust feature matching for remote sensing image registration via locally linear transforming, TGRS'2015 [[pdf]](https://yuan-gao.net/pdf/TGRS2015.pdf) [[code]](https://github.com/jiayi-ma?tab=repositories)
- Feature guided Gaussian mixture model with semi-supervised em and local geometric constraint for retinal image registration,IS'2017[[pdf]](https://sci-hub.ru/10.1016/j.ins.2017.07.010)
- [RPM-MR] Nonrigid Point Set Registration with Robust Transformation Learning under Manifold Regularization, TNNLS'2019 [[pdf]](https://pdfs.semanticscholar.org/e8c9/75165ffc5af6cad6961b25f29ea112ae50dd.pdf) [[code]](https://github.com/jiayi-ma?tab=repositories)

### Relaxed Methods
- A spectral technique for correspondence problems using pairwise constraints,ICCV'2005,[[pdf]](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/leordeanu-iccv-05.pdf)
- Common visual pattern discovery via spatially coherent correspondences,CVPR'2010,[[pdf]](https://sci-hub.ru/10.1109/cvpr.2010.5539780)
- Feature matching with bounded distortion,TOG'2014,[[pdf]](https://sci-hub.ru/10.1145/2602142)
- 

## Learning-based Methods

