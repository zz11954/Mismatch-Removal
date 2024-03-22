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
- [GroupSAC] GroupSAC: Efficient consensus in the presence of groupings,ICCV'2009 [[pdf]](https://ieeexplore.ieee.org/document/5459241)
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
- [BF] Bilateral Functions for Global Motion Modeling, ECCV'2014 [[pdf]](http://mftp.mmcheng.net/Papers/CoherentModelingS.pdf) [[project]](https://mmcheng.net/bfun/) [[code]](http://mftp.mmcheng.net/Data/eccv_2014_release.zip)
- [CODE] Code: Coherence based decision boundaries for feature correspondence, TPAMI'2018 [[pdf]](https://ora.ox.ac.uk/objects/uuid:0e5a62ab-fb69-472f-a1e1-49d49595db62/download_file?safe_filename=matching.pdf&file_format=application%2Fpdf&type_of_work=Journal+article) [[project]](http://www.kind-of-works.com/CODE_matching.html)
- [GLPM] Guided Locality Preserving Feature Matching for Remote Sensing Image Registration, TGRS'2018 [[pdf]](https://ieeexplore.ieee.org/abstract/document/8340808/)
- [LPM] Locality preserving matching, IJCV'2019 [[pdf]](https://link.springer.com/article/10.1007/s11263-018-1117-z) [[code]](https://github.com/jiayi-ma?tab=repositories)
- [GMS] GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence, CVPR'2017 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bian_GMS_Grid-based_Motion_CVPR_2017_paper.pdf) [[code]](https://github.com/JiawangBian/GMS-Feature-Matcher)
- [RFM-SCAN] Robust feature matching using spatial clustering with heavy outliers,TIP'2020,[[pdf]](https://starainj.github.io/Files/TIP2020-RFM-SACN.pdf)[[code]](https://github.com/StaRainJ/RFM-SCAN)
- Progressive feature matching: Incremental graph construction and optimization,TIP'2020,[[pdf]](https://sci-hub.ru/10.1109/tip.2020.2996092)[[code]](https://sites.google.com/view/sehyung/home/projects/progressive-feature-matching)

## Learning-based Methods
- Sinkhorn networks: Using optimal transport techniques to learn permutations,NIPS'2017,[[pdf]](http://www.stat.columbia.edu/~gonzalo/pubs/SinkhornOT.pdf)
- [N3Net] Neural Nearest Neighbors Networks, NeurIPS'2018 [[code]](https://github.com/visinf/n3net/)
- [LFGC] Learning to Find Good Correspondences, CVPR'2018 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1453.pdf) [[code]](https://github.com/vcg-uvic/learned-correspondence-release)
- [OANet] Learning Two-View Correspondences and Geometry Using Order-Aware Network ICCV'2019 [[code]](https://github.com/zjhthu/OANet)
- [LMR] LMR: Learning A Two-class Classifier for Mismatch Removal, TIP'2019 [[pdf]](https://ieeexplore.ieee.org/abstract/document/8672170/) [[code]](https://github.com/StaRainJ/LMR)
- [ACNe] ACNe: Attentive context normalization for robust permutation-equivariant learning, CVPR'2020[[code]](https://github.com/vcg-uvic/acne)
- [SuperGlue] SuperGlue: Learning Feature Matching with Graph Neural Networks, CVPR'2020 [[code]](https://github.com/magicleap/SuperGluePretrainedNetwork)
- [LSV-ANet] LSV-ANet: Deep Learning on Local Structure Visualization for Feature Matching,TGRS'2021,[[pdf]](https://ieeexplore.ieee.org/document/9377555)
- [GANet] Learning for mismatch removal via graph attention networks,ISPRS J PHOTOGRAMM'2022,[[pdf]](https://www.researchgate.net/profile/Yang-Wang-241/publication/361865594_Learning_for_mismatch_removal_via_graph_attention_networks/links/62ce43a06151ad090b9794dd/Learning-for-mismatch-removal-via-graph-attention-networks.pdf)[[code]](https://github.com/StaRainJ/Code-of-GANet)
- [ClusterGNN] Clustergnn: Cluster-based coarse-to-fine graph neural network for efficient feature matching,CVPR'2022,[[pdf]](https://arxiv.org/abs/2204.11700)
- [RoMa] RoMa: Robust Dense Feature Matching, CVPR'2024 [[pdf]](https://arxiv.org/abs/2305.15404) [[code]](https://github.com/Parskatt/RoMa)
















