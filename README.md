


# Awesome-Mismatch-Removal

Methods for eliminating mismatches can be broadly classified into manual methods and deep learning-based methods. Manual methods can be further classified into resampling-based methods, non-parametric model-based methods, and relaxed geometric constraint-based methods.

- [Awesome-Mismatch-Removal](#awesome-mismatch-removal)
  - [Manual Methods](#manual-methods)
    - [Resampling-Based Methods](#resampling-based-methods)
    - [Non-parametric Model-Based Methods](#non-parametric-model-based-methods)
    - [Relaxed Methods](#relaxed-methods)
  - [Learning-based Methods](#learning-based-methods)

***
## Manual Methods
### Resampling-Based Methods
- [RANSAC] Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography, 1981 [[pdf]](https://apps.dtic.mil/dtic/tr/fulltext/u2/a460585.pdf) [[wiki]](https://en.wikipedia.org/wiki/Random_sample_consensus)
- [MLESAC] MLESAC: A new robust estimator with application to estimating image geometry, CVIU'2000 [[pdf]](http://www.academia.edu/download/3436793/torr_mlesac.pdf) [[code_pcl]](https://github.com/PointCloudLibrary/pcl/tree/master/sample_consensus/include/pcl/sample_consensus)
- [NAPSAC] Napsac: High noise, high dimensional robust estimation-it’s in the bag，BMVC'2002[[pdf]](https://d1wqtxts1xzle7.cloudfront.net/51485218/NAPSAC_High_Noise_High_Dimensional_Robus20170123-23393-jxxcl1-libre.pdf?1485199178=&response-content-disposition=inline%3B+filename%3DNapsac_High_Noise_High_Dimensional_Robus.pdf&Expires=1711038934&Signature=Sk4FynbP~i6Ldh3fCffGfi2G-0MayDfLfkGaavAFkFyLhNRaM5VSccpsnqz3yXUos7Phy8cwLwMAjCZo3ye-yfCdEZVS3lQklsK1hdWyzwTgI9YUXHaI3xzYHNrQlIpa-4tSWIem9BP4oBCC7DDKoa5LOKuFXlRso98JBssYx7rz8SI5Ash2zYarn77~llu4YMKVb-dCnPZ6pJaD9wpMqt55JgsNd4Qs49iBVjR5XesQ8eD-V8Ol~Uag~UDac8aHx7TGQzkl7Nn8yHhzeYPwLSCuFN8-VupY94mZQPfkVl9TMhycSiBga45zAovHjGWgP9wRfLUhdbbkRXVeyLmWAg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
- [LO-RANSAC] Locally Optimized RANSAC,2003,[[pdf]](https://cmp.felk.cvut.cz/~chum/papers/chum-DAGM03.pdf)
- [PROSAC] Matching with PROSAC-progressive sample consensus, CVPR'2005 [[pdf]](https://dspace.cvut.cz/bitstream/handle/10467/9496/2005-Matching-with-PROSAC-progressive-sample-consensus.pdf?sequence=1) [[code_pcl]](https://github.com/PointCloudLibrary/pcl/tree/master/sample_consensus/include/pcl/sample_consensus)
- [GroupSAC] GroupSAC: Efficient consensus in the presence of groupings,ICCV'2009 [[pdf]](https://ieeexplore.ieee.org/document/5459241)
- [USAC] A universal framework for random sample consensus.TPAMI'2012,[[pdf]](https://ieeexplore.ieee.org/document/6365642)
- [DefRANSAC] In defence of RANSAC for outlier rejection in deformable registration, ECCV'2012 [[pdf]](https://media.adelaide.edu.au/acvt/Publications/2012/2012-In%20Defence%20of%20RANSAC%20for%20Outlier%20Rejection%20in%20Deformable%20Registration.pdf) [[code]](https://cs.adelaide.edu.au/~tjchin/lib/exe/fetch.php?media=code:eccv12code.zip)
- [EVSAC] EVSAC: Accelerating Hypotheses Generation by Modeling Matching Scores with Extreme Value Theory, ICCV'2013 [[pdf]](https://ieeexplore.ieee.org/document/6751418)
- [WxBS] WxBS: Wide Baseline Stereo Generalizations, BMVC'2015 [[pdf]](https://arxiv.org/pdf/1504.06603) [[project]](http://cmp.felk.cvut.cz/wbs/)
- [RepMatch] RepMatch: Robust Feature Matching and Pose for Reconstructing Modern Cities, ECCV'2016 [[pdf]](http://www.kind-of-works.com/papers/eccv_2016_repmatch.pdf) [[project]](http://www.kind-of-works.com/RepMatch.html) [[code]](http://www.kind-of-works.com/code/repmatch_code_bf_small.zip)
- [GC-RANSAC] Graph-Cut RANSAC, CVPR'2018 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Barath_Graph-Cut_RANSAC_CVPR_2018_paper.pdf) [[code]](https://github.com/danini/graph-cut-ransac)
- [MAGSAC] MAGSAC:marginalizing Sample Consensus, CVPR'2019 [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Barath_MAGSAC_3月ginalizing_Sample_Consensus_CVPR_2019_paper.pdf) [[code]](https://github.com/danini/magsac)
- [P-NAPSAC] Progressive NAPSAC: sampling from gradually growing neighborhoods,2019[[pdf]](https://arxiv.org/abs/1906.02295)
- [MAGSAC++] MAGSAC++, a fast, reliable and accurate robust estimator,CVPR'2020[[pdf]](https://arxiv.org/abs/1912.05909)
- [MAGSAC++] Marginalizing Sample Consensus, TPAMI'2022 [[pdf]](https://ieeexplore.ieee.org/document/9511155) [[code]](https://github.com/danini/magsac)
  
### Non-parametric Model-Based Methods
- Fast non-rigid surface detection, registration and realistic augmentation.,IJCV'2007[[pdf]](https://vincentlepetit.github.io/files/papers/comp_pilet_ijcv07.pdf)
- Direct estimation of nonrigid registrations with image-based self-occlusion reasoning.,ICCV'2007,[[pdf]](https://ieeexplore.ieee.org/document/4408989)
- [ICF/SVR] Rejecting mismatches by correspondence function, IJCV'2010 [[pdf]](http://www.nlpr.ia.ac.cn/2010papers/kz/gk24.pdf)
- [GS] Common visual pattern discovery via spatially coherent correspondences, CVPR'2010 [[pdf]](http://www.jdl.ac.cn/project/faceId/paperreading/Paper/Common%20Visual%20Pattern%20Discovery%20via%20Spatially%20Coherent%20Correspondences.pdf) [[code]](https://sites.google.com/site/lhrbss/home/papers/SimplifiedCode.zip?attredirects=0)
- [KC-CE] A novel kernel correlation model with the correspondence estimation, JMIV'2011 [[pdf]](https://www.researchgate.net/profile/P_Chen2/publication/225191068_A_Novel_Kernel_Correlation_Model_with_the_Correspondence_Estimation/links/02e7e5232cd89055ab000000/A-Novel-Kernel-Correlation-Model-with-the-Correspondence-Estimation.pdf) [[code]](http://web.nchu.edu.tw/~pengwen/WWW/Code.html)
- [VFC] A robust method for vector field learning with application to mismatch removing, CVPR'2011 [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.721.5913&rep=rep1&type=pdf) [[code]](https://github.com/jiayi-ma?tab=repositories)
- Regularized vector field learning with sparse approximation for mismatch removal,PR'2013[[pdf]](https://pages.ucsd.edu/~ztu/publication/pr13_robustmatching.pdf)[[code]](https://github.com/jiayi-ma/VFC)
- [CM] Robust Non-parametric Data Fitting for Correspondence Modeling, ICCV'2013 [[pdf]](https://mmcheng.net/mftp/Papers/DataFittingICCV13.pdf) [[code]](https://sites.google.com/site/laoszefei81/home/code-1/code-curve-fitting)
- [RPM-VFC] Robust Point Matching via Vector Field Consensus, TIP'2014 [[pdf]](http://or.nsfc.gov.cn/bitstream/00001903-5/99530/1/1000009269450.pdf) [[code]](https://github.com/jiayi-ma?tab=repositories)
- [AGMM] A symmetrical Gauss Mixture Models for Point Sets Matching, CVPR'2014 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2014/papers/Tao_Asymmetrical_Gauss_Mixture_2014_CVPR_paper.pdf)
- Robust l2e estimation of transformation for non-rigid registration,TSP'2015,[[pdf]](https://www.researchgate.net/profile/Jiayi_Ma/publication/273176803_Robust_L2E_Estimation_of_Transformation_for_Non-Rigid_Registration/links/552142f10cf2f9c130512304.pdf) [[code]](https://github.com/jiayi-ma?tab=repositories)
- [LLT] Robust feature matching for remote sensing image registration via locally linear transforming, TGRS'2015 [[pdf]](https://yuan-gao.net/pdf/TGRS2015.pdf) [[code]](https://github.com/jiayi-ma?tab=repositories)
- Feature guided Gaussian mixture model with semi-supervised em and local geometric constraint for retinal image registration,IS'2017[[pdf]](https://sci-hub.ru/10.1016/j.ins.2017.07.010)
- [SIM] The shape interaction matrix-based affine invariant mismatch removal for partial-duplicate image search, TIP'2017 [[pdf]](http://www.cis.pku.edu.cn/faculty/vision/zlin/Publications/2017-TIP-SIM.pdf) [[code]](https://github.com/lylinyang/demo_SIM)
- [RPM-MR] Nonrigid Point Set Registration with Robust Transformation Learning under Manifold Regularization, TNNLS'2019 [[pdf]](https://pdfs.semanticscholar.org/e8c9/75165ffc5af6cad6961b25f29ea112ae50dd.pdf) [[code]](https://github.com/jiayi-ma?tab=repositories)
- [PFFM] Progressive Filtering for Feature Matching,ICASSP'2019[[pdf]](https://sci-hub.ru/10.1109/icassp.2019.8682372)
- [GLOF] Robust Feature Matching Using Guided Local Outlier Factor, PR'2021 [[pdf](https://www.sciencedirect.com/science/article/pii/S0031320321001734)] [[code](https://github.com/gwang-cv/GLOF)]


### Relaxed Methods
- A spectral technique for correspondence problems using pairwise constraints,ICCV'2005,[[pdf]](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/leordeanu-iccv-05.pdf)
- Common visual pattern discovery via spatially coherent correspondences,CVPR'2010,[[pdf]](https://sci-hub.ru/10.1109/cvpr.2010.5539780)
- Feature matching with bounded distortion,TOG'2014,[[pdf]](https://sci-hub.ru/10.1145/2602142)
- [BF] Bilateral Functions for Global Motion Modeling, ECCV'2014 [[pdf]](http://mftp.mmcheng.net/Papers/CoherentModelingS.pdf) [[project]](https://mmcheng.net/bfun/) [[code]](http://mftp.mmcheng.net/Data/eccv_2014_release.zip)
- [TC] Epipolar geometry estimation for wide baseline stereo by Clustering Pairing Consensus, PRL'2014 [[pdf]](http://or.nsfc.gov.cn/bitstream/00001903-5/96605/1/1000007190373.pdf)
- [LMI] Consensus Maximization with Linear Matrix Inequality Constraints, CVPR'2017 [[pdf]](https://www.cvg.ethz.ch/research/conmax/paper/PSpeciale2017CVPR.pdf) [[project]](https://www.cvg.ethz.ch/research/conmax/) [[code]](https://www.cvg.ethz.ch/research/conmax/paper/PSpeciale2017CVPR_code_sample.tar.gz)
- [CODE] Code: Coherence based decision boundaries for feature correspondence, TPAMI'2018 [[pdf]](https://ora.ox.ac.uk/objects/uuid:0e5a62ab-fb69-472f-a1e1-49d49595db62/download_file?safe_filename=matching.pdf&file_format=application%2Fpdf&type_of_work=Journal+article) [[project]](http://www.kind-of-works.com/CODE_matching.html)
- [GLPM] Guided Locality Preserving Feature Matching for Remote Sensing Image Registration, TGRS'2018 [[pdf]](https://ieeexplore.ieee.org/abstract/document/8340808/)
- [LPM] Locality preserving matching, IJCV'2019 [[pdf]](https://link.springer.com/article/10.1007/s11263-018-1117-z) [[code]](https://github.com/jiayi-ma?tab=repositories)
- [GMS] GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence, CVPR'2017 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bian_GMS_Grid-based_Motion_CVPR_2017_paper.pdf) [[code]](https://github.com/JiawangBian/GMS-Feature-Matcher)
- [GOPAC] Globally-Optimal Inlier Set Maximisation for Simultaneous Camera Pose and Feature Correspondence, ICCV'2017 [[pdf]](https://drive.google.com/open?id=0BwzhzqTiWNEWTzE3ZW1lNnhBTUE) TPAMI'2018 [[pdf]](https://drive.google.com/open?id=1FV_SFoxVvsspK3uh9lRYsJPUSX0kuI_L) [[code]](https://drive.google.com/open?id=1H7gOQz7CAXSat56OPTgV2lOLUSI4D_vG)
- [SRC] Consensus Maximization for Semantic Region Correspondences, CVPR'2018 [[pdf]](https://www.cvg.ethz.ch/research/secon/paper/PSpeciale2018CVPR.pdf) [[code]](https://www.cvg.ethz.ch/research/secon/paper/PSpeciale2018CVPR_code_sample.zip)
- [RFM-SCAN] Robust feature matching using spatial clustering with heavy outliers,TIP'2020,[[pdf]](https://starainj.github.io/Files/TIP2020-RFM-SACN.pdf)[[code]](https://github.com/StaRainJ/RFM-SCAN)
- Progressive feature matching: Incremental graph construction and optimization,TIP'2020,[[pdf]](https://sci-hub.ru/10.1109/tip.2020.2996092)[[code]](https://sites.google.com/view/sehyung/home/projects/progressive-feature-matching)


## Learning-based Methods
- [DSAC] DSAC: differentiable RANSAC for camera localization, CVPR'2017 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Brachmann_DSAC_-_Differentiable_CVPR_2017_paper.pdf) [[code]](https://github.com/cvlab-dresden/DSAC)
- [LFGC] Learning to Find Good Correspondences, CVPR'2018 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1453.pdf) [[code]](https://github.com/vcg-uvic/learned-correspondence-release)
- [DFE] Deep fundamental matrix estimation, ECCV'2018[[pdf]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Rene_Ranftl_Deep_Fundamental_Matrix_ECCV_2018_paper.pdf) [[code]](https://github.com/isl-org/DFE)
- [N3Net] Neural Nearest Neighbors Networks, NeurIPS'2018 [[code]](https://github.com/visinf/n3net/)
- [KCNet] Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling, CVPR'2018 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Mining_Point_Cloud_CVPR_2018_paper.pdf) [[code]](http://www.merl.com/research/license#KCNet)
- [NM-Net] NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences, arXiv'2019 [[pdf]](https://arxiv.org/pdf/1904.00320)
- [OANet] Learning Two-View Correspondences and Geometry Using Order-Aware Network ICCV'2019 [[code]](https://github.com/zjhthu/OANet)
- [LMR] LMR: Learning A Two-class Classifier for Mismatch Removal, TIP'2019 [[pdf]](https://starainj.github.io/Files/TIP2019-LMR.pdf) [[code]](https://github.com/StaRainJ/LMR)
- [NG-RANSAC] Neural-Guided RANSAC: Learning Where to Sample Model Hypotheses, ICCV'2019 [[pdf](https://arxiv.org/pdf/1905.04132.pdf)] [[code](https://github.com/vislearn/ngransac)] [[project](https://hci.iwr.uni-heidelberg.de/vislearn/research/neural-guided-ransac/)]
- [ULCM] Unsupervised Learning of Consensus Maximization for 3D Vision Problems, CVPR'2019 [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Probst_Unsupervised_Learning_of_Consensus_Maximization_for_3D_Vision_Problems_CVPR_2019_paper.pdf)
- [GLHA] Cascade Network with Guided Loss and Hybrid Attention for Finding Good Correspondences,AAAI'2021[[pdf]](https://arxiv.org/abs/2102.00411)[[code]](https://github.com/wenbingtao/GLHA)
- [ACNe] ACNe: Attentive context normalization for robust permutation-equivariant learning, CVPR'2020[[code]](https://github.com/vcg-uvic/acne)
- [SuperGlue] SuperGlue: Learning Feature Matching with Graph Neural Networks, CVPR'2020 [[code]](https://github.com/magicleap/SuperGluePretrainedNetwork)
- [LSV-ANet] LSV-ANet: Deep Learning on Local Structure Visualization for Feature Matching,TGRS'2021[[pdf]](https://ieeexplore.ieee.org/document/9377555)
- [LMCNet] Learnable motion coherence for correspondence pruning,CVPR'2021 [[pdf]](https://arxiv.org/abs/2011.14563)[[code]](https://github.com/liuyuan-pal/LMCNet)
- [CLNet] Progressive correspondence pruning by consensus learning,ICCV'2021 [[pdf]](https://arxiv.org/abs/2101.00591)
- [TNet] T-Net: Effective permutation-equivariant network for two-view correspondence learning,ICCV'2021[[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhong_T-Net_Effective_Permutation-Equivariant_Network_for_Two-View_Correspondence_Learning_ICCV_2021_paper.pdf)
- Efficient Deterministic Search with Robust Loss Functions for Geometric Model Fitting, TPAMI'2021 [[code]](https://github.com/AoxiangFan/EifficientDeterministicSearch)
- [LoFTR] Detector-free local feature matching with transformers, CVPR'2021 [[code]](https://github.com/zju3dv/LoFTR)
- [CAT] Correspondence Attention Transformer: A Context-sensitive Network for Two-view Correspondence Learning, TMM'2022[[pdf]](https://www.researchgate.net/profile/Yang-Wang-241/publication/359451839_Correspondence_Attention_Transformer_A_Context-sensitive_Network_for_Two-view_Correspondence_Learning/links/62ce44b3b261d22751eb64d4/Correspondence-Attention-Transformer-A-Context-Sensitive-Network-for-Two-View-Correspondence-Learning.pdf) [[code]](https://github.com/jiayi-ma/CorresAttnTransformer)
- [MSANet]MSA-net: Establishing reliable correspondences by multiscale attention network,TIP'2022[[pdf]](https://guobaoxiao.github.io/papers/TIP2022b(1).pdf)
- [CSR-net]CSR-net: Learning adaptive context structure representation for robust feature correspondence,TIP'2022[[pdf]](https://ieeexplore.ieee.org/document/9758641)
- [GANet] Learning for mismatch removal via graph attention networks,ISPRS J PHOTOGRAMM'2022,[[pdf]](https://www.researchgate.net/profile/Yang-Wang-241/publication/361865594_Learning_for_mismatch_removal_via_graph_attention_networks/links/62ce43a06151ad090b9794dd/Learning-for-mismatch-removal-via-graph-attention-networks.pdf)[[code]](https://github.com/StaRainJ/Code-of-GANet)
- [NeFSAC] NeFSAC: Neurally filtered minimal samples,ECCV'2022[[pdf]](https://arxiv.org/abs/2207.07872)[[code]](https://github.com/cavalli1234/NeFSAC)
- [MS2DGNet] MS2DGNet: Progressive correspondence learning via multiple sparse semantics dynamic graph,CVPR'2022[[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Dai_MS2DG-Net_Progressive_Correspondence_Learning_via_Multiple_Sparse_Semantics_Dynamic_Graph_CVPR_2022_paper.pdf)[[code]](https://github.com/changcaiyang/MS2DG-Net)
- [TransMVSNet] TransMVSNet: Global Context-aware Multi-view Stereo Network with Transformers, CVPR'2022 [[code]](https://github.com/MegviiRobot/TransMVSNet)
- [ConvMatch] ConvMatch: Rethinking Network Design for Two-View Correspondence Learning, AAAI'2023 [[code]](https://github.com/SuhZhang/ConvMatch)
- [ANA-Net]Learning second-order attentive context for efficient correspondence pruning,AAAI'2023[[pdf]](https://arxiv.org/abs/2303.15761)
- [TopicFM] TopicFM: Robust and Interpretable Topic-Assisted Feature Matching,AAAI'2023[[pdf]](https://arxiv.org/abs/2207.00328)[[code]](https://github.com/TruongKhang/TopicFM)
- [ParaFormer] ParaFormer: Parallel Attention Transformer for Efficient Feature Matching,AAAI'2023[[pdf]](https://arxiv.org/abs/2303.00941)
- [LightGlue] LightGlue: Local Feature Matching at Light Speed, arxiv'2023 [[pdf]](https://arxiv.org/pdf/2306.13643.pdf) [[code]](https://github.com/cvg/LightGlue)
- [DKM] DKM: Dense Kernelized Feature Matching for Geometry Estimation, CVPR'2023 [[code]](https://github.com/Parskatt/DKM)
- [ASTR] ASTR: Adaptive Spot-Guided Transformer for Consistent Local Feature Matching, CVPR'2023 [[code]](https://astr2023.github.io/)
- [IMP] IMP: Iterative Matching and Pose Estimation with Adaptive Pooling, CVPR'2023 [[code]](https://github.com/feixue94/imp-release)
- [PATS] PATS: Patch Area Transportation with Subdivision for Local Feature Matching, CVPR'2023 [[code]](https://zju3dv.github.io/pats/)
- [NCMNet] Progressive Neighbor Consistency Mining for Correspondence Pruning, CVPR'2023 [[code]](https://github.com/xinliu29/NCMNet)
- [AdaMatcher] Adaptive Assignment for Geometry Aware Local Feature Matching, CVPR'2023 [[code]](https://github.com/AbyssGaze/AdaMatcher)
- [SEM] Structured Epipolar Matcher for Local Feature Matching, CVPRW'2023 [[code]](https://sem2023.github.io/)
- [FSNet] Two-View Geometry Scoring Without Correspondences, CVPR'2023 [[code]](https://github.com/nianticlabs/scoring-without-correspondences)
- [PGFNet]]PGFNet: Preference-guided filtering network for two-view correspondence learning,TIP'2023[[pdf]](https://ieeexplore.ieee.org/document/10041834)[[code]](https://github.com/guobaoxiao/PGFNet)
- [PDC-Net+] PDC-Net+: Enhanced Probabilistic Dense Correspondence Network,TPAMI'2023[[pdf]](https://arxiv.org/pdf/2109.13912.pdf)[[code]](https://github.com/PruneTruong/DenseMatching)
- [JRANet] JRA-net: Joint representation attention network for correspondence learning,PR'2023[[pdf]](https://www.sciencedirect.com/science/article/pii/S0031320322006598)
- [GCANet] Learning for feature matching via graph context attention,TGARS'2023[[pdf]](https://ieeexplore.ieee.org/document/10075633)
- [RLSAC] RLSAC: Reinforcement Learning enhanced Sample Consensus for End-to-End Robust Estimation,ICCV'2023[[pdf]](https://arxiv.org/pdf/2308.05318.pdf)[[code]](https://github.com/IRMVLab/RLSAC)
- [SiLK] SiLK: Simple Learned Keypoints,ICCV'2023[[pdf]](https://arxiv.org/abs/2304.06194)[[code]](https://github.com/facebookresearch/silk)
- [CasMTR] Improving Transformer-based Image Matching by Cascaded Capturing Spatially Informative Keypoints,ICCV'2023[[pdf]](https://arxiv.org/abs/2303.02885)[[code]](https://github.com/ewrfcas/CasMTR)
- [GlueStick] GlueStick: Robust Image Matching by Sticking Points and Lines Together, ICCV'2023 [[pdf]](https://arxiv.org/pdf/2304.02008.pdf) [[code]](https://github.com/cvg/GlueStick)
- [OAMatcher] OAMatcher: An Overlapping Areas-based Network for Accurate Local Feature Matching,arxiv'2023[[pdf]](https://arxiv.org/abs/2302.05846)[[code]](https://github.com/DK-HU/OAMatcher)
- [SGAM] Searching from Area to Point: A Hierarchical Framework for Semantic-Geometric Combined Feature Matching,arxiv'2023[[pdf]](https://arxiv.org/abs/2305.00194)
- Sparse-to-Local-Dense Matching for Geometry-Guided Correspondence Estimation,TIP'2023[[pdf]](https://ieeexplore.ieee.org/document/10159656)
- [HTMatch] HTMatch: An efficient hybrid transformer based graph neural network for local feature matching,SP'2023[[pdf]](https://www.sciencedirect.com/science/article/pii/S016516842200398X)
- [SGA-Net] SGA-Net: A Sparse Graph Attention Network for Two-View Correspondence Learning,TCSVT'2023[[pdf]](https://ieeexplore.ieee.org/abstract/document/10124002)
- [DeepMatcher] DeepMatcher: A deep transformer-based network for robust and accurate local feature matching,ESWA'2023[[pdf]](https://arxiv.org/abs/2301.02993)
- Topological RANSAC for instance verification and retrieval without fine-tuning,NeurIPS'2023[[pdf]](https://arxiv.org/abs/2310.06486)[[code]](https://github.com/anguoyuan/Topological-RANSAC)
- Emergent Correspondence from Image Diffusion,NeurIPS'2023[[pdf]](https://arxiv.org/abs/2306.03881)[[code]](https://github.com/Tsingularity/dift)
- [RoMa] RoMa: Robust Dense Feature Matching, CVPR'2024 [[pdf]](https://arxiv.org/abs/2305.15404) [[code]](https://github.com/Parskatt/RoMa)
- [MESA] MESA: Matching Everything by Segmenting Anything, CVPR'2024 [[pdf]](https://arxiv.org/abs/2401.16741)
- [Efficient LoFTR] Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed, CVPR'2024 [[pdf]](https://zju3dv.github.io/efficientloftr/files/EfficientLoFTR.pdf) [[code]](https://github.com/zju3dv/efficientloftr)
- [VSFormer] VSFormer: Visual-Spatial Fusion Transformer for Correspondence Pruning,AAAI'2024[[pdf]](https://arxiv.org/pdf/2312.08774.pdf)[[code]](https://github.com/sugar-fly/VSFormer)
- [MatchDet] MatchDet: A Collaborative Framework for Image Matching and Object Detection,AAAI'2024[[pdf]](https://arxiv.org/abs/2312.10983)
- [ResMatch] ResMatch: Residual Attention Learning for Local Feature Matching,AAAI'2024[[pdf]](https://arxiv.org/abs/2307.05180)[[code]](https://github.com/ACuOoOoO/ResMatch)
- [GCT-Net] Graph Context Transformation Learning for Progressive Correspondence Pruning,AAAI'2024[[pdf]](https://arxiv.org/abs/2312.15971)
- [BCLNet] BCLNet: Bilateral Consensus Learning for Two-View Correspondence Pruning,AAAI'2024[[pdf]](https://arxiv.org/abs/2401.03459)
- [MSGA-Net] MSGA-Net: Progressive Feature Matching via Multi-layer Sparse Graph Attention,TCSVT'2024[[pdf]](https://ieeexplore.ieee.org/document/10439184)
- [GIM] GIM: Learning Generalizable Image Matcher From Internet Videos,ICLR'2024[[pdf]](https://arxiv.org/abs/2402.11095)[[code]](https://github.com/xuelunshen/gim)
- Diffusion Model for Dense Matching，ICLR'2024[[pdf]](https://arxiv.org/pdf/2305.19094.pdf)[[code]](https://github.com/KU-CVLAB/DiffMatch)
- 












