# An Updating Survey for Bayesian Deep Learning (BDL)

This is an updating survey for Bayesian Deep Learning (BDL), an constantly updated and extended version for the manuscript, '[A Survey on Bayesian Deep Learning](http://wanghao.in/paper/CSUR20_BDL.pdf)', published in [**ACM Computing Surveys**](https://dl.acm.org/doi/10.1145/3409383) 2020.<br>

Bayesian deep learning is a powerful framework for designing models across a wide range of applications. See our [**Nature Medicine** paper](https://www.nature.com/articles/s41591-021-01273-1.pdf) for a possible application on healthcare. 

## Contents

* [Survey](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#survey)
* [BDL and Recommender Systems](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-recommender-systems)
* [BDL and Domain Adaptation (and Domain Generalization, Meta Learning, etc.)](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-domain-adaptation-and-domain-generalization-meta-learning-etc)
* [BDL and Healthcare](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-healthcare)
* [BDL and Natural Language Processing (NLP)](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-nlp)
* [BDL and Computer Vision (CV)](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-computer-vision)
* [BDL and Control/Planning](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-controlplanning)
* [BDL and Graphs (Link Prediction, Graph Neural Networks, Knowledge Graphs, etc.)](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-graphs-link-prediction-graph-neural-networks-knowledge-graphs-etc)
* [BDL and Topic Modeling](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-topic-modeling)
* [BDL and Speech Recognition/Synthesis](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-speech-recognitionsynthesis)
* [BDL and Forecasting (Time Series Analysis)](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-forecasting-time-series-analysis)
* [BDL and Distributed/Federated Learning](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-distributedfederated-learning)
* [BDL and Continual/Life-Long Learning](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-continuallife-long-learning)
* [BDL and AI4Science](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-ai4science)
* [BDL as a Framework (Miscellaneous)](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-as-a-framework-miscellaneous)
* [Bayesian/Probabilistic Neural Networks as Building Blocks of BDL](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bayesianprobabilistic-neural-networks-as-building-blocks-of-bdl)


## Survey

A Survey on Bayesian Deep Learning<br>
by Wang et al., ACM Computing Surveys (CSUR) 2020<br>
[[PDF]](http://wanghao.in/paper/CSUR20_BDL.pdf) [[Blog]](http://wanghao.in/BDL.html) [[BDL Framework in 2016]](http://wanghao.in/paper/TKDE16_BDL.pdf)

<p align="center">
<img src="./BDL_Table.png" alt="" data-canonical-src="./BDL_Table.png" width="930" height="580"/>
</p>

## BDL and Recommender Systems

Collaborative Deep Learning for Recommender Systems<br>
by Wang et al., KDD 2015<br>
[[PDF]](http://wanghao.in/paper/KDD15_CDL.pdf) [[Project Page]](http://wanghao.in/CDL.htm) [[2014 Arxiv Version]](https://arxiv.org/abs/1409.2944) [[Code]](https://github.com/js05212/CDL) [[MXNet Code]](https://github.com/js05212/MXNet-for-CDL) [[TensorFlow Code]](https://github.com/js05212/CollaborativeDeepLearning-TensorFlow) [[Dataset A]](https://github.com/js05212/citeulike-a) [[Dataset B]](https://github.com/js05212/citeulike-t) [[Jupyter Notebook]](https://github.com/js05212/MXNet-for-CDL/blob/master/collaborative-dl.ipynb) [[Slides]](http://wanghao.in/slides/CDL_slides.pdf) [[Slides (Long)]](http://wanghao.in/slides/CDL_slides_long.pdf)

Collaborative Recurrent Autoencoder: Recommend while Learning to Fill in the Blanks<br>
by Wang et al., NIPS 2016<br>
[[PDF]](https://arxiv.org/abs/1611.00454)

Collaborative Knowledge Base Embedding for Recommender Systems<br>
by Zhang et al., KDD 2016<br>
[[PDF]](https://dl.acm.org/citation.cfm?id=2939673)

Collaborative Deep Ranking: A Hybrid Pair-Wise Recommendation Algorithm with Implicit Feedback<br>
by Ying et al., PAKDD 2016<br>
[[PDF]](https://link.springer.com/chapter/10.1007/978-3-319-31750-2_44)

Collaborative Variational Autoencoder for Recommender Systems<br>
by Li et al., KDD 2017<br>
[[PDF]](https://www.kdd.org/kdd2017/papers/view/collaborative-variational-autoencoder-for-recommender-systems)

Variational Autoencoders for Collaborative Filtering<br>
by Liang et al., WWW 2018<br>
[[PDF]](https://arxiv.org/abs/1802.05814)

Probabilistic Metric Learning with Adaptive Margin for Top-K Recommendation<br>
by Ma et al., KDD 2020<br>
[[PDF]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403147)

## BDL and Domain Adaptation (and Domain Generalization, Meta Learning, etc.)
Probabilistic Model-Agnostic Meta-Learning<br>
by Finn et al., NIPS 2018<br>
[[PDF]](https://papers.nips.cc/paper/2018/file/8e2c381d4dd04f1c55093f22c59c3a08-Paper.pdf)

Bayesian Model-Agnostic Meta-Learning<br>
by Yoon et al., NIPS 2018<br>
[[PDF]](https://arxiv.org/pdf/1806.03836.pdf)

Recasting Gradient-Based Meta-Learning as Hierarchical Bayes<br>
by Grant et al., ICLR 2018<br>
[[PDF]](https://arxiv.org/abs/1801.08930)

Reconciling Meta-Learning and Continual Learning with Online Mixtures of Tasks<br>
by Jerfal et al., NIPS 2019<br>
[[PDF]](https://arxiv.org/abs/1812.06080)

Meta-Learning Probabilistic Inference For Prediction<br>
by Gordon et al., ICLR 2019<br>
[[PDF]](https://arxiv.org/abs/1805.09921)

Learning to Learn with Variational Information Bottleneck for Domain Generalization<br>
by Du et al., ECCV 2020<br>
[[PDF]](https://arxiv.org/pdf/2007.07645.pdf)

Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels<br>
by Patacchiola et al., NIPS 2020<br>
[[PDF]](https://arxiv.org/pdf/1910.05199.pdf)

Continuously Indexed Domain Adaptation<br>
by Wang et al., ICML 2020<br>
[[PDF]](http://wanghao.in/paper/ICML20_CIDA.pdf) 

A Bit More Bayesian: Domain-Invariant Learning with Uncertainty<br>
by Xiao et al., ICML 2021<br>
[[PDF]](https://arxiv.org/pdf/2105.04030.pdf)

Domain-Indexing Variational Bayes: Interpretable Domain Index for Domain Adaptation<br>
by Xu et al., ICLR 2023<br>
[[PDF]](http://wanghao.in/paper/ICLR23_VDI.pdf)




## BDL and Healthcare

Electronic Health Record Analysis via Deep Poisson Factor Models<br>
by Henao et al., JMLR 2016<br>
[[PDF]](http://www.jmlr.org/papers/volume17/15-429/15-429.pdf)

Structured Inference Networks for Nonlinear State Space Models<br>
by Krishnan et al., AAAI 2017<br>
[[PDF]](https://arxiv.org/pdf/1609.09869.pdf)

Causal Effect Inference with Deep Latent-Variable Models<br>
by Louizos et al., NIPS 2017<br>
[[PDF]](https://arxiv.org/pdf/1705.08821.pdf)

Black Box FDR<br>
by Tansey et al., ICML 2018<br>
[[PDF]](https://arxiv.org/abs/1806.03143)

Bidirectional Inference Networks: A Class of Deep Bayesian Networks for Health Profiling<br>
by Wang et al., AAAI 2019<br>
[[PDF]](https://arxiv.org/pdf/1902.02037)

Sampling-free Uncertainty Estimation in Gated Recurrent Units with Applications to Normative Modeling in Neuroimaging<br>
by Hwang et al., UAI 2019<br>
[[PDF]](http://auai.org/uai2019/proceedings/papers/296.pdf)

Neural Jump Stochastic Differential Equations<br>
by Jia et al., NIPS 2019<br>
[[PDF]](https://arxiv.org/pdf/1905.10403.pdf)

Towards Interpretable Clinical Diagnosis with Bayesian Network Ensembles Stacked on Entity-Aware CNNs<br>
by Chen et al., ACL 2020<br>
[[PDF]](https://www.aclweb.org/anthology/2020.acl-main.286.pdf)

Continuously Indexed Domain Adaptation<br>
by Wang et al., ICML 2020<br>
[[PDF]](http://wanghao.in/paper/ICML20_CIDA.pdf) [Cross Referenced in [BDL and Domain Adaptation](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-domain-adaptation-and-domain-generalization-meta-learning-etc)]

Assessment of medication self-administration using artificial intelligence<br>
by Zhao et al., Nature Medicine 2021<br>
[[PDF]](https://www.nature.com/articles/s41591-021-01273-1.pdf)

Neural Pharmacodynamic State Space Modeling<br>
by Hussain et al., ICML 2021<br>
[[PDF]](https://arxiv.org/pdf/2102.11218.pdf)

Self-Interpretable Time Series Prediction with Counterfactual Explanations<br>
by Yan et al., ICML 2023<br>
[[PDF]](http://wanghao.in/paper/ICML23_CounTS.pdf) [Cross Referenced in [BDL and Forecasting (Time Series Analysis)](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-forecasting-time-series-analysis)]

## BDL and NLP

Sequence to Better Sequence: Continuous Revision of Combinatorial Structures<br>
by Mueller et al., ICML 2017<br>
[[PDF]](http://proceedings.mlr.press/v70/mueller17a.html)

QuaSE: Sequence Editing under Quantifiable Guidance<br>
by Liao et al., EMNLP 2018<br>
[[PDF]](https://arxiv.org/pdf/1804.07007.pdf)

Dispersed Exponential Family Mixture VAEs for Interpretable Text Generation<br>
by Shi et al., ICML 2020<br>
[[PDF]](https://proceedings.icml.cc/static/paper_files/icml/2020/3242-Paper.pdf)

Towards Interpretable Clinical Diagnosis with Bayesian Network Ensembles Stacked on Entity-Aware CNNs<br>
by Chen et al., ACL 2020<br>
[[PDF]](https://www.aclweb.org/anthology/2020.acl-main.286.pdf) [Cross Referenced in [BDL and Healthcare](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-healthcare)]

What You Say and How You Say it: Joint Modeling of Topics and Discourse in Microblog Conversations<br>
by Zeng et al., ACL 2020<br>
[[PDF]](https://aclanthology.org/Q19-1017.pdf)

Latent Diffusion Energy-Based Model for Interpretable Text Modeling<br>
by Yu et al., ICML 2022<br>
[[PDF]](https://arxiv.org/abs/2206.05895)

Diffusion-LM Improves Controllable Text Generation<br>
by Li et al., NeurIPS 2022<br>
[[PDF]](https://proceedings.neurips.cc/paper_files/paper/2022/file/1be5bc25d50895ee656b8c2d9eb89d6a-Paper-Conference.pdf)

Tractable Control for Autoregressive Language Generation<br>
by Zhang et al., ICML 2023<br>
[[PDF]](https://arxiv.org/pdf/2304.07438.pdf)

Variational Language Concepts for Interpreting Foundation Language Models<br>
by Wang et al., EMNLP 2024<br>
[[PDF]](http://wanghao.in/paper/EMNLP24_VALC.pdf)

Multi-agent Architecture Search via Agentic Supernet<br>
by Zhang et al., ICML 2025<br>
[[PDF]](https://arxiv.org/pdf/2502.04180)

## BDL and Computer Vision
Attend, Infer, Repeat: Fast Scene Understanding with Generative Models<br>
by Eslami et al., NIPS 2016<br>
[[PDF]](https://arxiv.org/abs/1603.08575)

Efficient Inference in Occlusion-aware Generative Models of Images<br>
by Huang et al., ICLR 2016<br>
[[PDF]](https://arxiv.org/abs/1511.06362)

Sequential Attend, Infer, Repeat: Generative Modelling of Moving Objects<br>
by Kosiorek et al., NIPS 2018<br>
[[PDF]](https://arxiv.org/abs/1806.01794)

Gaussian Process Prior Variational Autoencoders<br>
by Casale et al., NIPS 2018<br>
[[PDF]](https://arxiv.org/pdf/1810.11738.pdf)

Spatially Invariant Unsupervised Object Detection with Convolutional Neural Networks<br>
by Crawford et al., AAAI 2019<br>
[[PDF]](https://www.aaai.org/ojs/index.php/AAAI/article/view/4216)

Faster Attend-Infer-Repeat with Tractable Probabilistic Models<br>
by Stelzner et al., ICML 2019<br>
[[PDF]](http://proceedings.mlr.press/v97/stelzner19a.html)

Asynchronous Temporal Fields for Action Recognition<br>
by Sigurdsson et al., CVPR 2017<br>
[[PDF]](https://arxiv.org/pdf/1612.06371.pdf)

Generalizing Eye Tracking with Bayesian Adversarial Learning<br>
by Wang et al., CVPR 2019<br>
[[PDF]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Generalizing_Eye_Tracking_With_Bayesian_Adversarial_Learning_CVPR_2019_paper.pdf)

Sequential Neural Processes<br>
by Singh et al., NIPS 2019<br>
[[PDF]](http://papers.nips.cc/paper/9214-sequential-neural-processes.pdf)

SPACE: Unsupervised Object-Oriented Scene Representation via Spatial Attention and Decomposition<br>
by Lin et al., ICLR 2020<br>
[[PDF]](https://arxiv.org/pdf/2001.02407.pdf)

 Being Bayesian about Categorical Probability<br>
 by Joo et al., ICML 2020<br>
 [[PDF]](https://proceedings.icml.cc/static/paper_files/icml/2020/3560-Paper.pdf)

 NVAE: A Deep Hierarchical Variational Autoencoder<br>
 by Vahdat et al., NIPS 2020<br>
 [[PDF]](https://arxiv.org/abs/2007.03898)
 
 Learning Latent Space Energy-Based Prior Model<br>
 by Pang et al., NIPS 2020<br>
 [[PDF]](https://arxiv.org/pdf/2006.08205.pdf)
 
 Generative Neurosymbolic Machines<br>
 by Jiang et al., NIPS 2020<br>
 [[PDF]](https://arxiv.org/pdf/2010.12152.pdf)
 
 Denoising Diffusion Probabilistic Models<br>
 by Ho et al., NIPS 2020<br>
 [[PDF]](https://arxiv.org/pdf/2006.11239.pdf)
 
A Causal View of Compositional Zero-shot Recognition<br>
by Atzmon et al., NIPS 2020<br>
[[PDF]](https://arxiv.org/pdf/2006.14610.pdf)

Counterfactuals Uncover the Modular Structure of Deep Generative Models<br>
by Besserve et al., ICLR 2020<br>
[[PDF]](https://openreview.net/pdf?id=SJxDDpEKvH)

ROOTS: Object-Centric Representation and Rendering of 3D Scenes<br>
by Chen et al., JMLR 2021<br>
[[PDF]](https://jmlr.csail.mit.edu/papers/volume22/20-1176/20-1176.pdf)
 
Improved Denoising Diffusion Probabilistic Models<br>
by Nichol et al., ICML 2021<br>
[[PDF]](https://arxiv.org/pdf/2102.09672.pdf)
 
Generative Interventions for Causal Learning.<br>
by Mao et al., CVPR 2021<br>
[[PDF]](http://wanghao.in/paper/CVPR21_GenInt.pdf)

Adversarial Attacks are Reversible with Natural Supervision<br>
by Mao et al., ICCV 2021<br>
[[PDF]](http://www.wanghao.in/paper/ICCV21_ReverseAttack.pdf)

Counterfactual Zero-Shot and Open-Set Visual Recognition<br>
by Yue et al., CVPR 2021<br>
[[PDF]](https://arxiv.org/pdf/2103.00887.pdf)

ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models<br>
by Choi et al., ICCV 2021<br>
[[PDF]](https://arxiv.org/pdf/2108.02938.pdf)

Diffusion Models Beat GANs on Image Synthesis<br>
by Dhariwal et al., NIPS 2021<br>
[[PDF]](https://arxiv.org/pdf/2105.05233.pdf)

Diffusion Visual Counterfactual Explanations<br>
by Augustin et al., NIPS 2022<br>
[[PDF]](https://arxiv.org/pdf/2210.11841)

DiffuseVAE: Efficient, Controllable and High-Fidelity Generation from Low-Dimensional Latents<br>
by Pandey et al., TMLR 2022<br>
[[PDF]](https://arxiv.org/pdf/2201.00308)

Diffusion Causal Models for Counterfactual Estimation<br>
by Sanchez et al., CleaR 2022<br>
[[PDF]](https://arxiv.org/abs/2202.10166)

Relational Learning with Variational Bayes<br>
by Liu, ICLR 2022<br>
[[PDF]](https://openreview.net/pdf?id=Az-7gJc6lpr)

High-Resolution Image Synthesis with Latent Diffusion Models<br>
by Rombach et al., CVPR 2022<br>
[[PDF]](https://arxiv.org/abs/2112.10752)

GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models<br>
by Nichol et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/nichol22a.html)

Diffusion Models for Adversarial Purification<br>
by Nie et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/nie22a.html)

A Conditional Point Diffusion-Refinement Paradigm for 3D Point Cloud Completion<br>
by Lyu et al., ICLR 2022<br>
[[PDF]](https://iclr.cc/virtual/2022/poster/7026)

Label-Efficient Semantic Segmentation with Diffusion Models<br>
by Baranchuk et al., ICLR 2022<br>
[[PDF]](https://iclr.cc/virtual/2022/poster/6569)

Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality<br>
by Watson et al., ICLR 2022<br>
[[PDF]](https://openreview.net/pdf?id=VFBjuF8HEp)

Flexible Diffusion Modeling of Long Videos<br>
by Harvey et al., NIPS 2022<br>
[[PDF]](https://arxiv.org/pdf/2205.11495.pdf)

ProtoVAE: A Trustworthy Self-Explainable Prototypical Variational Model<br>
by Gautam et al., NIPS 2022<br>
[[PDF]](https://proceedings.neurips.cc/paper_files/paper/2022/file/722f3f9298a961d2639eadd3f14a2816-Paper-Conference.pdf)

Causal Transportability for Visual Recognition<br>
by Mao et al., CVPR 2022<br>
[[PDF]](http://wanghao.in/paper/CVPR22_CausalTrans.pdf)

Posterior Matching for Arbitrary Conditioning<br>
by Strauss et al., NIPS 2022<br>
[[PDF]](https://openreview.net/pdf?id=EFnI8Qc--jE)

On the Relationship between Variational Inference and Auto-Associative Memory<br>
by Annabi et al., NIPS 2022<br>
[[PDF]](https://openreview.net/pdf?id=uCBx_6Hc7cu)

Robust Perception through Equivariance<br>
by Mao et al., ICML 2023<br>
[[PDF]](http://wanghao.in/paper/ICML23_RobustEquivariance.pdf)

Object-Centric Slot Diffusion<br>
by Jiang et al. NeurIPS 2023<br>
[[PDF]](https://arxiv.org/abs/2303.10834)

PreDiff: Precipitation Nowcasting with Latent Diffusion Models<br>
by Gao et al., NeurIPS 2023<br>
[[PDF]](https://arxiv.org/abs/2307.10422)

Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective<br>
by Dou et al., ICLR 2024<br>
[[PDF]](https://openreview.net/forum?id=tplXNcHZs1)

Directly Denoising Diffusion Models<br>
by Zhang et al., ICML 2024<br>
[[PDF]](https://proceedings.mlr.press/v235/zhang24bl.html)

Causal Representation Learning Made Identifiable by Grouping of Observational Variables<br>
by Morioka et al., ICML 2024<br>
[[PDF]](https://proceedings.mlr.press/v235/morioka24a.html)

Counterfactual Image Editing
by Pan et al., ICML 2024<br>
[[PDF]](https://proceedings.mlr.press/v235/pan24a.html)

Probabilistic Conceptual Explainers: Towards Trustworthy Conceptual Explanations for Vision Foundation Models<br>
by Wang et al., ICML 2024<br>
[[PDF]](http://wanghao.in/paper/ICML24_PACE.pdf)




## BDL and Control/Planning

Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images<br>
by Watter et al., NIPS 2015<br>
[[PDF]](https://arxiv.org/abs/1506.07365)

Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data<br>
by Karl et al., ICLR 2017<br>
[[PDF]](https://arxiv.org/pdf/1605.06432.pdf)

Probabilistic Recurrent State-Space Models<br>
by Doerr et al., ICML 2018<br>
[[PDF]](http://proceedings.mlr.press/v80/doerr18a/doerr18a.pdf)

Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models<br>
by Chua et al., NIPS 2018<br>
[[PDF]](https://proceedings.neurips.cc/paper/2018/file/3de568f8597b94bda53149c7d7f5958c-Paper.pdf)

Robust Locally-Linear Controllable Embedding<br>
by Banijamali et al., AISTATS 2018<br>
[[PDF]](http://proceedings.mlr.press/v84/banijamali18a/banijamali18a.pdf)

Learning Latent Dynamics for Planning from Pixels<br>
by Hafner et al., ICML 2019<br>
[[PDF]](https://arxiv.org/pdf/1811.04551.pdf)

Planning with Diffusion for Flexible Behavior Synthesis<br>
by Janner et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/janner22a.html)

A Hierarchical Bayesian Approach to Inverse Reinforcement Learning with Symbolic Reward Machines<br>
by Zhou et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/zhou22b/zhou22b.pdf)

## BDL and Graphs (Link Prediction, Graph Neural Networks, Knowledge Graphs, etc.)

Relational Deep Learning: A Deep Latent Variable Model for Link Prediction<br>
by Wang et al., AAAI 2017<br>
[[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14346/14463)

Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs<br>
by Trivedi et al., ICML 2017<br>
[[PDF]](https://arxiv.org/pdf/1705.05742.pdf)

Graphite: Iterative Generative Modeling of Graphs<br>
by Grover et al., ICML 2019<br>
[[PDF]](https://arxiv.org/pdf/1803.10459.pdf)

Relational Variational Autoencoder for Link Prediction with Multimedia Data<br>
by Li et al., ACM MM 2017<br>
[[PDF]](https://dl.acm.org/citation.cfm?id=3126774)

Stochastic Blockmodels meet Graph Neural Networks<br>
by Mehta et al., ICML 2019<br>
[[PDF]](https://arxiv.org/pdf/1905.05738.pdf)

Scalable Deep Generative Modeling for Sparse Graphs<br>
by Dai et al., ICML 2020<br>
[[PDF]](https://arxiv.org/pdf/2006.15502.pdf)

PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks<br>
by Vu et al., NIPS 2020<br>
[[PDF]](https://arxiv.org/pdf/2010.05788.pdf)

Dirichlet Graph Variational Autoencoder<br>
by Li et al., NIPS 2020<br>
[[PDF]](https://arxiv.org/pdf/2010.04408.pdf)

Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs<br>
by Ren et al., NIPS 2020<br>
[[PDF]](https://arxiv.org/pdf/2010.11465.pdf)

GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation<br>
by Xu et al., ICLR 2022<br>
[[PDF]](https://arxiv.org/pdf/2203.02923.pdf)

Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations<br>
by Jo et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/jo22a.html)

Equivariant Diffusion for Molecule Generation in 3D<br>
by Hoogeboom et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/hoogeboom22a.html)

LIMO: Latent Inceptionism for Targeted Molecule Generation<br>
by Eckmann et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/eckmann22a/eckmann22a.pdf)

3DLinker: An E(3) Equivariant Variational Autoencoder for Molecular Linker Design<br>
by Huang et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/huang22g/huang22g.pdf)

Crystal Diffusion Variational Autoencoder for Periodic Material Generation<br>
by Xie et al., ICLR 2022<br>
[[PDF]](https://openreview.net/pdf?id=03RLpj-tc_)

OrphicX: A Causality-Inspired Latent Variable Model for Interpreting Graph Neural Networks<br>
by Lin et al., CVPR 2022<br>
[[PDF]](http://wanghao.in/paper/CVPR22_OrphicX.pdf)

## BDL and Topic Modeling

Relational Stacked Denoising Autoencoder for Tag Recommendation<br>
by Wang et al., AAAI 2015<br>
[[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9350/9980)

Scalable Deep Poisson Factor Analysis for Topic Modeling<br>
by Gan et al., ICML 2015<br>
[[PDF]](http://proceedings.mlr.press/v37/gan15.html)

Deep Latent Dirichlet Allocation with Topic-layer-adaptive Stochastic Gradient Riemannian MCMC<br>
by Cong et al., ICML 2017<br>
[[PDF]](https://dl.acm.org/citation.cfm?id=3305471)

Deep Unfolding for Topic Models<br>
by Chien et al., TPAMI 2017<br>
[[PDF]](https://ieeexplore.ieee.org/abstract/document/7869412/)

Neural Relational Topic Models for Scientific Article Analysis<br>
by Bai et al., CIKM 2018<br>
[[PDF]](https://dl.acm.org/citation.cfm?id=3271696)

Dirichlet Belief Networks for Topic Structure Learning<br>
by Zhao et al., NIPS 2018<br>
[[PDF]](http://papers.nips.cc/paper/8020-dirichlet-belief-networks-for-topic-structure-learning)

Deep Relational Topic Modeling via Graph Poisson Gamma Belief Network<br>
by Wang et al., NIPS 2020<br>
[[PDF]](https://proceedings.neurips.cc//paper/2020/hash/05ee45de8d877c3949760a94fa691533-Abstract.html)

Sawtooth Factorial Topic Embeddings Guided Gamma Belief Network<br>
by Duan et al., ICML 2021<br>
[[PDF]](http://proceedings.mlr.press/v139/duan21b/duan21b.pdf)

Poisson-Randomised DirBN: Large Mutation is Needed in Dirichlet Belief Networks<br>
by Fan et al., ICML 2021<br>
[[PDF]](http://proceedings.mlr.press/v139/fan21a/fan21a.pdf)

Torsional Diffusion for Molecular Conformer Generation<br>
by Jing et al., NIPS 2022<br>
[[PDF]](https://openreview.net/pdf?id=w6fj2r62r_H)

Knowledge-Aware Bayesian Deep Topic Model<br>
by Wang et al., NIPS 2022<br>
[[PDF]](https://openreview.net/forum?id=N2AGw9s-wvX)

## BDL and Speech Recognition/Synthesis

Unsupervised Learning of Disentangled and Interpretable Representations from Sequential Data<br>
by Hsu et al., NIPS 2017<br>
[[PDF]](https://arxiv.org/pdf/1709.07902.pdf)

Scalable Factorized Hierarchical Variational Autoencoder Training<br>
by Hsu et al., Interspeech 2018<br>
[[PDF]](https://arxiv.org/pdf/1804.03201.pdf)

Hierarchical Generative Modeling for Controllable Speech Synthesis<br>
by Hsu et al., ICLR 2019<br>
[[PDF]](https://arxiv.org/pdf/1810.07217.pdf)

Recurrent Poisson Process Unit for Speech Recognition<br>
by Huang et al., AAAI 2019<br>
[[PDF]](https://pdfs.semanticscholar.org/4970/fa3189cd9a9c817ba72082e2f3d5fc9a7df1.pdf)

Deep Graph Random Process for Relational-thinking-based Speech Recognition<br>
by Huang et al., ICML 2020<br>
[[PDF]](http://wanghao.in/paper/ICML20_DGP.pdf)

DiffWave: A Versatile Diffusion Model for Audio Synthesis<br>
by Kong et al., ICLR 2021<br>
[[PDF]](https://arxiv.org/abs/2009.09761)

WaveGrad: Estimating Gradients for Waveform Generation<br>
by Chen et al., ICLR 2021<br>
[[PDF]](https://arxiv.org/pdf/2009.00713.pdf)

Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech<br>
by Popov et al., ICML 2021<br>
[[PDF]](https://arxiv.org/pdf/2105.06337.pdf)

STRODE: Stochastic Boundary Ordinary Differential Equation<br>
by Huang et al., ICML 2021<br>
[[PDF]](http://www.wanghao.in/paper/ICML21_STRODE.pdf)

Guided-TTS: A Diffusion Model for Text-to-Speech via Classifier Guidance<br>
by Kim et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/kim22d.html)

Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme<br>
by Popov et al., ICLR 2022<br>
[[PDF]](https://openreview.net/forum?id=8c50f-DoWAu)

BDDM: Bilateral Denoising Diffusion Models for Fast and High-Quality Speech Synthesis<br>
by Lam et al., ICLR 2022<br>
[[PDF]](https://iclr.cc/virtual/2022/poster/6010)

Unsupervised Mismatch Localization in Cross-Modal Sequential Data with Application to Mispronunciations Localization<br>
by Wei et al., TMLR 2022<br>
[[PDF]](http://wanghao.in/paper/TMLR22_ML-VAE.pdf)


## BDL and Forecasting (Time Series Analysis)

DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks<br>
by Salinas et al., 2017<br>
[[PDF]](https://arxiv.org/pdf/1704.04110.pdf)

Deep State Space Models for Time Series Forecasting<br>
by Rangapuram et al., NIPS 2018<br>
[[PDF]](https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf)

Deep Factors for Forecasting<br>
by Wang et al., ICML 2019<br>
[[PDF]](https://arxiv.org/pdf/1905.12417.pdf)

Probabilistic Forecasting with Spline Quantile Function RNNs<br>
by Gasthaus et al., AISTATS 2019<br>
[[PDF]](http://proceedings.mlr.press/v89/gasthaus19a/gasthaus19a.pdf)

Adversarial Attacks on Probabilistic Autoregressive Forecasting Models<br>
by Dang-Nhu et al., ICML 2020<br>
[[PDF]](https://proceedings.icml.cc/static/paper_files/icml/2020/526-Paper.pdf)

Neural Jump Stochastic Differential Equations<br>
by Jia et al., NIPS 2019<br>
[[PDF]](https://arxiv.org/pdf/1905.10403.pdf)

Segmenting Hybrid Trajectories using Latent ODEs<br>
by Shi et al., ICML 2021<br>
[[PDF]](https://arxiv.org/pdf/2105.03835.pdf)

RNN with Particle Flow for Probabilistic Spatio-temporal Forecasting<br>
by Pal et al., ICML 2021<br>
[[PDF]](http://proceedings.mlr.press/v139/pal21b/pal21b.pdf)

End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series<br>
by Rangapuram et al., ICML 2021<br>
[[PDF]](http://proceedings.mlr.press/v139/rangapuram21a/rangapuram21a.pdf)

Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting<br>
by Rasul et al., ICML 2021<br>
[[PDF]](http://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf)

Deep Explicit Duration Switching Models for Time Series<br>
by Ansari et al., NIPS 2021<br>
[[PDF]](https://arxiv.org/pdf/2110.13878.pdf)

Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting<br>
by Rasul et al., ICML 2021<br>
[[PDF]](https://arxiv.org/pdf/2101.12072.pdf)

CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation<br>
by Tashiro et al., NIPS 2021<br>
[[PDF]](https://arxiv.org/pdf/2107.03502.pdf)

TACTiS: Transformer-Attentional Copulas for Time Series<br>
by Drouin et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/drouin22a/drouin22a.pdf)

Reconstructing Nonlinear Dynamical Systems from Multi-Modal Time Series<br>
by Kramer et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/kramer22a/kramer22a.pdf)

Deep Variational Graph Convolutional Recurrent Network for Multivariate Time Series Anomaly Detection<br>
by Chen et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/chen22x.html)

Vector Quantized Time Series Generation with a Bidirectional Prior Model<br>
by Lee at al., AISTATS 2023<br>
[[PDF]](https://arxiv.org/pdf/2303.04743.pdf)

Self-Interpretable Time Series Prediction with Counterfactual Explanations<br>
by Yan et al., ICML 2023<br>
[[PDF]](http://wanghao.in/paper/ICML23_CounTS.pdf) [Cross Referenced in [BDL and Healthcare](https://github.com/js05212/BayesianDeepLearning-Survey/blob/master/README.md#bdl-and-healthcare)]

CauDiTS: Causal Disentangled Domain Adaptation of Multivariate Time Series<br>
by Lu et al., ICML 2024<br>
[[PDF]](https://proceedings.mlr.press/v235/lu24i.html)

## BDL and Distributed/Federated Learning
Stochastic Expectation Propagation<br>
by Li et al., NIPS 2015<br>
[[PDF]](https://papers.nips.cc/paper/2015/file/f3bd5ad57c8389a8a1a541a76be463bf-Paper.pdf)

## BDL and AI4Science
Dirichlet Flow Matching with Applications to DNA Sequence Design<br>
by Stark et al., ICML 2024<br>
[[PDF]](https://arxiv.org/pdf/2402.05841)

Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models<br>
by Corso et al., ICLR 2024<br>
[[PDF]](https://openreview.net/pdf?id=KqbCvIFBY7)

## BDL and Continual/Life-Long Learning
Continual Learning with Deep Generative Replay<br>
by Shin et al., NIPS 2017<br>
[[PDF]](https://proceedings.neurips.cc/paper/2017/file/0efbe98067c6c73dba1250d2beaa81f9-Paper.pdf)

Continual Unsupervised Representation Learning<br>
by Rao et al., NIPS 2019<br>
[[PDF]](https://arxiv.org/pdf/1910.14481.pdf)

Life-Long Disentangled Representation Learning with Cross-Domain Latent Homologies<br>
by Achille et al., NIPS 2018<br>
[[PDF]](https://arxiv.org/pdf/1808.06508.pdf)

Learning Latent Representations Across Multiple Data Domains Using Lifelong VAEGAN<br>
by Ye et al., ECCV 2020<br>
[[PDF]](https://dl.acm.org/doi/abs/10.1007/978-3-030-58565-5_46)

A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning<br>
by Lee et al., ICLR 2020<br>
[[PDF]](https://arxiv.org/abs/2001.00689)

## BDL as a Framework (Miscellaneous)

Towards Bayesian Deep Learning: A Framework and Some Existing Methods<br>
by Wang et al., TKDE 2016<br>
[[PDF]](https://arxiv.org/abs/1608.06884)

Composing Graphical Models with Neural Networks for Structured Representations and Fast Inference<br>
by Johnson et al., NIPS 2016<br>
[[PDF]](https://arxiv.org/abs/1603.06277)

Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Probabilistic Interpretations<br>
by Xu et al., ICLR 2024<br>
[[PDF]](http://wanghao.in/paper/ICLR24_ECBM.pdf)

## Bayesian/Probabilistic Neural Networks as Building Blocks of BDL

Learning Stochastic Feedforward Networks<br>
by Neal et al., Technical Report 1990<br>
[[PDF]](https://www.cs.toronto.edu/~hinton/absps/sff.pdf)

A Practical Bayesian Framework for Backprop Networks<br>
by MacKay et al., Neural Computation 1992<br>
[[PDF]](https://pdfs.semanticscholar.org/b0f2/433c088591d265891231f1c22424047f1bc1.pdf)

Keeping Neural Networks Simple by Minimizing the Description Length of the Weights<br>
by Hinton et al., COLT 1993<br>
[[PDF]](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.44.3435)

Bayesian Learning via Stochastic Gradient Langevin Dynamics<br>
by Welling et al., ICML 2011<br>
[[PDF]](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.441.3813&rep=rep1&type=pdf)

Practical Variational Inference for Neural Networks<br>
by Alex Graves, NIPS 2011<br>
[[PDF]](https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks)

Auto-Encoding Variational Bayes<br>
by Kingma et al., ArXiv 2014<br>
[[PDF]](https://arxiv.org/pdf/1312.6114.pdf) [[Code]](https://github.com/AntixK/PyTorch-VAE)

Deep Exponential Families<br>
by Ranganath et al., AISTATS 2015<br>
[[PDF]](https://arxiv.org/abs/1411.2581)

Weight Uncertainty in Neural Networks<br>
by Blundell et al., ICML 2015<br>
[[PDF]](https://arxiv.org/abs/1505.05424)

Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks<br>
by Hernandez-Lobato et al., ICML 2015<br>
[[PDF]](http://proceedings.mlr.press/v37/hernandez-lobatoc15.pdf)

Variational Dropout and the Local Reparameterization Trick<br>
by Kingma et al., NIPS 2015<br>
[[PDF]](https://arxiv.org/pdf/1506.02557.pdf)

The Poisson Gamma Belief Network<br>
by Zhou et al., NIPS 2015<br>
[[PDF]](http://papers.nips.cc/paper/5645-the-poisson-gamma-belief-network)

Deep Poisson Factor Modeling<br>
by Henao et al., NIPS 2015<br>
[[PDF]](http://papers.nips.cc/paper/5786-deep-poisson-factor-modeling)

Natural-Parameter Networks: A Class of Probabilistic Neural Networks<br>
by Wang et al., NIPS 2016<br>
[[PDF]](http://wanghao.in/paper/NIPS16_NPN.pdf) [[Project Page]](https://github.com/js05212/NPN) [[Code]](https://github.com/js05212/NPN)

Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks<br>
by Mescheder et al., ICML 2017<br>
[[PDF]](https://arxiv.org/pdf/1701.04722.pdf)

Stick-Breaking Variational Autoencoders<br>
by Nalisnick et al., ICLR 2017<br>
[[PDF]](https://openreview.net/forum?id=S1jmAotxg)

Bayesian GAN<br>
by Saatchi et al, NIPS 2017<br>
[[PDF]](https://arxiv.org/abs/1705.09558)

Neural Expectation Maximization<br>
by Greff et al., NIPS 2017<br>
[[PDF]](https://papers.nips.cc/paper/7246-neural-expectation-maximization.pdf)

Lightweight Probabilistic Deep Networks<br>
by Gast et al., CVPR 2018<br>
[[PDF]](http://openaccess.thecvf.com/content_cvpr_2018/html/Gast_Lightweight_Probabilistic_Deep_CVPR_2018_paper.html)

Feed-forward Propagation in Probabilistic Neural Networks with Categorical and Max Layers<br>
by Shekhovtsov et al., ICLR 2018<br>
[[PDF]](https://openreview.net/forum?id=SkMuPjRcKQ)

Glow: Generative Flow with Invertible 1x1 Convolutions<br>
by Kingma et al., NIPS 2018<br>
[[PDF]](https://papers.nips.cc/paper/8224-glow-generative-flow-with-invertible-1x1-convolutions.pdf)

Evidential Deep Learning to Quantify Classification Uncertainty<br>
by Sensoy et al., NIPS 2018<br>
[[PDF]](https://papers.nips.cc/paper_files/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf)

ProbGAN: Towards Probabilistic GAN with Theoretical Guarantees<br>
by He et al., ICLR 2019<br>
[[PDF]](http://wanghao.in/paper/ICLR19_ProbGAN.pdf) [[Project Page]](https://github.com/hehaodele/ProbGAN)

Sampling-free Epistemic Uncertainty Estimation Using Approximated Variance Propagation<br>
by Postels et al., ICCV 2019<br>
[[PDF]](https://arxiv.org/abs/1908.00598)

Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors<br>
by Dusenberry et al., ICML 2020<br>
[[PDF]](https://proceedings.icml.cc/static/paper_files/icml/2020/5657-Paper.pdf)

Neural Clustering Processes<br>
by Pakman et al., ICML 2020<br>
[[PDF]](https://proceedings.icml.cc/static/paper_files/icml/2020/3997-Paper.pdf)

Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks<br>
by Kristiadi et al., ICML 2020<br>
[[PDF]](http://proceedings.mlr.press/v119/kristiadi20a/kristiadi20a.pdf)

Activation-level Uncertainty in Deep Neural Networks<br>
by Morales-Alvarez et al., ICLR 2021<br>
[[PDF]](https://openreview.net/pdf/6d7935927e30fe5bf2be87f8e871229560145392.pdf)

Bayesian Deep Learning via Subnetwork Inference<br>
by Daxberger et al., ICML 2021<br>
[[PDF]](http://proceedings.mlr.press/v139/daxberger21a/daxberger21a.pdf)

On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks<br>
by Seitzer et al., ICLR 2022<br>
[[PDF]](https://openreview.net/pdf?id=aPOpXlnV1T)

Evidential Turing Processes<br>
by Kandemir et al., ICLR 2022<br>
[[PDF]](https://openreview.net/pdf?id=84NMXTHYe-)

How Tempering Fixes Data Augmentation in Bayesian Neural Networks<br>
by Bachmann et al., ICML 2022<br>
[[PDF]](https://proceedings.mlr.press/v162/bachmann22a/bachmann22a.pdf)

SIMPLE: A Gradient Estimator for k-Subset Sampling<br>
by Ahmed et al., ICLR 2023<br>
[[PDF]](https://openreview.net/forum?id=GPJVuyX4p_h)

Collapsed Inference for Bayesian Deep Learning<br>
by Zeng et al., NeurIPS 2023<br>
[[PDF]](https://arxiv.org/pdf/2306.09686.pdf)

Variational Imbalanced Regression: Fair Uncertainty Quantification via Probabilistic Smoothing<br>
by Wang et al., NeurIPS 2023<br>
[[PDF]](http://www.wanghao.in/paper/NIPS23_VIR.pdf)

BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models<br>
by Wang et al., NeurIPS 2024<br>
[[PDF]](https://arxiv.org/pdf/2406.11675)
