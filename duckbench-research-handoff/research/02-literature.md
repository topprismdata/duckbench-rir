# 核心文献索引与审计优先级
链接为前期研究积累的入口，本包未重新联网核验每条。优先以出版方、作者、官方仓库为准；预印本和厂商案例单列。

## RD04 首批八篇（并行只读审计，按顺序裁决）
|ID|文献|入口|审计重点|
|---|---|---|---|
|P01|DeepMind Agile Soccer 2024|https://doi.org/10.1126/scirobotics.adi8022 ; https://zenodo.org/records/10793725|全样本、Table1、Fig7、蒸馏/自博弈、实体失败|
|P02|Action Space Matter 2017|https://xbpeng.github.io/projects/ActionSpace/index.html|力矩、位置、速度、肌肉动作语义及对照公平性|
|P03|PMTG 2018|https://proceedings.mlr.press/v87/iscen18a.html|轨迹生成器、线性策略、IMU、预算|
|P04|Cassie Versatile Locomotion|https://doi.org/10.1177/02783649241285161|残差负收益、长短历史、任务随机化|
|P05|Periodic Reward Composition|https://arxiv.org/abs/2011.01387|接触周期、奖励与实体步态|
|P06|BAM Extended Friction Models|https://bam.readthedocs.io/en/latest/ ; https://ieeexplore.ieee.org/abstract/document/11128640|M1–M6、辨识、留出验证、误差|
|P07|ToddlerBot|https://toddlerbot.github.io/|小型机器人结构、控制与开放资产|
|P08|Real-world Humanoid Locomotion|https://learning-humanoid-locomotion.github.io/|简单通用学习反证|

## 其他核心研究（按主题）
- 动力学：McGeer 1990 https://doi.org/10.1177/027836499000900206 ; Collins 2005 https://doi.org/10.1126/science.1107799 ; Full & Koditschek https://pubmed.ncbi.nlm.nih.gov/10562515/ ; Geyer https://pmc.ncbi.nlm.nih.gov/articles/PMC1664632/ ; Pratt https://www.cs.cmu.edu/~cga/legs/Pratt_Goswami_Humanoids2006.pdf ; MIT https://underactuated.mit.edu/ ; Grizzle https://doi.org/10.1016/j.automatica.2014.04.021
- 生物控制：Taga global entrainment；Ijspeert CPG https://doi.org/10.1016/j.neunet.2008.03.014 ; Todorov/Jordan https://www.nature.com/articles/nn963 ; Geyer/Herr https://pubmed.ncbi.nlm.nih.gov/20378480/ ; muscle synergy Tresch/Jarc https://pmc.ncbi.nlm.nih.gov/articles/PMC2818278/ ; CPG-RL https://arxiv.org/abs/2211.00458
- 经典控制：Hogan impedance https://doi.org/10.1115/1.3140702 ; Westervelt HZD https://doi.org/10.1109/TAC.2002.806653 ; Kajita preview IEEE1241826；Kuindersma https://doi.org/10.1007/s10514-015-9479-3 ; Cassie ALIP https://arxiv.org/abs/2109.14862 ; MPC/RL benchmark https://arxiv.org/abs/2501.16590
- 搜索：CMA-ES https://arxiv.org/abs/1604.00772 ; ES https://arxiv.org/abs/1703.03864 ; ARS https://arxiv.org/abs/1803.07055 ; CEM https://doi.org/10.1007/s10479-005-5724-z ; MPPI https://doi.org/10.2514/1.G001921 ; MAP-Elites https://arxiv.org/abs/1504.04909 ; damage adaptation https://doi.org/10.1038/nature14422 ; black-box RL movement primitives https://proceedings.mlr.press/v205/otto23a.html
- RL：Sutton/Barto http://incompleteideas.net/book/the-book-2nd.html ; GAE https://arxiv.org/abs/1506.02438 ; PPO https://arxiv.org/abs/1707.06347 ; SAC https://arxiv.org/abs/1801.01290 ; TD3 https://proceedings.mlr.press/v80/fujimoto18a.html ; time limits https://proceedings.mlr.press/v80/pardo18a.html ; implementation matters https://arxiv.org/abs/2005.12729 ; on-policy study https://arxiv.org/abs/2006.05990 ; reward shaping Ng/Harada/Russell 1999；reproducibility https://arxiv.org/abs/1709.06560 ; rliable https://agarwl.github.io/rliable/
- 模仿：DAgger https://proceedings.mlr.press/v15/ross11a.html ; distillation https://arxiv.org/abs/1511.06295 ; GAIL https://arxiv.org/abs/1606.03476 ; DeepMimic https://arxiv.org/abs/1804.02717 ; AMP https://xbpeng.github.io/projects/AMP/ ; ASE https://arxiv.org/abs/2205.01906 ; animal imitation https://arxiv.org/abs/2004.00784 ; AMASS https://amass.is.tue.mpg.de/ ; BeyondMimic https://beyondmimic.github.io/ ; XHugWBC https://arxiv.org/abs/2602.05791 (预印本)
- 世界模型：Dyna Sutton 1990；PETS https://arxiv.org/abs/1805.12114 ; PlaNet https://arxiv.org/abs/1811.04551 ; MBPO https://arxiv.org/abs/1906.08253 ; MBRL benchmark https://arxiv.org/abs/1907.02057 ; Dreamer https://arxiv.org/abs/1912.01603 ; TD-MPC https://arxiv.org/abs/2203.04955 ; TD-MPC2 https://arxiv.org/abs/2310.16828 ; contact gradients https://proceedings.mlr.press/v211/zhong23a.html ; XSoma https://arxiv.org/abs/2611.20452 (预印本)
- 物理：MuJoCo https://doi.org/10.1109/IROS.2012.6386109 ; capturability https://doi.org/10.1177/0278364912452673 ; MIT Cheetah actuator https://doi.org/10.1109/TRO.2016.2640183 ; Minitaur https://arxiv.org/abs/1804.10332 ; ANYmal https://doi.org/10.1126/scirobotics.aau5872
- 技能与博弈：Options https://doi.org/10.1016/S0004-3702(99)00052-1 ; Option-Critic https://doi.org/10.1609/aaai.v31i1.10916 ; HIRO https://arxiv.org/abs/1805.08296 ; DIAYN https://arxiv.org/abs/1802.06070 ; PSRO https://arxiv.org/abs/1711.00832 ; competition https://arxiv.org/abs/1710.03748 ; Re-evaluating Evaluation https://arxiv.org/abs/1806.02643 ; curriculum https://www.ijcai.org/proceedings/2020/671 ; egocentric soccer https://proceedings.mlr.press/v270/tirumala25a.html
- 鲁棒：EPOpt https://arxiv.org/abs/1610.01283 ; RARL https://proceedings.mlr.press/v70/pinto17a.html ; UP-OSI https://arxiv.org/abs/1702.02453 ; dynamics randomization https://arxiv.org/abs/1710.06537 ; SimOpt https://sites.google.com/view/simopt/home ; BayesSim https://arxiv.org/abs/1906.01728 ; ADR https://arxiv.org/abs/1910.07113 ; RMA https://arxiv.org/abs/2107.04034 ; A-RMA https://arxiv.org/abs/2205.15299 ; review https://pmc.ncbi.nlm.nih.gov/articles/PMC9038844/
- 效率：Rudin https://proceedings.mlr.press/v164/rudin22a.html ; EnvPool https://arxiv.org/abs/2206.10558 ; Sample Factory https://proceedings.mlr.press/v119/petrenko20a.html ; Walk in Park https://arxiv.org/abs/2208.07860 ; FastTD3 https://arxiv.org/abs/2505.22642 ; FastSAC https://arxiv.org/abs/2512.01996

## 复用资产入口（未审计版本与许可）
P0: https://github.com/pollen-robotics/microduck ; https://github.com/pollen-robotics/microduck_rl ; https://github.com/jonathanhawkins/microduck-lab ; https://github.com/Rhoban/bam ; https://github.com/google-deepmind/mujoco ; https://duckbench.app/skill.md ; https://openchallengelist.com/c/duckbench （第三方，不替代官方）。
P1: https://github.com/DLR-RM/stable-baselines3 ; https://github.com/google-deepmind/mujoco_playground ; https://github.com/CMA-ES/pycma ; https://github.com/google-research/rliable ; https://github.com/google-deepmind/open_spiel ; https://github.com/sail-sg/envpool ; https://github.com/younggyoseo/FastTD3 ; https://github.com/icaros-usc/pyribs ; https://github.com/adaptive-intelligent-robotics/QDax
P2: https://github.com/ml-explore/mlx ; https://mujoco.readthedocs.io/en/stable/mjx.html ; https://developer.apple.com/metal/jax/ ; https://github.com/nicklashansen/tdmpc2 ; https://github.com/danijar/dreamerv3
厂商案例： https://developer.nvidia.com/blog/using-ai-agents-to-teach-robots-like-microduck-to-walk-in-simulation/ （先核验页面及日期，不当作同行评审实验）。
