# NTTDA 三个自旋通道的非绝热耦合

## 推导：从激发梯度和态间密度到 NAC — 2026-10-10

**目标：** 在 NEST 的同一个 `NTTDA` 对象内计算两个不同根之间的 NAC，
覆盖 `deltaS=-1,0,+1`，复用已经验证的激发梯度及态间密度。

**假设：** 实空间轨道、实 TDA 振幅、收敛的 ROKS 或 AOCSCF 参考；
采用自旋适配辅助波函数（AWF）定义。DFT 导数与梯度模块一致，不含积分网格响应。
这里只耦合同一目标总自旋的根；不是不同 `deltaS` 对象之间的自旋轨道耦合。
不包含密度拟合、X2C、溶剂或 NLC 响应。以下非简并推导的能隙分母不能在交叉点直接使用。

**符号：** $R$ 为一个核笛卡尔坐标；$I,J$ 为态；$P,Q$ 为组态；
$i,j$、$u,v$、$a,b$ 分别为闭壳层 C、开壳层 O、虚轨道 V；$p,q$ 为任意空间 MO；
$\mu,\nu$ 为 AO；$\sigma$ 为电子自旋。$S=n_{\mathrm O}/2$ 为参考总自旋，
$S_t=S+\Delta S$ 为目标总自旋。求和均显式写出。
矩阵内积 $B:C=\sum_{pq}B_{pq}C_{pq}$，不隐含转置。

### 1. 振幅导数和移动组态基

在物理组态空间内，NTTDA 为实对称本征问题。$\Delta S=-1$ 的 OO 块
必须先去掉单位阵方向；该方向是冗余零态，不能作为 NAC 的根。

$$
A X_J=\omega_J X_J,\qquad X_I^{T}X_J=\delta_{IJ},\qquad
|\Psi_J\rangle=\sum_P X_{PJ}|\Phi_P\rangle .
$$

→ **解释：** NEST 的根在欧氏振幅度量下正交归一；降低自旋通道的这个度量
仅在 OO 无迹子空间与自旋适配组态度量相同。$\omega_J$ 为激发能。

$$
d_{IJ}^{R}=\langle\Psi_I|\partial_R\Psi_J\rangle
=X_I^T\partial_R X_J+\sum_{PQ}X_{PI}X_{QJ}
\langle\Phi_P|\partial_R\Phi_Q\rangle .
$$

→ **解释：** 对 AWF 求导产生振幅变化和组态基变化两部分。后者不能通过
单个态的总能量梯度直接得到。

$$
X_I^T(\partial_R A)X_J+\omega_I X_I^T\partial_R X_J
=\omega_J X_I^T\partial_R X_J,\qquad I\ne J .
$$

→ **解释：** 对本征方程求导并左乘 $X_I^T$；正交性消去激发能导数。
这里 $\partial_R A$ 包含 SCF 轨道响应，并与组态联络使用同一轨道规范。

$$
h_{IJ}^{R}=X_I^T(\partial_R A)X_J
=\frac{G^R[X_I+X_J]-G^R[X_I-X_J]}{4},\qquad
G^R[X]=X^T(\partial_R A)X .
$$

→ **解释：** 对称二次型的极化恒等式给出态间分子项。`grad_elec` 恰为这个
二次型；**不归一化**和差振幅，也不调用会添加参考态梯度的 `Gradients.kernel`。
继承参考代码的命名，函数叫 `get_hf_interstate_numerator`；这里的 HF 指
Hellmann–Feynman 型振幅项，不是只支持 Hartree–Fock，也不宣称是真实多电子哈密顿量矩阵元。

### 2. 用态间一粒子密度计算组态联络

$$
\Gamma^{IJ}_{pq}=\langle\Psi_I|\sum_\sigma
a_{p\sigma}^{\dagger}a_{q\sigma}|\Psi_J\rangle,\qquad
d_{IJ,\mathrm{CSF}}^R=\sum_{pq}\Gamma^{IJ}_{pq}
\langle\phi_p|\partial_R\phi_q\rangle .
$$

→ **解释：** 空间轨道变化由自旋标量的一体算符生成；这个关系对三个目标自旋均成立。
直接使用 `nest.nttda.nttda.transition_dm`，其下标为零基根编号。该函数针对不同正交态；
参考占据乘态重叠的对角项因正交性消失。

令 $C^T S_{\mathrm{AO}}C=1$，并在 AO 系数空间写 $\partial_R C=C\kappa^R$。
记 $T^R_{\mu\nu}=\langle\chi_\mu|\partial_R\chi_\nu\rangle$。

$$
\kappa^R+\kappa^{R,T}=-C^T(\partial_R S_{\mathrm{AO}})C,
\qquad \partial_R S_{\mathrm{AO}}=T^R+T^{R,T} .
$$

→ **解释：** 对 MO 正交关系求导，固定了系数导数的对称部分。

$$
\langle\phi_p|\partial_R\phi_q\rangle
=\left[\kappa_A^R+C^T\frac{T^R-T^{R,T}}{2}C\right]_{pq},
\qquad \kappa_A^R=\frac{\kappa^R-\kappa^{R,T}}{2} .
$$

→ **解释：** AO 联络的对称部分抵消正交化项，剩下反对称轨道旋转和反对称 AO 联络。
PySCF `get_ovlp` 给出 bra 导数，故在核对应的 AO 行上它是 $T^{R,T}$ 而不是 $T^R$。
`_ao_connection` 特别保留此转置和符号。

$$
M=\frac{\Gamma^{IJ}-\Gamma^{IJ,T}}{2},\qquad
b_{pq}=M_{pq}-M_{qp},\qquad H^T z=b .
$$

→ **解释：** 只有密度的反对称部分与轨道旋转收缩。对独立的 C/O/V 间旋转打包，
复用 ROKS 或 AOCSCF 的转置 Hessian 求解；同一占据空间内取平行输运规范。
`roks.pack_m_matrix` 的差分已经包含因子 2，不能再额外乘 2。

$$
d_{IJ,\mathrm{orb}}^R=-z^T F^{[R]}
 +(M-G_z):\kappa_{\mathrm{sym}}^R,\qquad
\kappa_{\mathrm{sym}}^R=-\frac12C^T(\partial_R S_{\mathrm{AO}})C .
$$

→ **解释：** $F^{[R]}$ 表示固定 MO 系数的参考轨道残差核导数，$G_z$ 是
该残差与 $z$ 收缩后对轨道系数的导数矩阵。该表达式就是现有梯度的伴随收缩。
$M:\kappa_{\mathrm{sym}}^R=0$，但 $G_z$ 的重叠响应仍需保留。
`nobeta` 只改变激发算符；此处必须微分实际 SCF 参考，不能把激发的等自旋密度修正加进来。

求解后检查原始伴随方程的残差。原有 Krylov 实现用试探向量的线性相关阈值终止，
在 C/O/V 内部旋转轨道后可能留下约 $10^{-6}$ 的方程残差。
共享求解器现在最多进行两次缩放残差校正；这不改变 Hessian 或右端项。
每次 Krylov 求解遵循 `cphf_max_cycle`，最终要求最大绝对残差不超过
`max(cphf_conv_tol, 1e-12)`。另行检查最终组装的伴随矩阵，并测试轨道规范不变性。

### 3. 三个通道与自旋适配相位

张量分量采用 Condon–Shortley 约定：

$$
T_{pq}^{1,-1}=a_{p\beta}^{\dagger}a_{q\alpha},\qquad
T_{pq}^{1,0}=\frac{a_{p\alpha}^{\dagger}a_{q\alpha}
-a_{p\beta}^{\dagger}a_{q\beta}}{\sqrt2},\qquad
T_{pq}^{1,+1}=-a_{p\alpha}^{\dagger}a_{q\beta} .
$$

→ **解释：** 最高权参考为 $|S,S\rangle$；相邻参考分量由归一化的自旋降低算符生成。
下列行列式展开仅用于独立重叠验证，解析 NAC 不枚举行列式。

对于 $\Delta S=+1$，只有 C→V 振幅：

$$
|\Phi_{ia}^{+}\rangle=T_{ai}^{1,+1}|S,S\rangle,\qquad
\Gamma_{ji}^{IJ}=-\sum_a X_{ia}^{I}X_{ja}^{J},\qquad
\Gamma_{ab}^{IJ}=\sum_i X_{ia}^{I}X_{ib}^{J} .
$$

→ **解释：** 从闭壳层 beta 空穴向虚轨道 alpha 粒子激发，产生 $S_t=M_t=S+1$ 的单行列式。
态间密度只有 CC/VV 块，所以独立 C/O/V 间旋转的 CSF 源为零；**AO 联络仍不为零**。
即使参考 $S=0$，此通道也成立。

对于 $\Delta S=0$，秩一组态在适当归一化后为：

$$
|\Phi_{qp}^{0,1}\rangle\ \propto\
\sqrt{\frac{S}{S+1}}T_{pq}^{1,0}|S,S\rangle
-\frac1{\sqrt{S+1}}T_{pq}^{1,+1}|S,S-1\rangle .
$$

→ **解释：** CO、CV、OV 采用这个耦合；OO 振幅是对应参考行列式的一个标量。
此通道要求 $S\geq1/2$。

$$
|\Phi_{ia}^{0,0}\rangle=-\frac1{\sqrt2}
\left(a_{a\alpha}^{\dagger}a_{i\alpha}
+a_{a\beta}^{\dagger}a_{i\beta}\right)|S,S\rangle .
$$

→ **解释：** **CV(0) 的负号是 NEST 的能量和跃迁密度约定**。BSCC 参考展开取正号，
不能直接照搬；否则 OO–CV(0)、CO–CV(0)、OV–CV(0) 等交叉项都会错号。

对于 $\Delta S=-1$：

$$
|\Phi_{qp}^{-}\rangle=N_{qp}\left[
\sqrt{\frac{2S-1}{2S+1}}T_{pq}^{1,-1}|S,S\rangle
-\sqrt{\frac{2S-1}{S(2S+1)}}T_{pq}^{1,0}|S,S-1\rangle
+\frac{T_{pq}^{1,+1}|S,S-2\rangle}{\sqrt{S(2S+1)}}\right] .
$$

→ **解释：** 要求 $S\geq1$。CO/OV 的 $N_{qp}=\sqrt{2S/(2S+1)}$，
CV 的因子为 1，OO 的因子为 $\sqrt{(2S-1)/(2S+1)}$。
OO 组态有一个冗余方向，不能逐个归一化；输入物理根必须无迹。
三个通道的行列式态间密度均在随机正交振幅上与 NEST 的闭式密度独立比较。

### 4. 最终结果、ETF 选项及数值验证

$$
n_{IJ}^{R}=h_{IJ}^{R}+(\omega_J-\omega_I)
\left(d_{IJ,\mathrm{AO}}^R+d_{IJ,\mathrm{orb}}^R\right),\qquad
d_{IJ}^R=\frac{n_{IJ}^R}{\omega_J-\omega_I} .
$$

→ **解释：** `ediff=False` 返回能量加权耦合 $n_{IJ}$，单位 Eh/Bohr；
`ediff=True` 返回导数耦合，单位 Bohr$^{-1}$。实态交换时 $n_{JI}=n_{IJ}$、
$d_{JI}=-d_{IJ}$。接近简并时仍可返回分子；除能隙会明确报错。
不同调用直接使用当前根相位，不用上一次根的振幅擅自翻转符号。

$$
d_{IJ,\mathrm{ETF}}^R=\frac{h_{IJ}^R}{\omega_J-\omega_I}
+d_{IJ,\mathrm{orb}}^R .
$$

→ **解释：** `use_etfs=True` 消去反对称 AO 联络，**保留轨道响应**。
这与 BSCC 把整个 CSF 联络设零的选项不同。对只有 CC/VV 密度的升自旋通道两者等价；
对含 C/O/V 间密度的降低及保持自旋通道，一般不等价。这里不把标准单激发空间中
“ETF 等于仅振幅项”的结论直接外推到张量组态空间。

常规 CIS/SF-CIS 中梯度复用、CSF 联络及反对称 AO 项的背景见
[Zhang–Herbert, JCP 141, 064104 (2014)](https://doi.org/10.1063/1.4891984)。
本文的三通道扩展来自 NEST 的振幅与密度约定及上面的直接微分，不以该文作为 NTTDA 实现的证明。

$$
d_{IJ}^R\approx\frac{
\langle\Psi_I(R-h)|\Psi_J(R+h)\rangle
-\langle\Psi_I(R+h)|\Psi_J(R-h)\rangle}{4h} .
$$

→ **解释：** 对称跨几何差分为二阶精度；每次位移都重新收敛 SCF/NTTDA，
用完整 AWF 重叠做根匹配和相位对齐。行列式重叠是 alpha/beta 占据 MO 重叠子式行列式的乘积，
包含原子基和轨道变化。DFT 默认固定网格，与解析导数的定义对应。
`finite_difference` 始终返回完整导数耦合，不随 `ediff` 或 `use_etfs` 改变定义。

**代码对应：** `src/nest/nac/nttda.py`（解析组装），
`src/nest/nac/_nttda_overlap.py`（独立自旋适配重叠），
`src/nest/nac/tests/test_nttda_nac.py`（代数、相位及核位移回归）。
测试中的小行列式检验承担符号验证，不再另建一份重复的符号脚本。

### 本地验收 — 2026-10-10

在单线程 `nest-dev` 环境下，运行 `src/nest/nac/tests/test_nttda_nac.py` 和
`src/nest/grad/tests/test_nttda*.py`：**72 项测试及 175 个子测试全部通过**，
其中 48 项为新增 NAC 检查。修改涉及的 Python 文件通过 Ruff 检查。

- NOH / STO-3G / 参考自旋 1：三个通道共 24 组核位移比较，全部满足
  最大绝对误差小于 $3\times10^{-5}$ Bohr$^{-1}$。
  覆盖 ROKS 的 HF、SVWN、PBE、TPSS、M06-2X（含 `nobeta`）、CAM-B3LYP，
  以及 AOCSCF 的 HF、PBE。DFT 使用 level-0 固定网格。
- 额外检查闭壳层 LiH 的升自旋、LiH$^+$ 双重态参考的保持自旋，
  以及没有闭壳层和虚轨道的高自旋 H$_3$ 参考的降低自旋。
- 代数检查覆盖总自旋最高权性质、态间密度、非连续占据索引、矩形 MO 重叠、
  根重排和相位、态交换、能隙缩放、原子选择及 C/O/V 内轨道旋转不变性。
- 边界检查覆盖未收敛参考/根、未收敛轨道响应、近零能隙和不支持的参考扩展。

这些是小分子实现验证，不替代生产体系的基组、网格和态跟踪收敛检查。
