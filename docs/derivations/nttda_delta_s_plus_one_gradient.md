# NTTDA 升自旋通道的解析核梯度

## 推导：从 `gen_vind_sfu` 到总能量梯度 — 2026-10-10

**目标：** 对仓库当前 `src/nest/nttda/nttda.py` 中
`gen_vind_sfu` / `gen_rohf_response_sfu` 定义的 $\Delta S=+1$
本征值求导，并接入 `grad/nttda/driver.py`。这里推导的是当前能量实现的导数，
不另行修改其激发算符或交换关联近似。

**假设：**

- Born–Oppenheimer、TDA、绝热交换关联核、实空间轨道，沿用能量模块的限制。
- SCF 与目标 TDA 根均已收敛；目标根孤立，位移时保持同一根。
  简并点需要在简并子空间内处理导数，本推导不规定那里的选根方向。
- AO 积分直接计算；无密度拟合、X2C、溶剂响应或 NLC。
- DFT 使用固定的数值积分网格坐标与权重；包含 AO 中心导数，省略网格响应。
- 所有求和均显式写出。$A:B=\operatorname{Tr}(A^T B)$ 为实矩阵 Frobenius 内积。

**符号与布局：**

- $i,j$ 属于双占据 closed 空间 C；$u,v$ 属于单占据 open 空间 O；
  $a,b$ 属于 virtual 空间 V；$p,q$ 为任意空间 MO；$\mu,\nu,\lambda,\theta$ 为 AO。
- $C_{\mu p}$ 为 MO 系数，$S$ 为 AO 重叠矩阵，$C^T S C=I$。
  $C_c,C_o,C_v$ 是相应列块；占据数 $n_p$ 分别为 2、1、0。
- $X_{ia}$ 是从 $\bar i$（beta）到 $a$（alpha）的升自旋振幅，
  存储形状为 `(nclosed, nvirtual)`。$X$ 不含 open 空间的振幅块。
- $R$ 表示一个原子的一项笛卡尔坐标，$E_{\rm ref}$ 是参考态总能量，
  $\omega$ 是激发能，$E=E_{\rm ref}+\omega$。
- $\mathcal F^\alpha,\mathcal F^\beta$ 是参考 SCF 的实际自旋 Fock；
  它们与激发算符中的 $F^0\pm F^z$ 必须区分。
- $\mathcal K_0$ 表示代码中的 `vref0` 响应算符；
  $\mathcal K_{\rm x}^{\rm full}$、$\mathcal K_{\rm x}^{\rm LR}$ 表示正号定义的交换映射。
  $h$ 是全程精确交换比例，$\alpha$ 是长程比例，$\gamma$ 是范围分离参数。

### 1. 先明确能量代码实际使用的响应核

$$
D_o=C_o C_o^T,\qquad
D=2C_c C_c^T+D_o,\qquad D_0=\frac12D,
\qquad F^z=\frac12\mathcal K_0[D_o].
$$

→ **解释：** `gen_rohf_response_sfu` 返回的第二项是 `0.5 * delta`，
其中 `delta` 是同一响应算符作用于 open 密度的结果。$D_0$ 是每个自旋的平均参考密度。

`cache_xc_kernel(..., mo_coeff, mo_occ, spin=1)` 在传入二维空间轨道时，
先计算总密度，再给两个自旋各分配一半；这与传入两个独立自旋轨道数组的 UKS 路径不同。
已核对本地 PySCF 2.13.1 和
[PySCF 官方 `cache_xc_kernel` 源码](https://pyscf.org/_modules/pyscf/dft/numint.html#cache_xc_kernel)。

为同时表述 LDA、GGA 和 MGGA，令 $I,J,K$ 标记密度变量：
LDA 为 $\rho$，GGA 为 $(\rho,\partial_x\rho,\partial_y\rho,\partial_z\rho)$，
MGGA 再加入 $\tau$。$f^{st}_{IJ}$ 是局部能量密度对自旋 $s,t$ 的这些变量的二阶导数。

$$
f^0_{IJ}=\frac12\left(f^{\alpha\alpha}_{IJ}
-f^{\alpha\beta}_{IJ}-f^{\beta\alpha}_{IJ}+f^{\beta\beta}_{IJ}\right)
\bigg|_{\rho_\alpha=\rho_\beta=\rho_0},
\qquad
\mathcal K_0[T]=\mathcal K^0_{\rm xc}[T]
-h\mathcal K_{\rm x}^{\rm full}[T]
-(\alpha-h)\mathcal K_{\rm x}^{\rm LR}[T].
$$

→ **解释：** 第一式逐项对应 `fxc_ref`；第二式对应 `vresp`。
范围分离参数为零时省略最后一项。纯 HF 时 XC 部分为零且 $h=1$。
此通道的响应项没有 `vref1`；$F^0$ 仍然包含普通 Coulomb 势。

令 $B_{\mu\nu,I}(\mathbf r)$ 为 AO 密度变量的线性映射：
$B_{\mu\nu,\rho}=\chi_\mu\chi_\nu$，梯度分量是该乘积的空间导数，
$B_{\mu\nu,\tau}=\tfrac12\nabla\chi_\mu\cdot\nabla\chi_\nu$。

$$
\rho_{T,I}(\mathbf r)=\sum_{\mu\nu}B_{\mu\nu,I}(\mathbf r)T_{\mu\nu},
\qquad
\bigl(\mathcal K^0_{\rm xc}[T]\bigr)_{\mu\nu}
=\sum_{IJ}\int B_{\mu\nu,I}f^0_{IJ}\rho_{T,J}\,d\mathbf r,
\qquad
\bigl(\mathcal K_{\rm x}^{\rm full}[T]\bigr)_{\mu\nu}
=\sum_{\lambda\theta}(\mu\lambda|\nu\theta)T_{\lambda\theta}.
$$

→ **解释：** 核对两个 AO 密度因子线性。实轨道下积分置换对称性及 XC 二阶导数对称性给出
$L:\mathcal K_0[T]=T:\mathcal K_0[L]$，即 Frobenius 内积下的自伴性。
长程交换用带范围分离的同类积分替换最后一式。

$$
F^0=\begin{cases}
\tfrac12(\mathcal F^\alpha+\mathcal F^\beta), & \texttt{nobeta=False},\\
\tfrac12\bigl(\mathcal F^\alpha[D_0,D_0]+
\mathcal F^\beta[D_0,D_0]\bigr), & \texttt{nobeta=True}.
\end{cases}
$$

→ **解释：** `nobeta` 只替换激发算符 $F^0$ 的求值参考。
轨道仍由原来的 ROKS/AOCSCF 驻定条件确定，其响应 Hessian 不随这个开关替换。
AOCSCF 本身使用平均占据参考，两条 $F^0$ 路径一致。

### 2. 从矩阵作用得到可求导标量

定义有向 AO 跃迁密度 $T=C_vX^TC_c^T$，其一般不对称，不能在交换响应中擅自对称化。
将 `gen_vind_sfu` 中三个累加项逐项写出：

$$
(AX)_{ia}
=\sum_b(F^0+F^z)_{ab}X_{ib}
-\sum_j(F^0-F^z)_{ji}X_{ja}
 +\bigl(C_v^T\mathcal K_0[T]C_c\bigr)_{ai},
\qquad
\sum_{ia}X_{ia}^2=1,\quad AX=\omega X.
$$

→ **解释：** $F$ 的下标在此表示相应 MO 块；行布局始终为 closed，列布局始终为 virtual。
粒子使用 $F^0+F^z$，空穴使用 $F^0-F^z$ 并带整体负号。

$$
D_p=C_v(X^TX)C_v^T,\qquad D_h=C_c(XX^T)C_c^T,
\qquad P_0=D_p-D_h,\quad P_z=D_p+D_h,
\qquad
\boxed{\omega=P_0:F^0+P_z:F^z+T:\mathcal K_0[T].}
$$

→ **解释：** 用 $X$ 左乘矩阵作用并求和。
两项 Fock 收缩分别给出粒子/空穴密度；空穴的两个负号相消，使 $P_z$ 是和而不是差。
响应项来自完整的 $X^TAX$，前面没有额外 $1/2$。
非单位振幅时右侧仍精确等于 $X^TAX$，但不直接等于激发能。

对应通道系数只需要三项：

| 项 | MO 系数矩阵 | $F^0$ 权重 | $F^z$ 权重 |
| --- | --- | --- | --- |
| virtual–virtual | $X^TX$ | $+1$ | $+1$ |
| closed–closed | $XX^T$ | $-1$ | $+1$ |
| CV–CV 响应 | `ResponseTerm("CV", "CV", 1, 0)` | `vref0=1` | `vref1=0` |

### 3. 固定振幅的全微分

由本征方程和归一化的微分，$2(\delta X)^TAX=\omega\,\delta(X^TX)=0$。
因此对收敛且孤立的根不需要求解 $X^R$，但必须对轨道、参考密度和 AO 积分求导。

$$
\begin{aligned}
\delta\omega={}&\delta P_0:F^0+P_0:\delta F^0
+\delta P_z:F^z
+\frac12P_z:\mathcal K_0[\delta D_o]
+\frac12P_z:(\delta\mathcal K_0)[D_o]\\
&+2\delta T:\mathcal K_0[T]+T:(\delta\mathcal K_0)[T].
\end{aligned}
$$

→ **解释：** 对上一节标量用乘积法则，并代入
$\delta F^z=\tfrac12\mathcal K_0[\delta D_o]+\tfrac12(\delta\mathcal K_0)[D_o]$。
响应项的系数 2 来自两个跃迁密度因子；核本身的导数只有一次。

这里 $\delta\mathcal K_0$ 包括 AO 积分/密度变量的变化，也包括 XC 核对参考密度的变化。
定义 $k^{stu}_{IJK}$ 为局部能量密度的三阶导数，$\eta_\alpha=1,\eta_\beta=-1$，则核的参考密度部分为：

$$
\left.\delta f^0_{IJ}\right|_{\rm ref}
=\frac12\sum_{s,t,u\in\{\alpha,\beta\}}\sum_K
\eta_s\eta_t\,k^{stu}_{IJK}\,\delta\rho_{0,K},
\qquad \delta\rho_0=\frac12\delta\rho_D.
$$

→ **解释：** 两个自旋的核求值点都为 $\rho_0$，所以它们沿平均密度一起变化。
这部分需要三阶 XC 导数；仅求 AO 积分或二阶核的导数会漏项。
公共 `xc.response_terms` 和 `xc.fockz_terms` 已包含该项。

### 4. 对 MO 系数求导，构造 Z-vector 右端

允许一般的系数变化 $\delta C=C\kappa$，暂不要求 $\kappa$ 反对称。
定义全矩阵 $M$ 使 $\delta_C\omega=M:\kappa$。令
$V=\mathcal K_0[T]$、$W=\tfrac12\mathcal K_0[P_z]$。
固定算符时的显式 Fock 投影导数为：

$$
M^{\rm proj}_{:,V}=2C^T(F^0+F^z)C_v(X^TX),\qquad
M^{\rm proj}_{:,C}=-2C^T(F^0-F^z)C_c(XX^T).
$$

→ **解释：** 对粒子/空穴密度的两个 MO 因子分别求导，两个相同贡献相加。
这里尚未计入 $F^0,F^z$ 随参考轨道的变化。

$$
M^{\rm tr}_{:,V}=2C^TVC_cX,\qquad
M^{\rm tr}_{:,C}=2C^TV^TC_vX^T,\qquad
M^{o}_{:,O}=C^T(W+W^T)C_o.
$$

→ **解释：** 上标 $\rm tr$ 表示跃迁密度贡献；$V^T,X^T$ 表示转置。
前两项来自 $2\delta T:V$，最后一项来自 $W:\delta D_o$。
实现中两个自旋暂存矩阵相加后构成同一个空间轨道导数 $M$。

令 $n^s_p$ 为参考自旋 $s$ 的占据数，$V_s^0$ 满足
$P_0:\delta F^0=\sum_s V_s^0:\delta D_s$（只取轨道引起的变化）。
普通 ROKS 的 $n^\alpha$ 为 closed/open 上的 1，$n^\beta$ 仅 closed 为 1；
AOCSCF 则均为 $n_p/2$。

$$
M^{F^0}_{pq}=\sum_{s\in\{\alpha,\beta\}}
\bigl[C^T(V_s^0+(V_s^0)^T)C\bigr]_{pq}n^s_q,
\qquad
M=M^{\rm proj}+M^{\rm tr}+M^o+M^{F^0}+M^{\rm kernel}.
$$

→ **解释：** $M^{\rm kernel}$ 来自前一节的 XC 三阶导数两项。
$V_s^0$ 由相应参考的 Fock 响应求得，`nobeta=True` 时需使用等自旋密度处的导数。
公共驱动中的 HF 快捷路径把 $M^o$ 与 $M^{F^0}$ 合并计算，代数结果相同。

为显式给出核的参考密度贡献，定义网格上的二阶核收缩权重 $Q_{IJ}$，
以及对总密度求导的三阶核 $\bar k_{IJK}$：

$$
Q_{IJ}=\rho_{T,I}\rho_{T,J}
+\frac12\rho_{P_z,I}\rho_{D_o,J},\qquad
\bar k_{IJK}=\frac14\sum_{s,t,u\in\{\alpha,\beta\}}
\eta_s\eta_t\,k^{stu}_{IJK},\qquad
(V^{\rm kernel})_{\mu\nu}
=\sum_{IJK}\int B_{\mu\nu,K}Q_{IJ}\bar k_{IJK}\,d\mathbf r.
$$

→ **解释：** $Q$ 的两部分分别来自跃迁响应和 $F^z$ 标量。
$\bar k$ 的 $1/4$ 包含 $f^0$ 定义中的 $1/2$，以及 $D_0=D/2$ 的链式求导系数。
因此两项核变化共同满足 $\delta\omega|_{\rm kernel}=V^{\rm kernel}:\delta D$。

$$
M^{\rm kernel}_{pq}
=\bigl[C^T(V^{\rm kernel}+(V^{\rm kernel})^T)C\bigr]_{pq}n_q.
$$

→ **解释：** 对 $D=C\operatorname{diag}(n)C^T$ 的两个轨道因子求导。
代码把同一贡献拆成 alpha/beta 探针；普通无外磁场泛函在等自旋密度处具有自旋交换对称性，
两个自旋的核导数相等，所以使用实际自旋占据数后求和与上式的总占据数表示一致。

### 5. 核导数、轨道驻定条件与重叠导数

上标 $[R]$ 表示固定 AO 密度矩阵元素和振幅，只对 AO 中心、积分及核求值求导。
于是显式部分为：

$$
d_R=P_0:(F^0)^{[R]}
+\frac12P_z:\mathcal K_0^{[R]}[D_o]
+T:\mathcal K_0^{[R]}[T].
$$

→ **解释：** 固定 AO 系数时 $P_0,P_z,T,D_o$ 的矩阵元素不变，
但它们生成的实空间密度随 AO 中心移动，相关导数包含在 $[R]$ 中。
因此 DFT 的第二、三项仍需 AO 中心导数和参考密度导致的三阶 XC 项。

令 $r$ 为独立轨道旋转的 SCF 驻定残差。普通 ROKS 的残差选取为
$r_{ui}=\mathcal F^\beta_{ui}$、
$r_{ai}=\mathcal F^\alpha_{ai}+\mathcal F^\beta_{ai}$、
$r_{au}=\mathcal F^\alpha_{au}$；AOCSCF 使用 $r_{pq}=(n_q-n_p)\mathcal F_{pq}$。
令 $U^T=-U$ 表示独立旋转，将每个自由元 $U_{pq}$ 打包为向量 $u$。

$$
\kappa^R=U^R-\frac12S^R_{\rm MO},\qquad
S^R_{\rm MO}=C^TS^{[R]}C,\qquad
b_{pq}=M_{pq}-M_{qp},\qquad H=\frac{\partial r}{\partial u},
\qquad H^Tz=b.
$$

→ **解释：** 第一式来自 $C^TSC=I$ 的导数；反对称部分用 SCF 响应消除。
$M:U=b^Tu$，因此右端正是公共 `pack_m_matrix` 的反对称收缩。
这里采用正号右端，后面的 Z-vector 核导数项必须取负号。

定义 $G$ 为完整轨道伴随矩阵，使 $\delta_C(z^Tr)=G:\kappa$。
对驻定条件 $r=0$ 求导并左乘 $z^T$：

$$
0=r^{[R]}+Hu^R+\left.\delta_Cr\right|_{\kappa=-S^R_{\rm MO}/2},
\qquad
b^Tu^R=-z^Tr^{[R]}+\frac12G:S^R_{\rm MO}.
$$

→ **解释：** $H^Tz=b$ 消除了未知的反对称轨道导数。
$G$ 必须定义在完整的 MO 矩阵空间上，才能同时处理重叠引起的对称变化。

### 最终结果与实现映射

$$
\boxed{
\frac{dE}{dR}=\frac{dE_{\rm ref}}{dR}
+d_R-z^Tr^{[R]}-\frac12(M-G):S^R_{\rm MO}.
}
$$

→ **解释：** 总梯度等于参考态总能量梯度，加显式激发项、轨道松弛项和重叠项。
这与 `driver._orbital_gradient` 的符号一致；不能只返回 $d_R$ 或仅补充一项交换积分导数。

| 推导量 | 实现位置 |
| --- | --- |
| $X$、$T$、两个 Fock 投影、单一 CV–CV 响应 | `grad/nttda/delta_s_plus_one.py` |
| $F^0,F^z,\mathcal K_0$ 的能量定义 | `nttda/nttda.py:gen_vind_sfu, gen_rohf_response_sfu` |
| $M$、$d_R$ 的组装与 J/K 收缩 | `grad/nttda/driver.py`, `common.py` |
| 半局部核及其参考密度/AO 导数 | `grad/nttda/xc.py` |
| $H^Tz=b$、$G$、$z^Tr^{[R]}$ 的密度探针 | `grad/nttda/roks.py`, `ensemble.py` |
| 参考梯度与激发梯度相加 | `grad/nttda/__init__.py:Gradients.kernel` |

`tools/symbolic/check_nttda_delta_s_plus_one.py` 用确定性随机张量验证
$X^TAX$ 与标量表达式、一般 MO 变化下的 $M$、以及固定 AO 系数的显式导数。
它只检查代数；真实体系还须通过生产 `gen_vind_sfu` 的标量闭合、
ROKS/AOCSCF 的轨道微扰和总能量中心有限差分验证。

## 数值验证 — 2026-10-10

本次验收只运行五个 `src/nest/grad/tests/test_nttda*.py` 文件，
结果为 **23 passed, 173 subtests passed**。环境为 PySCF 2.13.1、NumPy 2.2.6，
OMP/BLAS/MKL 均设为单线程。命令：

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest src/nest/grad/tests/test_nttda*.py -q
python tools/symbolic/check_nttda_delta_s_plus_one.py
```

| 验证层次 | 升自旋通道的检查与结果 |
| --- | --- |
| 随机张量代数 | 标量闭合误差 $4.44\times10^{-16}$；MO 导数误差 $6.44\times10^{-10}$；显式导数误差 $7.42\times10^{-11}$ |
| 生产能量标量闭合 | 六类泛函、`nobeta` 两种取值、本征向量及未归一化随机向量，误差均小于 $10^{-11}$ |
| 完整 MO 导数 $M$ | 一般非反对称轨道扰动，五点差分步长 $10^{-4}$，误差小于 $10^{-8}$；AOCSCF 双重态参考另有独立检查 |
| 固定 AO 系数的核导数 $d_R$ | 中心差分步长 $10^{-4}$ Bohr；HF 误差小于 $10^{-6}$，DFT 小于 $3\times10^{-6}$ Eh/Bohr |
| ROKS 总能量梯度 | NOH/STO-3G、参考自旋 $S=1$、第 2 根、所有原子；步长 $2\times10^{-4}$ Bohr，最大分量误差小于 $10^{-5}$ Eh/Bohr |
| AOCSCF 总能量梯度 | CH2/STO-3G、参考自旋 $S=1$、第 1 根、一个 H 原子的三个方向；步长 $2\times10^{-3}$ Bohr，最大分量误差小于 $5\times10^{-5}$ Eh/Bohr |
| 公共接口与批处理 | 总梯度与参考/激发分量一致，scanner 能量/梯度一致，极小内存分批保持结果，原有 $\Delta S=-1,0$ 测试通过 |

六类泛函为 HF、SVWN、PBE、TPSS、M06-2X、CAM-B3LYP；
总能量梯度还检查了 M06-2X 和 CAM-B3LYP 的 `nobeta=True`。
DFT 物理测试使用 0 级固定网格，因此验证的是上文明确规定的无网格响应导数。

两个测试数值细节保留在回归中：

- NOH 的升自旋空间只有 14 维。部分泛函用少量初猜的 Davidson 残差停留在
  $1.0$–$1.8\times10^{-9}$ 左右，略高于 $10^{-9}$ 收敛要求；完整空间初始化后降至约
  $10^{-14}$。总梯度测试据此请求全部 14 个根，未降低收敛要求或跳过收敛检查。
- 完整 MO 导数的两点差分在 $10^{-6}$ 步长下有消减误差，个别误差达到
  $4.3\times10^{-8}$。步长扫描确认五点差分在 $10^{-4}$ 时的升自旋误差低于
  $6\times10^{-10}$；回归采用这一差分方案，保留原 $10^{-8}$ 验收阈值。

## 合并已有梯度分支后的验证 — 2026-10-10

整合到 `feat/nttda-grad` 时保留了该分支原有的闭壳层到三重态测试：
LiH/STO-3G、HF/PBE、$S_i=0$ 到 $S_f=1$，总梯度的有限差分误差小于
$10^{-6}$ Eh/Bohr。此小体系同样使用完整激发空间初始化，保留原有
$10^{-10}$ TDA 收敛阈值和位移后收敛检查。

合并后的五个 NTTDA 梯度测试文件得到 **24 passed, 175 subtests passed**；
原分支的 `examples/grad/02_nttda_grad.py` 也运行通过。
