# Proposal for 机制发现 Benchmark

## 问题背景

* 符号回归旨在从观测数据中发现能够简洁描述变量关系的数学表达式 (Schmidt & Lipson, 2009; Makke & Chawla, 2024)；
* 然而，符号回归发现的可观测 input-output behavior 并不必然揭示产生该 behavior 的 internal mechanistic structure，后者要求反映产生这一 behavior 的内部组成、活动以及组织方式 (Machamer et al., 2000; Kaplan & Craver, 2011)；
* Craver & Kaplan 将这种 “只描述输入-调节因素-输出之间关系，却不描述其 relevant internal causal structure” 的模型称为 phenomenal model (Kaplan & Craver, 2011; Craver & Kaplan, 2020)。
* 相比于知道 phenomenal model 所描述的可观测规律，科学揭示通常还试图说明 “该规律如何由更基础的内部组成、活动、以及组织方式产生” (Hempel & Oppenheim, 1948; Machamer et al., 2000; Bechtel & Abrahamsen, 2005)  
    - 例如：开普勒第三定律描述了恒星质量、行星公转轨道半长轴和行星公转周期之间的关系，Newton 则进一步通过运动定律与万有引力定律说明这一规律为何产生

## 研究目标

提出一个 Benchmark 以区分 Agent 发现 phenomenological model 的能力与发现 mechanism model 的能力。

## 核心研究假设 (其实也是预期结论)

当前 Agent 在发现 phenomenological model recovery 与 mechanism model recovery 之间存在显著的 performance gap。

## 方法论

### 一、如何形式化 “机制模型”？

对于观测变量 $x, y$，唯象模型 $\mathcal{P}$ 描述了它们之间的关系 $F_\mathcal{P}(x, y) = 0$；而机制模型 $\mathcal{M} = \{m_1, \dots, m_K\}$ 则要求给出一组具有明确科学语义的形式关系与约束构成 $m_i$，使得 $\mathcal{M} \Rightarrow F_\mathcal{P}(x,y)=0$。具体而言，机制模型在唯象模型的基础上进一步给出了具有明确科学语义的内部 mechanistic structure，使得这一结构能够生成可观测现象。
- 具体地，我们将 $\mathcal{M}$ 中的关系形式化地分为五类：governing/dynamical laws、conservation/balance relations、constitutive/component relations、kinematic/geometric constraints、auxiliary/regime constraints
- 例如：$\{F_g=GMm/r^2; F_g=F_a; F_a=mv^2/r; T=2\pi r/v\} \Rightarrow T^2=4\pi^2r^3/GM (开普勒第三定律)$

### 二、如何评估模型发现机制的能力？

给定一个机制模型 $\mathcal{M} \Rightarrow F_\mathcal{P}(x, y) = 0$ 作为任务实例，基于其导出的可观测变量关系 $F_\mathcal{P}(x, y)=0$ 生成观测样本 $\{(x_i, y_i)\}_{i=1}^N$。以此观测样本以及对应的可观测变量描述为唯一输入，要求 Agent 提出 $\mathcal{M}$，以导出能够描述可观测变量间关系的数学方程 $F_\mathcal{P}(x, y) = 0$，且 $\mathcal{M}$ 应该蕴含了关于未观测内部状态的可检验预测。
- 例如：给定行星周期 $T$、行星轨道半长轴 $r$、恒星质量 $M$，要求 Agent 找到 $\mathcal{M} \Rightarrow F_\mathcal{P}(T,r,M)=0$，并基于此提问 Agent：“给定 $T, r, M$，请写出行星所受的向心加速度的大小”[^注1]


### 三、如何设计任务实例？

为避免 Agent 凭借 memorization 而非 mechanism reconstruction 完成题目，不应直接采用标准教科书机制进行测试。为此，我们建议以教科书机制为 “蓝本”，针对机制模型中的 $\mathcal{M}$ 引入可能的 “变体” $\mathcal{M}' \equiv \mathcal{M} + \Delta \mathcal{M}$[^注2]，并通过将多个变体组合生成更大程度上偏离蓝本的变体 $\mathcal{M} + \Delta \mathcal{M}_1 + \Delta \mathcal{M}_2 + \cdots$，降低直接记忆标准答案的可能性。
- 针对 governing / dynamical laws，可能的变体包括（但不限于）：
    * 改变 inertia / momentum law，例如 $p=mv \rightarrow p=\gamma(v)mv$
    * 改变动力学阶数，例如 $\dot{x}=f(x,u)\rightarrow \tau\ddot{x}+\dot{x}=f(x,u)$ (**慎用**, 可能造成 $\mathcal{M}\Rightarrow\mathcal{P}$ 的推导依赖于微积分运算或微分方程求解, 我们的框架尚不支持)
    * 引入 hidden relaxation state，例如 $\dot{x}=f(x,u)\rightarrow \dot{x}=f(x,h), \tau\dot{h}=g(x,u)-h$ (**慎用**, 原因同上)
    * 引入 delayed / memory dynamics，例如 $\dot{x}(t)=f(x(t),u(t))\rightarrow \dot{x}(t)=f(x(t),u(t-\tau))$
    * 引入 subsystem coupling，例如 $\dot{x}_i=f(x_i)\rightarrow \dot{x}_i=f(x_i)+\sum_jK_{ij}h(x_j-x_i)$
- 针对 conservation / balance relations，可能的变体包括（但不限于）：
    * 加入 source / sink，例如 $\dot{M}=J_{\mathrm{in}}-J_{\mathrm{out}}\rightarrow\dot{M}=J_{\mathrm{in}}-J_{\mathrm{out}}+S(M)$ (**慎用**, 可能造成 $\mathcal{M}\Rightarrow\mathcal{P}$ 的推导依赖于微积分运算或微分方程求解, 我们的框架尚不支持；但当 $S$ 不依赖于 $M$ 时通常是可行的)
    * 加入 dissipation / leakage，例如 $\dot{E}=P_{\mathrm{in}}-P_{\mathrm{out}}\rightarrow\dot{E}=P_{\mathrm{in}}-P_{\mathrm{out}}-\lambda E$
    * 将单一 reservoir 拆分为多个相互交换的 compartments，例如 $\dot{M}=J_{\mathrm{in}}-J_{\mathrm{out}}\rightarrow\dot{M}_1=J_{\mathrm{in}}-J_{12}, \dot{M}_2=J_{12}-J_{\mathrm{out}}$
    * 改变 transfer / reaction stoichiometry，例如 $A\rightarrow B$ 变为 $2A\rightarrow B$
    * 引入 hidden storage channel，例如 $C\dot{T}=Q\rightarrow C\dot{T}+L\dot{\phi}=Q$，使部分输入能量储存在未观测内部状态 $\phi$ 中 (**慎用**, 原因同上)
- 针对 constitutive / component relations，可能的变体包括（但不限于）：
    * 将线性响应变为 nonlinear response，例如 $F=-kx\rightarrow F=-k_1x-k_3x^3$ (**适当使用**, 尽量避免 $\mathcal{M}\Rightarrow\mathcal{P}$ 的推导依赖于反解高阶方程组, 我们的框架支持得尚不完善)
    * 改变 drag / dissipation law，例如 $F_d=-cv\rightarrow F_d=-c|v|^{q-1}v$ (**适当使用**, 原因同上)
    * 改变 interaction scaling law，例如 $F=K/r^2\rightarrow F=K/r^p$
    * 将 constant material property 变为 state-dependent property，例如 $J=-D\nabla c\rightarrow J=-D(c)\nabla c$ (**慎用**, 可能造成 $\mathcal{M}\Rightarrow\mathcal{P}$ 的推导依赖于偏微分方程求解, 我们的框架尚不支持)
    * 将 linear current-voltage relation 变为 nonlinear relation，例如 $I=GV\rightarrow I=GV+\beta V^3$
    * 将 ideal equation of state 变为 non-ideal relation，例如 $PV=nRT\rightarrow [P+a(n/V)^2](V-nb)=nRT$ (**慎用**, 绝大多数 LLM 能想到的 non-ideal relation 同样是教科书知识)
    * 将 linear reaction kinetics 变为 saturating kinetics，例如 $r=kS\rightarrow r=V_{\max}S/(K+S)$。
- 针对 kinematic / geometric constraints，可能的变体包括（但不限于）：
    * 改变 no-slip constraint，例如 $v=R\omega\rightarrow v=R\omega + v_{\mathrm{slip}}$
    * 将 small-angle geometry 恢复为 exact geometry，例如 $x\simeq L\theta\rightarrow x=L\sin\theta$ (**慎用**, 可能造成方程组无法显式求解, 且 exact geometry 同样是教科书知识)
    * 改变轨道或路径几何，例如 圆轨道 $r=\mathrm{const}\rightarrow$ 椭圆轨道 $r(\nu)=a(1-e^2)/(1+e\cos\nu)$ (**慎用**, exact geometry 同样是教科书知识)
    * 改变固定几何尺度为 state-dependent geometry，例如 $A=A_0\rightarrow A=A(x)$，进而 $Q=A(x)v$
    * 取消默认的正交关系，例如 $\tau=rF\rightarrow\tau=rF\sin\theta$
- 针对 auxiliary/regime constraints，可能的变体包括（但不限于）：
    * 改变运动或几何 regime，例如 circular orbit $\rightarrow$ elliptical orbit (**慎用**, 此类变体属经典教科书知识)
    * 取消 scale-separation / negligible-term approximation，例如 $m\ll M\rightarrow$ 完整 two-body treatment (**慎用**, 原因同上)
    * 改变 steady-state assumption，例如 steady state $\rightarrow$ transient dynamics
    * 改变系统边界，例如 isolated / closed system $\rightarrow$ externally forced / open system
    * 改变热力学条件，例如 isothermal $\rightarrow$ adiabatic
    * 改变介质假设，例如 incompressible $\rightarrow$ compressible
    * 改变接触条件，例如 no-slip $\rightarrow$ finite slip
    * 取消 small-perturbation / small-angle approximation；
    * 改变 boundary condition，例如 fixed-value / Dirichlet $\rightarrow$ fixed-flux / Neumann
    * 改变 closure assumption，使原本被瞬时闭合的未观测过程具有独立动力学。

**【重要】构造变体的原则**：
1. 在构造变体时，尽量避免引入新的 Variable，否则 Variable Description 很容易泄露此变体的关键假设。取而代之地，应尽量使用数值常量。例如：与其将 “$P_{total} = P_{in} - P_{out}$” 改成 “$P_{total} = P_{in} - P_{out} - P_{loss};$ $P_{loss}$ variable description: loss of $P_{total}$”，不如将它改成 “$P_{total} = P_{in} - P_{out} - P_{loss}; P_{loss} = 1.234567$”
2. 如果必须引入新的 Variable，且新变量的语义会明显泄露 $\Delta \mathcal{M}$ 设计时，应同时引入 2~4 个作为混淆的冗余变量 distractors（即，不参与 $\mathcal{M}\Rightarrow\mathcal{P}$ 推导过程的变量）。该 distractors 不应是明显无关的随机变量，而应具有和该 variable 同等级的科学合理性。不过，极不推荐这一做法，因为它将从与我们研究目的所不同的另一个角度 —— 识别相关变量 —— 引入复杂性。
3. 在构造变体时，尽量避免使用教科书上常见的变体（例如，圆轨道->椭圆轨道、小量近似->精确形式、理想气体状态方程->真实气体状态方程、理想方程->常见现实修正）
4. 在构造变体时，原则上应避免 $\mathcal{M} \Rightarrow \mathcal{P}$ 的推导过程依赖于复杂的数学运算（例如，反解高阶方程组、求解超越方程、求解微分方程、求解偏微分方程、求解不等式等等）。当前 Framework 具有一定程度的问题反解能力，但无法自动反解过于复杂的方程。
5. 在构造变体时，先依照上述变体设计方案尽可能枚举可行的变体方案 $\{\Delta \mathcal{M}_i\}_{i=1}^N$，再将所有方案两两组合、三三组合、……、直至五五组合。完成组合后，将其中涉及自相矛盾/不自洽的变体组合删掉；将 $\mathcal{M} \Rightarrow \mathcal{P}$ 推导过程涉及 Framework 所不支持的复杂数学运算（通过实际运行 `mdbench evaluate` 判断究竟是否支持）的变体组合标记为 `.yaml.archived`。
6. 对于构造的机制模型 $\mathcal{M}$，如果存在实质不同的另一个机制模型 $\mathcal{M}'$，使得二者可导出 observationally equivalent 的 $\mathcal{P} \equiv \mathcal{P}'$，则说明该任务实例是 mechanism non-identifiable 的。此类题目应当被尽量避免。（然而，对于给定的 $\mathcal{M}$，证明不存在其它 $\mathcal{M}'$ 并不容易，因此不建议在构造任务实例时去主动识别这类情况，而要在具体评测过程中根据评测结果识别这类情况。）


### 四、如何评估 Agent 提交的 $\hat{\mathcal{M}}$？

评估包括两部分：
- 第一部分评估**唯象模型识别能力**，根据 Agent 提交的 $\hat{\mathcal{M}}$ 推导对应的 $\hat{\mathcal{P}}$，检验 $F_{\hat{\mathcal{P}}}(x, y) = 0$ 与 $F_{\mathcal{P}}(x, y) = 0$ 的符号等价性与数值等价性；
- 第二部分评估**机制模型识别能力**，选取若干出现于 $\mathcal{M} \Rightarrow \mathcal{P}$ 过程中、却不出现于最终 $F_\mathcal{P}(x, y)$ 方程中的中间变量 $z$ 作为 “机制探针”，要求 Agent 根据其所提供的 $\hat{\mathcal{M}}$ 给出 $z$ 与可观测变量 $(x, y)$ 间的数学关系。该数学关系可以与标准答案通过符号等价性和数值等价性进行客观检验。
    - 如果存在多个机制探针，应该将它们并行地各自分别发给 Agent 以要求提供数学关系。串行地顺序提供机制探针可能导致提供顺序对结果产生影响。
    - 在提供探针的过程中，原则上应禁止 Agent 基于看到的探针变量修改自己的 $\hat{\mathcal M}$。然而在实践中很难自动化地对这一点施加硬约束。因此，Framework 只会通过在提问时设置 Prompt 以施加软约束。
- 最终的核心研究假设（也即预期实验结论）为，这两个能力之间存在显著的 Performance Gap。

关于机制探针 $z$ 的选取，应满足如下原则：
- 不可观测性：$z$ 不能是可观测变量
- 有关联性：$z$ 应当出现于 $\mathcal{M} \Rightarrow \mathcal{P}$ 的推导过程中
- 客观性：$z$ 应有独立且明确的科学意义
- 非平凡性：$z$ 可以从 $\mathcal{M}$ 中推出，却不能从 $\mathcal{P} + \text{领域知识}$ 中推出 (注: 对于作为蓝本的教科书机制, $\mathcal{M}$ 本身就是领域知识，因此无法选出满足此非平凡性的 $z$)
- 可求解性：$z$ 必须能够仅根据任务中提供给 Agent 的 variables（即 input 与 auxiliary variables）和数值常数，唯一表示为一个 Framework 可处理的显式代数表达式。

### 五、如何处理 non-identifiable 的问题？

如果任务实例设计得不好，Agent 可能提交不等价于标准机制 $\mathcal{M}$、但仍然完全合理的另一个“竞争性机制模型”  $\mathcal{M}'$，使得 $\mathcal{M}'$ 同样蕴含了 $\mathcal{P}$。在这种情况下，它虽然无法通过探针测试，却不能说明它 “无法找到机制模型”。因此，这类任务实例应当被尽可能避免。

为此，我们通过两阶段方法识别并避免这一问题：
1. 在完成任务实例设计后，我们先对每个任务实例运行一次实验，重点观察那些 “Agent 能够找到正确唯象方程、但无法通过机制探针” 的任务。对于这些任务，如果 Agent 提交的机制模型 $\hat{\mathcal{M}}$ **自洽，且不等价也不显著复杂于**任务实例中设计的机制模型 $\mathcal{M}$，则删除或重新生成这一任务。根据经验（ZJD结果），这一步将识别出约 10% 的问题为 non-identifiable 的，需要重新生成。
2. 经过第一步后，我们在每个任务实例上测试多个不同 Agent 的性能。在汇报最终结果之前，我们再次观察那些 “找到了正确唯象方程、但无法通过机制探针” 的 Agent 运行结果，检查 Agent 提交的机制模型是否自洽、且既不等价也不显著复杂于任务实例中设计的机制模型。如果某个 Agent 找到了这样的竞争性机制模型，则将对应任务标记为 non-identifiable 的；如果所有 Agent 都没找到这样的竞争性机制模型，可认为原任务实例中设计的机制模型不存在明显的竞争者，并将其纳入最终结果统计。依靠第一步的初筛，此步中发现的 non-identifiable 问题比例应当尽可能少（例如，<5%）。

### 附：任务实例格式约定

【注意】：
1. 尽管该约定是用中文呈现，但实际任务实例应使用英文撰写。
2. 尽管该约定中使用了 `|` 这个 YAML 块标量语法（literal block scalar），但实际的任务实例中通常不应该存在太长的文本，请尽量避免使用此语法。
3. 文件应当被保存在 ./tasks/ 中，文件名与 task_name 保持一致。

```yaml
task_name: |
    任务实例名称, 只包含字母、数字、空格、连词符 (hyphen, "-")，不要包括其他符号。
    同一任务族（蓝本 + 变体 + 组合变体）共享相同名称，仅在结尾上进行区分。
    蓝本以 “ - Original” 结尾；变体以 “ - Variant 1” “ - Variant 2” 结尾；组合变体以对应的 “ - Variant 1-2-3” 结尾。

task_description: |
    任务描述，应当保持简洁，通常不应超过 30 词。
    任务描述应当以作为蓝本的任务实例为参考撰写。
    变体和组合变体的问题描述应当与蓝本保持完全相同，以避免通过问题描述泄露变体设计。

mutation: |
    用简单、清晰的语言说明本变体相比于蓝本所作的修改。
    对于只涉及单个修改的变体，该描述通常不应超过 30 词。
    对于涉及多个修改的组合变体，只描述 “组合了 1&2&3 变体”，并简要说明在组合的过程中所作的非平凡设计（如果有）即可，同样不应超过 30 词。
    此字段不被 Framework 使用，仅供人类理解任务时参考

mechanism_model:
    - formula: |
        变量和数学符号组成的公式，例如 “F = G * M * m / r^2”。公式可采用任意数学等价写法，不要求左侧是新变量。例如，这里也可以写成 “F - G * M * m / r^2 = 0”。
      role: |
        从以下五类中选择一类 governing/dynamical laws、conservation/balance relations、constitutive/component relations、kinematic/geometric constraints、auxiliary/regime constraints。该字段不被 Framework 使用，仅供生成变体时参考
      description: |
        用简单的语言描述该方程，通常不应超过 30 词。该字段不被 Framework 使用，仅供人类理解任务时参考
    - formula: |
        Framework 对机制模型中多个公式的出现顺序不敏感，顺序可随意调换。
      role: ...
      description: ...
    - formula: |
        为保证能够求解，应保证 “机制模型公式数量 = 中间 (internal) 变量数量 + 1”
      role: ...
      description: ...
    - formula: |
        如果存在数值常数，例如 pi=3.141592653589793, G=6.67430e-11，可以在 mechanism_model 中添加一个 “formula: G=6.67430e-11”。其物理单位会被自动推导。应避免将这些数值常数写在 variables 部分，否则数值常数的相关描述会给 Agent 提供领域方向上的提示。

phenomenal_model: |
    唯象方程，例如 T = sqrt((4 * π^2 / G / M) * a^3)。此方程要求将 target variable 写在等号左侧，其余可观测变量写在等号右侧。

variables:
    - name: 变量名，仅包括字母、数字、下划线，且不能以数字开头。Framework 对变量出现顺序不敏感
      description: 变量描述。该字段不被 Framework 使用，仅供人类理解任务时参考
      unit: 变量单位
      role: |
        从 target / input / internal / auxiliary 中选择。
        target 有且仅有一个，表示 phenomenal_model 左侧的变量。
        input 有若干个，表示 phenomenal_model 右侧依赖的变量。
        internal 有若干个（数量等同于 mechanism_model 中 formula 数量 - 1），表示 M=>P 推导过程中的中间变量。
        auxiliary 有零至若干个，表示 phenomenal_model 中未出现，但又无法像 internal 那样在推导过程中自然得到其取值的变量。为避免后续地麻烦，这里极不建议将变量声明为 auxiliary。取而代之地，一种更被推荐的方案是在 mechanism_model 中通过指定 “formula: some_auxiliary_variable = constant value (e.g., 1.23456789)” 将其设置为一个数值常数（并且在 variables 这里对应定义为 internal）。
      sampling: |
        对于 target & internal，无需提供此字段。
        对于 input & auxiliary，需要提供类似于这样的字典 {min: 1.0e20, max: 1.0e28, ood_boundary: 1.0e24, distribution: log_uniform}，其中 min-max-distribution 需根据问题背景和变量含义具体选择具有恰当科学含义的取值。
    - name: a_demo_internal_variable
      description: The description of a demo internal variable
      role: internal
    - name: ...
      description: ...
      role: ...
      sampling: ...

mechanism_probes:
    - probe: |
        一个变量名，仅包括字母、数字、下划线，且不能以数字开头，例如 z。
        作为探针的变量应满足第四部分所提到的不可观测性、有关联性、客观性、非平凡性、可求解性。
        在绝大多数情况下，建议从 variables 中 role=internal 的变量集内选择变量作为探针。
        但通常不建议选择 “P_loss = 1.23456789” “G=6.67430e-11” 这种事实上是数值常数的 internal variable 作为探针。
      description: |
        对变量的描述，应当简洁、准确（体现探针变量的客观性），不额外添加任何机制提示。
        通常建议直接复用 variables 中对应 internal 变量的 description。
      answer: |
        该变量关于 source variables（即 input 与 auxiliary variables）的数学公式。
        也可以写成该变量关于 internal variable 的数学公式，Framework 会将此处所有 internal variable 自动展开成关于 source variables 的数学公式。
    - probe: a_demo_internal_variable # variables 中的一个 internal variable
      description: The description of a demo internal variable # 直接复用对应的 description
      answer: 2.0 * another_internal_variable + input_variable_1 # Framework 会将这里面所有 internal variable 自动转换为它们关于 source variables 的表达式，因此原则上这里完全可以直接写 a_demo_internal_variable

## 实际使用探针时，会将每个 probe / description / answer 字段格式化为下面的文本：
# Frozen submitted mechanism model:
# <formula 1>
# <formula 2>
# ...
#
# Using only the mechanism model you submitted above,
# derive {probe} ({description}) as a function of the variables: {variables}.
#
# Return exactly one equation in the form:
# {probe} = <expression>
#
# The right-hand side may contain only the given variables (including internal
# variables already defined in the frozen submitted mechanism model), numeric literals,
# and mathematical operators/functions supported by the benchmark.
# Do not introduce new variables, additional equations, or prose.
# Do not modify, replace, or extend your previously submitted mechanism model.
```


[^注1] 这个问题仅供示意，向心加速度不满足第四部分提到的 “非平凡性”，无法作为一个有效的探针

[^注2] 这里的 “$+$” 并非加法，而是形式化地表示 “对 $\mathcal{M}$ 进行 $\Delta \mathcal{M}$ 这个修改，例如将 $\{F_g = GMm/r^2; F_g = F_a; F_a = mv^2/r; T = 2 \pi r/v\}$ 改成 $\{F_g = GMm/r^{1.5}; F_g = F_a; F_a = mv^2/r; T = 2 \pi r/v\}$”
