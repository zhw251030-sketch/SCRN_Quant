# SCRN-BRECQ 激活量化研究日志

本文档专门记录 SCRN-BRECQ 中激活量化失败问题的分析、修复设计和实验过程。

建立本日志的原因是：W4A8 激活量化目前是本阶段最关键的问题，可能形成后续工作的创新点。以后凡是涉及 activation quantization 的代码、配置、诊断、实验或结论，除继续写入 `DEVELOPMENT_LOG.md` 外，也必须在本文档中追加更细的实验记录。

## 记录规则

- 激活量化相关代码或文档变更：必须同步更新 `DEVELOPMENT_LOG.md` 和本文档。
- 激活量化相关实验：必须在本文档中记录，不论结果好坏。
- 长实验启动前应先有小范围诊断或 smoke 验证，避免把未知问题带入正式 run。
- 运行产物目录只作为实验证据引用，不提交 `.npy`、`.pth`、`.pt`、`.ckpt`、日志、缓存或 run 产物。
- 每次实验记录应尽量包含：实验目的、假设、代码状态、命令、配置、关键指标、诊断结果、结论和下一步。
- 如果实验发现异常但没有修复，也要记录异常现象和未解决问题，避免后续重复排查。

## 当前基线事实

截至 2026-05-04，当前已知事实如下：

- W4A32 / W-only 基本可用，packed export/restore 链路已能和 checkpoint final 对齐。
- 理论 packed 模型大小约 `0.2427 MiB`，压缩率约 `6.765x`。
- W4A8 checkpoint 已能保存和恢复 activation quantizer 状态：
  - `activation_delta_count=52`
  - `activation_zero_point_count=52`
  - packed eval 能恢复 `52` 个 activation quantizers。
- W4A8 精度明显异常：
  - W4A32 weight reconstruction 后单样本约 `11.6661 dB`
  - 打开 A8 后约 `5.0205 dB`
  - activation reconstruction 后约 `5.2277 dB`
- 已确认强信号：正式 W4A8 final checkpoint 中有 activation `delta <= 0`。
  - pre-activation-reconstruction checkpoint 中 52 个 activation delta 全部为正。
  - final checkpoint 中有 2 个 activation delta 为负。
  - 负值层：
    - `model.stage4.0.block.trans_branch.attn.proj.act_quantizer.delta`
    - `model.stage5.0.block.trans_branch.attn.proj.act_quantizer.delta`
- 当前判断：packed/export/restore 不是 W4A8 掉点主因；优先排查 activation quantizer 参数化、激活分布、重建目标和量化插入位置。

## 用户初步总结区

本节留给用户先记录你认为的潜在原因和潜在解决方案。可以直接在下面追加、改写或排序。

### 0）明显bug：activation delta出现负数
在final checkpoint里面出现了activation delta < 0 ，优化成了负数，比较明显的bug

首先这是一个需要优先修复的问题，尽管可能效果未必会好很多，但绝对很重要。此外负值出现在FFB-S的注意力模块，这也可以说明对transformer量化困难

而且要注意修复不能只是简单的防止出现负，要多层考虑。

### 1）离群值/量化粒度选择
原因是离群值的存在导致量化范围被拉宽，同时遇到了过粗的量化粒度，因此量化效果差。而相应的解决办法就是针对离群值处理如把相应离群值去除或改变量化粒度，不用现在的tensor-wise，让离群值影响的范围小。

下面是具体一点的统计指标  
第一诊断离群值存在：计算 absmax/p99.9 与 absmax/p99，即观察一下离群值对于绝大多数值的上界值比例  
第二判断通道间尺度差异：计算每个通道的absmax，对一个tensor即现在的粒度计算 max(per-channel absmax) / median(per-channel absmax)  
第三观察量化后的值：计算有效int level数，看量化后的整数值占多少整数格点如INT8应该是256，看这256被用了多少，还是判断离群值方面  
第四通过效果来判断，计算原始activation与先量化再反量化后的activation之间的均方误差  
第五就不太像这些了，是计算某个激活量化器打开后最终SNR掉了多少

以下是潜在解决方案  
第一把量化范围由现在的（可能是）per-tensor min/max改为percentile clipping/MSE clipping这两个分别是百分比截断和根据MSE效果截断  
第二把量化粒度由per-tensor改为per-channel，但代码层面要注意channel-wise = True对激活量化不成立  
第三是量化粒度改为group-wise，把channel分组，算是上面两者中间值  
第四结合SCRN网络结构通道均分FFB-S与FFB-C，二者分别作为粒度量化处理。关于这一点后续出现了重复，再强调一遍，branch-wise或更具体一点，再对结构细分，总之很具网络结构特色。出现了三遍，需要重点考虑，毕竟是有卷积有注意力的结构。不分开的粒度肯定是不行的。  
以上三点是针对量化粒度处理的，一和下面的方法是针对离群值  
第五是考虑在特定结构考虑SmoothQuant

### 2）量化具体位置
这一点还挺有意思，涉及到代码部署层面，即量化不是我们想象中的对一个结构的输出进行量化处理，而是递归替换卷积和线性层，总之就是稍微特殊一点，不是我想象中的那么简单。因此这个量化器的插入位置可能就会带来一些问题，可能量化过密导致误差累积过多？

在进行量化敏感性实验时感觉还是比较重要的，需要知道量化的具体位置在哪里

activation diagnositics 后续又看到，好像还比较重要，就是分清楚量化究竟在哪里非常重要，这样不会把不同模块的量化责任混在一起

### 3）zero_point没有参与重建优化
对于量化来说需要两个值，delta（scale）与zero_point，但后者在初始化计算后即固定，没有参与重建优化。因此可能是重建效果没那么好的原因之一。

注意这里原作者代码有把zero_point作为参数进行优化的操作，但注释掉了。因此可能成为潜在“创新点”

对于一个问题依旧是诊断方法与解决方案  
这里就不详细讲了，因为如果真处理到这里作为潜在创新需要详细考虑，但把大概内容记录一下

诊断方案包括，  
第一计算当前可表示范围，根据当前量化参数计算，与真实activation分布匹配  
第二计算activation的均值与偏斜，看zero_point是否合适  
第三计算量化均值误差  
第四计算当前与理想zero_point的差距

解决方案  
第一让zero_point参与重建优化，注意zero_point本质上是整数，可能遇到round不可导问题  
第二，不直接学delta（scale）与zero_point而是学lower，upper来推断前两个  
这样似乎温和很多，但感觉坏处是或许代码改变较大，好处是同时解决别的问题如量化范围等。还有好处是潜在创新点。

第三delta更新后同步更新zero_point，在重建中或重建后计算学习，不太懂

### 4）混合精度量化
这点将其总结为混合精度量化，主要涉及不同模块的占用显存不同，对量化敏感性不同。结合这两点考虑对该模块使用什么量化手段。这一点是我目前认为比较有可能的潜在创新点，因为不主要是算法问题，而是部署问题，根据具体的网络有特定的混合精度量化。

把这一点具体详述一下，就是发现（总感觉用这个词怪怪的）量化也有木桶定律，什么意思呢，量化有两个目的，压缩与加速，对权重量化，实现了模型大小的压缩，对激活量化，降低/压缩了推理时的显存占用，让模型更易于部署，当然加速也是源于数据格式的变化。  
因此我们现在遇到的问题是激活量化效果差，激活量化的目的是降低显存占用，而显存这个东西是有峰值的，不是全程一致的，最重要的是峰值能减小，木桶能装的水取决于最短的木条，显卡能装进的显存取决于最大的峰值显存。

当然这里要提到一点是至少我现在测试的时候感觉用到的肯定都还是FP32进行计算，这个显存能否真的减小，不好说。。实验与真实部署之间的区别要注意，并且还要注意的一点是如果做混合精度量化，推理时dtype切换也需要开销，即部署成本。

这个跟前面两点不太一样，没有诊断方案，如果有的话就是分析不同结构的敏感度，方法可能就是控制变量？加入量化看是否掉点，但要多注意网络结构，有很多细节，这个到时候具体分析吧，反正需要尽可能详细。

下面是一些具体一点的方法介绍  
最开始我想的是应该画出来关于网络结构的显存占用变化图，这个能看出来在显存方面的量化收益，但根据gpt介绍，还应该再画一张每个位置/结构被量化后的效果变化如对SNR的影响。因此总结起来应该再画一张二位散点图，因为这两个都是我们需要的收益，理想情况下把结构分为四类，显存高/低收益，效果高/低损失。  
“在SNR约束下最大化memory/latency收益，或在memory/latency约束下最大化SNR收益”

### 5）激活重建目标与最终SNR不一致
先提一个比较重要的点，感觉后续可以写进论文里面，回归/恢复任务比分类任务更容易受activation误差伤害，因为分类任务输出的是logits，中间的误差可能不会导致最终排序变化。而SCRN这种恢复任务输出本身是一张连续值地震剖面，每个采样点的幅值最终都会进入SNR。

激活重建时优化的目标是让中间某次局部MSE降低，可能不一定会让最终SNR变好。但我感觉这不是一样的吗，可能是解释是存在后续结构把局部误差变大

提到了几点，或许算解决方案？

第一是当前代码的asym是False，这会忽略后续模块的输入可能已经产生误差，因此可能需要改进，但不太懂原来为什么会这么设，感觉本来就是错误的，感觉这一点还是比较重要的，因为这个有问题就能说明小标题为什么有问题了。  
第二教师模型不同FP32而是用W4A32  
第三提到把重建目标换了，但也不是完全换，换成加权和稀泥，这里给几个考虑对象，最终输出MSE/残差target/频域loss

### 6）校准数据覆盖
这里涉及到两部分，第一是1024的校准数据可能不够？但其实训练集也才10750个patch，不好说，我感觉应该加不了了，所以这里可能更多考虑第二方面，校准集初始化。

在量化重建前对delta（scale）与zero_point要先初始化再重建优化，虽然有1024个样本，但初始化只选64个，而第二点提到zero_point是初始化固定后不再参与优化的，因此可能会导致效果差。因此对应的解决方法就是初始化的样本就选用1024个。

还有就是对选择的校准样本进行分布诊断  
统计每层每样本activation absmax分布、calib max vs eval max、calib p99.9 vs eval p99.0、eval scale utilization/effective levels。反正就是这些，只是给后续如果考虑这点时提供一些思路，进而诊断并选用更具代表性的数据。

### 7）代码工程层面
关于现在的代码存在一些可能的问题，会影响后面的实验，这里先记录一下

第一，channel-wise = True对激活量化不成立

第二，当前代码的asym是False，这会忽略后续模块的输入可能已经产生误差

第三，当前代码只支持2-8bit的量化，前面有很多建议是有关改变量化粒度如A16，但要注意这里最高是A8，需要后续处理时注意

第四，Fisher相关代码在激活重建时并没有用上，硬编码用的是MSE，不好说会不会有用，但先写在这里

第五，activation_lr = 4e-4，由于只有一个delta参数需要学习，这个学习率可能有点点大？

## Codex 深度分析与实验路线整理（2026-05-04）

本节将用户初步总结转写为更系统的技术分析和后续实验路线。这里的核心判断是：当前 W4A8 失败不应被理解为一个单点 bug，而应理解为一个由“量化器合法性、激活分布、量化粒度、插入位置、重建目标、校准代表性和部署约束”共同构成的问题簇。负 `delta` 是最明确、最优先修的硬错误，但它更像是暴露系统不稳定性的窗口，而不是全部原因。

### 1. 问题总框架

当前 W4A8 路径已经证明三件事：

1. 权重量化本身不是主瓶颈。W4A32 / W-only 在单样本和 packed restore 上都能保持较高 SNR，说明 AdaRound 权重量化、checkpoint 保存、packed export/restore 的主链路基本可信。
2. 激活量化一打开就出现断崖式掉点。W4A32 weight recon 后约 `11.6661 dB`，A8 初始化后约 `5.0205 dB`，这说明主要误差来自 activation quantization 的表示方式或插入方式，而不是 activation reconstruction 没训练够。
3. activation reconstruction 只能小幅恢复。A8 reconstruction 后约 `5.2277 dB`，只提升约 `0.207 dB`，说明当前重建目标或参数化空间没有把模型带回有效区域。

因此后续不能只做“把负 delta clamp 成正数”这种症状修补。正确路线应是先建立诊断工具，把每个 activation quantizer 的位置、分布、量化误差、最终 SNR 敏感性和资源收益都量化出来，再逐个验证假设。

### 2. 量化器合法性：负 delta 是必须优先处理的硬错误

当前 `UniformAffineQuantizer` 的 activation `delta` 在 `leaf_param=True` 时直接注册为 `nn.Parameter`，前向直接使用 `x / delta`。这意味着优化器可以把 `delta` 推到 0 以下。一旦 scale 为负，仿射量化的物理意义被破坏：整数网格方向反转，`zero_point` 不再表示浮点 0 的整数位置，clamp 区间也不再对应正常动态范围。

负值出现在 `stage4/stage5` 的 transformer attention projection 层，这一点很有信息量。它说明最不稳定的不是普通卷积层，而是 FFB-S 注意力分支深层投影。这个现象同时支持两个判断：

- 代码层面必须保证 activation scale 正值，至少要提供诊断和保护。
- 算法层面 transformer 分支的 activation 分布可能比 CNN 分支更难量化，需要单独统计和单独处理。

后续修复不应只做最终 checkpoint 前的 clamp。更合理的实验序列是：

1. 先在 verify/diagnostics 中报告所有 activation delta 的 `min/max/non_positive_count/offender_layers`。
2. 然后验证 optimizer step 后 clamp 到 `eps` 的最小修复能否消除负 scale。
3. 再比较更规范的正值参数化，例如 `softplus(raw_delta) + eps` 或 log-scale 参数化。
4. 同时观察正值约束是否只是消除了非法值，还是也改善了 SNR。如果合法性修复后 SNR 仍接近 5 dB，就说明主因还包括分布、粒度或重建目标。

### 3. 离群值与量化粒度：tensor-wise A8 可能被少数极值支配

用户总结中的离群值诊断非常关键。A8 名义上有 256 个整数格点，但如果 activation 的少量极值把 min/max 范围拉得过宽，绝大多数普通值会挤在很窄的整数区间内，实际有效精度远低于 8 bit。这个问题在 transformer activation 中尤其常见，在 SCRN 的 FFB-S 注意力分支中也高度可疑。

当前代码对 activation 使用 `channel_wise=False`，也就是 tensor-wise scale。对于 `[N, C, H, W]` activation，这相当于所有通道和空间位置共享同一套 `delta/zero_point`。如果不同通道尺度差异很大，一个 tensor-wise scale 会让小尺度通道被粗糙量化，大尺度通道又受离群值影响。

后续诊断应至少包括以下指标：

- `absmax / p99`、`absmax / p99.9`：判断少数极值是否主导量化范围。
- `max(per-channel absmax) / median(per-channel absmax)`：判断通道间尺度差异。
- 有效整数格点数：统计量化后实际使用了多少 int levels。
- activation fake-quant MSE：统计原始 activation 与量化反量化 activation 的局部误差。
- 单量化器敏感性：只打开某一个 activation quantizer 时，最终 SNR 掉多少。

解决方向应按侵入性从低到高排列：

1. percentile clipping 或 MSE clipping：先在 tensor-wise 粒度内控制离群值。
2. per-channel activation quantization：需要重新设计 activation scale 的 shape 和广播逻辑，不能直接复用当前 weight channel-wise 路径。
3. group-wise activation quantization：在 per-tensor 和 per-channel 之间折中，减少元数据和 dtype 切换复杂度。
4. branch-wise / structure-wise quantization：按 SCRN 的 FFB-S、FFB-C、CNN branch、Transformer branch 分别建量化粒度。这一点最有 SCRN 结构特色，也最可能成为论文中的方法贡献。
5. SmoothQuant 类思路：针对 attention projection 或离群值严重层，把 activation 难度迁移到 weight 或前后层缩放中。

### 4. 量化插入位置：当前是递归替换 Conv/Linear，不等价于“结构输出量化”

当前 QuantModel 是递归替换 `Conv2d` 和 `Linear`，activation quantizer 位于每个 `QuantModule` 的 Conv/Linear 输出之后。这和直觉中的“对一个 block 或一个结构的输出做量化”不同。对于 SCRN，这个差异很重要：

- FFB 内部有 CNN branch 与 Transformer branch，当前可能在多个内部 Linear/Conv 后都量化，导致误差密度高。
- branch merge、residual/add 之后的边界是否量化，与真实部署边界可能不一致。
- block-level `act_quantizer` 目前不参与 forward，因此现有的 block 激活量化语义并不等同于“block 输出量化”。

后续必须先建立 activation quantizer 位置清单。每个量化器都应映射到结构角色，例如 head、stage1-5、FFB-C CNN branch、FFB-S attention qkv/proj/mlp、tail/output 等。没有这张表，敏感性实验会把不同模块的责任混在一起。

建议的诊断路线：

1. 导出 52 个 activation quantizer 的名称、所属 stage、所属 branch、module type、是否输出层禁用。
2. 统计每个位置的 activation 分布和 fake-quant MSE。
3. 做单点开关实验：只打开一个 activation quantizer 或只关闭一个 activation quantizer，记录最终 SNR 变化。
4. 聚合到结构级别：按 CNN branch、attention branch、MLP、stage、head/tail 统计敏感性。

### 5. `zero_point` 固定：当前只学习 scale，可能不足以重建 asymmetric range

当前 activation reconstruction 只收集 `act_quantizer.delta` 参数，`zero_point` 初始化后固定。对于 asymmetric quantization，动态范围由 `delta` 和 `zero_point` 共同决定：

```text
lower ~= -zero_point * delta
upper ~= (n_levels - 1 - zero_point) * delta
```

只学习 `delta` 等价于只能整体缩放这个区间，而不能平移区间中心。如果 activation 分布偏斜、均值漂移或校准初始化不准，固定 `zero_point` 可能会让优化器用不合理的 scale 去补偿范围错位，甚至把 scale 推向非法区域。

这里确实存在潜在创新空间，但需要谨慎。`zero_point` 本质上应是整数，直接作为连续参数学习会遇到 round 不可导和部署一致性问题。更温和、也更清晰的方案是学习连续的 `lower/upper` 或 `center/range`，再推导出 `delta/zero_point`。这样可以同时解决正 scale、range clipping、zero point 适配和离群值处理。

建议实验顺序：

1. 先诊断当前 `zero_point` 是否合理：比较真实 activation min/max/mean/p99 与可表示区间。
2. 计算理想 min/max 初始化下的 `zero_point` 与当前 `zero_point` 差距。
3. 尝试 reconstruction 后重新根据当前 `delta` 和 activation 分布更新 `zero_point`，作为低侵入性验证。
4. 再考虑学习 `lower/upper` 的新量化器参数化。

### 6. 混合精度：应从“全网 A8”转向“结构敏感性与资源收益联合决策”

用户提出的“木桶效应”很重要。激活量化的目标并不只是让所有 activation 都变成 A8，而是降低推理部署中的峰值显存、带宽或 latency。峰值资源由少数关键位置决定；如果某些位置 A8 极其敏感但资源收益很小，强行量化这些位置是不划算的。

因此后续应把每个结构位置放到二维决策空间里：

- 横轴：量化该位置的资源收益，例如 activation tensor 大小、估计峰值显存贡献、读写带宽或 latency proxy。
- 纵轴：量化该位置的精度损失，例如最终 SNR 下降、SSIM 下降、FP32-output MSE 上升。

理想策略不是“全网 A8”，而是在约束下做选择：

- 在 SNR 下降不超过阈值的前提下最大化 memory/latency 收益。
- 或在 memory/latency 目标下最小化 SNR 损失。

注意当前 PyTorch fake quant 推理仍然以 FP32 计算，不能直接证明真实显存下降。因此实验阶段可以先用 activation tensor volume、bitwidth proxy 和结构生命周期估计收益；真正部署收益需要后续 int8/int4 runtime 或更接近部署的仿真。

### 7. 激活重建目标：局部 MSE 可能无法代表最终 SNR

SCRN 是连续值恢复任务，不是分类任务。分类模型中间 feature 的误差未必改变最终类别排序；SCRN 输出的每个采样点都会进入 SNR/SSIM 计算。因此 activation quantization 对恢复任务更敏感。

当前 activation reconstruction 的目标是局部层/块输出 MSE。这个目标不一定等价于最终恢复质量，因为后续结构可能放大某些局部误差，也可能对某些局部误差不敏感。更重要的是，当前 activation reconstruction 使用 `asym=False`，意味着缓存输入时没有模拟“前序模块已经量化”的误差累积。这可能导致重建目标过于理想化：当前层学会拟合 FP32 输入下的局部输出，但真实全网 W+A 前向时输入已经包含前序 activation quantization 误差。

后续应验证：

1. `asym=True` 是否能改善 activation reconstruction。这里要先确认原 BRECQ 对 activation 阶段为什么常用当前设置，避免盲改。
2. 教师输出是否应从 FP32 换成 W4A32。因为激活量化是在权重已经 W4 的模型上叠加，教师用 W4A32 可能更符合阶段目标。
3. 是否加入最终输出约束，例如局部 MSE + 最终输出 MSE。
4. 是否引入任务相关 loss，例如残差 target、频域 loss 或结构相似性相关项。

这类实验应放在合法性和诊断工具之后做，因为它们更容易引入多变量耦合。

### 8. 校准数据与初始化：`init_batch_size=64` 可能低估深层 activation 范围

当前正式量化使用 `num_samples=1024`，但 activation quantizer 初始化只取 `init_batch_size=64`。如果 `zero_point` 固定不参与重建，那么这 64 个样本对最终 activation range 有长期影响。深层 transformer activation 的极值可能低频出现，64 个样本不足以覆盖。

后续应把“校准样本用于初始化”和“校准样本用于 reconstruction”区分开：

- 初始化阶段：影响初始 `delta/zero_point`，尤其是固定 zero_point。
- 重建阶段：影响 loss 估计和优化方向。

建议先做低成本实验：`init_batch_size=64` vs `256` vs `1024`，同时记录每层 activation range 和最终 SNR。若仅扩大初始化样本就改善 A8 init SNR，说明当前初始化代表性不足。如果最终 SNR 仍差，则问题更多在粒度、插入位置或重建目标。

### 9. 工程约束与实验前置条件

正式修复前需要先补齐若干工程诊断，否则后续实验难以解释：

- activation channel-wise 不能直接沿用 weight channel-wise，需要专门定义 activation channel 维度和广播形状。
- 当前量化器只支持 2 到 8 bit，若要做 A16 或 FP16 fallback，需要扩展 bitwidth 逻辑或把它定义为 skip/fallback 策略，而不是直接设 `n_bits_a=16`。
- activation reconstruction 当前硬编码 `opt_mode="mse"`，Fisher 相关路径没有用于激活阶段。
- `activation_lr=4e-4` 对单个或少量 scale 参数可能偏大，应纳入小范围 sweep。
- block-level activation quantizer 当前不参与 forward，任何“block 输出量化”实验都需要先明确插入点。

### 10. 推荐实验路线

后续实验应按“先诊断、再合法性、再敏感性、再算法”的顺序推进，避免一开始就做长实验或多变量混合修改。

#### E001：Activation diagnostics 工具与基线报告

目标：不改变模型行为，只把问题看清楚。

记录内容：

- 52 个 activation quantizer 的结构位置清单。
- 每个 quantizer 的 `delta/zero_point/inited/disabled` 状态。
- 每层 calibration activation 分布：min/max/p99/p99.9/absmax。
- per-channel absmax 差异。
- fake-quant MSE 和有效 int level 数。
- final W4A8 checkpoint 中非正 delta 层。

验收标准：

- 能明确指出哪些层存在离群值、哪些层有效 level 极低、哪些层量化误差最大。
- 能输出结构分组统计，例如 CNN branch vs Transformer branch。

#### E002：正 scale 约束最小修复

目标：先修掉非法量化器状态，验证负 delta 对最终 SNR 的影响。

候选实现：

- optimizer step 后 clamp `delta >= eps`。
- 或 softplus/log-scale 参数化。

最小实验：

- 小样本 W4A8 smoke，确认不再出现 `delta <= 0`。
- 对比 A8 init、A8 recon 后 SNR。
- 检查是否仍集中在 stage4/stage5 attention proj 掉点。

判断：

- 如果 SNR 明显恢复，负 scale 是主因之一。
- 如果 SNR 仍很差，负 scale 是必须修的合法性问题，但不是主精度瓶颈。

#### E003：初始化覆盖与学习率 sweep

目标：验证当前 activation 初始化和优化稳定性是否不足。

变量：

- `init_batch_size=64/256/1024`
- `activation_lr=4e-4/1e-4/4e-5`

保持其他设置不变。每次都记录 delta 合法性、A8 init SNR、A8 recon SNR 和 offender layers。

#### E004：量化插入位置与敏感性图谱

目标：知道哪些 activation quantizer 最该保留高精度或单独处理。

实验：

- 单点开启或单点关闭 activation quantizer。
- 按 stage/branch/module type 分组开启或关闭。
- 记录最终 SNR/SSIM 变化和结构位置。

输出：

- sensitivity ranking。
- memory/activation-volume proxy ranking。
- sensitivity vs resource benefit 二维图或表。

#### E005：离群值和粒度实验

目标：验证 tensor-wise A8 是否被 outlier 和通道差异主导。

候选方案：

- percentile clipping。
- MSE clipping。
- activation per-channel。
- group-wise。
- FFB branch-wise / structure-wise。
- 对 attention proj 试 SmoothQuant 类缩放。

优先级建议：先 percentile/MSE clipping，再 branch-wise/group-wise，最后考虑更复杂的 SmoothQuant。

#### E006：重建目标与教师模型实验

目标：验证当前 local MSE reconstruction 是否和最终 SNR 脱节。

变量：

- activation reconstruction `asym=False` vs `asym=True`。
- 教师输出 FP32 vs W4A32。
- local MSE vs local MSE + final output MSE。
- 可选 residual/frequency loss。

这组实验应在 E001-E004 后做，因为需要知道关键层和量化插入位置，否则解释成本很高。

### 11. 当前优先级结论

最优先做的不是正式长实验，而是建立诊断与合法性闭环：

1. 先做 activation diagnostics，输出结构位置、分布、level 使用、fake-quant MSE、delta 合法性。
2. 再做正 scale 约束最小修复，确认不会再出现非法 checkpoint。
3. 然后做初始化覆盖和学习率 sweep，排除优化不稳定。
4. 接着做量化器敏感性图谱，为混合精度和结构化粒度提供依据。
5. 最后再进入 clipping、branch-wise/group-wise、zero_point/lower-upper、SmoothQuant、重建目标等算法实验。

如果这些实验能证明“SCRN 的 FFB-S/FFB-C 结构需要 branch-aware activation quantization 或 resource-aware mixed activation precision”，那么这条线有机会成为本工作的创新点，而不仅是把 BRECQ 移植到 SCRN 的工程修复。


## 当前候选问题池

本节用于后续动态维护，不把这些候选项直接视为结论。

| 编号 | 候选问题 | 当前证据 | 优先级 | 状态 |
| --- | --- | --- | --- | --- |
| A1 | Activation `delta` 缺少正值约束，优化后出现非法 scale | E001 smoke 在 final W4A8 checkpoint 复现 2 个负 delta | 最高 | 已复现，待 E002 修复验证 |
| A2 | Transformer/Swin attention 分支存在 activation outlier，tensor-wise A8 被极值主导 | 负 delta 位于 stage4/stage5 attention proj；E001 smoke 的 `absmax_over_p99_max=44.54` | 高 | 已有初步诊断，待正式 E001 64/1024 样本统计 |
| A3 | 只优化 `delta`，`zero_point` 固定导致 asymmetric range 不稳定 | 当前 activation reconstruction 只收集 delta 参数 | 高 | 待验证 |
| A4 | Activation reconstruction 局部 MSE 目标与最终 SNR/SSIM 不一致 | A8 init 后大幅掉点，act recon 只恢复约 0.207 dB | 高 | 待验证 |
| A5 | Calibration 数据覆盖不足或与 eval/部署输入分布不一致 | activation scale 依赖输入分布 | 中 | 待验证 |
| A6 | 全网 A8 一刀切，敏感层需要混合精度或跳过 activation quant | stage4/stage5 attention proj 已暴露异常 | 中 | 待验证 |
| A7 | 激活量化插入位置和 SCRN 实际部署边界不完全一致 | block-level act quantizer 当前不参与 forward | 中 | 待审查 |
| A8 | Activation 初始化只用 `init_batch_size=64`，可能不足以覆盖深层 activation range | `zero_point` 固定，初始化样本对最终范围影响大 | 高 | 待验证 |
| A9 | Activation reconstruction 使用 `asym=False`，可能没有模拟前序量化误差累积 | 代码中 activation 阶段硬编码 `asym=False` | 高 | 待验证 |
| A10 | 教师模型使用 FP32 可能与 W4A8 阶段目标不匹配 | 激活量化叠加在 W4 权重模型之后 | 中 | 待验证 |
| A11 | 当前量化器位置过密，可能在 FFB 内部多次量化造成误差累积 | E001 smoke 已输出 52 个 activation quantizer 的结构位置 | 高 | 已具备诊断工具，待完整统计 |
| A12 | 当前 2-8 bit 限制不支持直接 A16 fallback | `UniformAffineQuantizer` 限制 `2 <= n_bits <= 8` | 中 | 待设计 fallback |
| A13 | Activation LR 可能过大，导致 scale 参数越界或震荡 | `activation_lr=4e-4`，最终出现负 delta | 中 | 待 sweep |

## 实验索引

| 实验编号 | 日期 | 目的 | 主要改动或配置 | 关键结果 | 结论 |
| --- | --- | --- | --- | --- | --- |
| E000 | 2026-05-04 | 建立激活量化研究日志 | 新增本文档 | 不涉及实验 | 后续实验从 E001 开始记录 |
| A000 | 2026-05-04 | 整理用户初步总结并形成实验路线 | 追加 Codex 深度分析 | 不涉及运行 | 后续实验按 E001-E006 推进 |
| P000 | 2026-05-04 | 正式实验前建立目录规范 | 新增激活量化配置目录和运行产物目录 | 不涉及运行 | 后续配置和实验产物分开存放 |
| E001 | 2026-05-04 | 建立 activation diagnostics 工具 | 新增诊断模块、CLI、默认配置和单元测试 | 2-sample smoke 复现 52 个 activation quantizers、2 个负 delta | 可进入正式 64/1024 样本诊断和 E002 正 scale 修复 |
| E001a | 2026-05-04 | 补齐 activation diagnostics 指标 | 增加 per-channel、top-k 和结构分组统计 | 不运行正式 baseline | 工具指标覆盖 E001 验收标准，下一步进入 E001b |

## 实验目录约定

后续激活量化相关配置、诊断和实验产物统一放在现有
`SCRN_BRECQ_app/scrn_brecq/` 内，不在仓库根目录或 `SCRN_BRECQ_app/`
外另建项目。

- `configs/activation_quantization/`: 保存可复现的文本配置，例如 E001 诊断配置、
  E002 正 scale 约束配置、E003 初始化样本数和学习率 sweep 配置。
- `runs/activation_quantization/`: 保存运行产物，例如诊断报告、summary、CSV/JSONL
  统计、图片和临时对比结果。该目录原则上只提交 README，不提交实验输出。

建议按实验编号继续分组：

```text
runs/activation_quantization/
  E001_diagnostics/
  E002_positive_scale/
  E003_init_lr_sweep/
  E004_sensitivity/
  E005_outlier_granularity/
  E006_reconstruction_target/
```

需要继续遵守 Git 规则：不提交 checkpoint、`.npy`、`.pth`、`.pt`、`.ckpt`、
`.segy`、日志、缓存和运行产物。若某次实验需要长期保留结果，应优先把小型
文本摘要写入本日志或单独的 Markdown summary，而不是提交完整 run 目录。

补充约束（2026-05-04）：

- 激活量化实验结果只能写入
  `/home/data1/hanwen/project/Project/SCRN_Quant/SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/`
  及其子目录，不再写入 `/tmp` 或其他项目外路径。
- 仓库根目录 `.gitignore` 已忽略 `runs/activation_quantization/*`，并仅保留
  `runs/activation_quantization/README.md` 可跟踪。
- 后续若需要保存正式诊断结果，应让 CLI 默认输出到上述项目内目录；Git 只记录日志中的
  关键摘要，不提交 run 目录中的具体产物文件。

## 实验记录模板

复制以下模板到“实验详细记录”中使用。

```text
### E编号：实验标题

- 日期：
- 负责人：
- 代码状态：
  - branch：
  - commit：
  - dirty files：
- 实验目的：
- 假设：
- 相关候选问题：
- 代码/配置改动：
- 命令：
- 输入 checkpoint / packed artifact：
- 输出目录：
- 最小检查：
- 关键指标：
  - FP32：
  - W-only pre weight recon：
  - W-only post weight recon：
  - W+A pre act recon：
  - W+A post act recon：
  - packed/checkpoint 对齐：
- Activation quantizer 诊断：
  - delta_count：
  - zero_point_count：
  - delta_min：
  - delta_max：
  - non_positive_delta_count：
  - offender_layers：
- 现象：
- 结论：
- 下一步：
```

## 实验详细记录

### E000：建立激活量化研究日志

- 日期：2026-05-04
- 目的：为 W4A8 激活量化失败问题建立独立研究日志，便于记录潜在原因、修复方案和每次实验。
- 改动：新增 `ACTIVATION_QUANTIZATION_LOG.md`，并要求后续激活量化相关内容同时记录到 `DEVELOPMENT_LOG.md` 和本文档。
- 结论：本文档只建立记录框架，不涉及代码修复或量化实验。

### E001：Activation diagnostics 工具与 smoke

- 日期：2026-05-04
- 负责人：Codex
- 代码状态：
  - branch：`main`
  - commit：提交前记录，目标提交信息为 `Add activation quantization diagnostics`
  - dirty files：E001 诊断源码、CLI、配置、测试、开发日志和激活量化日志
- 实验目的：建立不改变模型行为的激活量化诊断入口，先看清 W4A8 final checkpoint 中 activation quantizer 的合法性、结构位置、分布、fake-quant 局部误差和有效 int level。
- 假设：W4A8 失败至少包含 activation scale 合法性问题；负 `delta` 应能被诊断工具稳定复现，并映射到具体 SCRN 结构位置。
- 相关候选问题：A1、A2、A3、A8、A11、A13。
- 代码/配置改动：
  - 新增 `quant/activation_diagnostics.py`。
  - 新增 `cli/diagnose_activation_quantization.py`。
  - 新增 `configs/activation_quantization/e001_diagnostics.json`。
  - 新增 `tests/test_activation_diagnostics.py`。
  - 更新 `quant/__init__.py`。
- 命令：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.diagnose_activation_quantization --num-samples 2 --batch-size 1 --device cpu --run-name smoke_e001 --run-root /tmp/scrn_brecq_e001_diagnostics`
- 输入 checkpoint / packed artifact：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_194908_w4a8_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq.pth`
- 输出目录：
  - `/tmp/scrn_brecq_e001_diagnostics/20260504_193842_smoke_e001`
  - 输出放在 `/tmp`，避免将 smoke run 产物留在 Git 工作区。
- 最小检查：
  - TDD 红灯：初始单元测试因缺少 `activation_diagnostics` 模块失败。
  - `py_compile` 诊断模块和 CLI。
  - `unittest` 运行 `test_activation_diagnostics`，`Ran 3 tests ... OK`。
  - CLI `--help` 正常输出参数。
- 关键指标：
  - FP32：不涉及。
  - W-only pre weight recon：不涉及。
  - W-only post weight recon：不涉及。
  - W+A pre act recon：不涉及。
  - W+A post act recon：本 smoke 读取 final W4A8 checkpoint。
  - packed/checkpoint 对齐：不涉及。
- Activation quantizer 诊断：
  - delta_count：52
  - zero_point_count：52
  - delta_min：`-0.0031369472853839397`
  - delta_max：`0.043170016258955`
  - non_positive_delta_count：2
  - non_positive_delta_elements：2
  - activation_stat_count：52
  - effective_int_levels_min：17
  - effective_int_levels_max：256
  - absmax_over_p99_max：`44.540536475105085`
  - fake_quant_mse_max：`0.003358484013006091`
  - offender_layers：
    - `model.stage4.0.block.trans_branch.attn.proj`
    - `model.stage5.0.block.trans_branch.attn.proj`
- 现象：
  - E001 smoke 稳定复现 final W4A8 checkpoint 中 52 个 activation quantizer 均已恢复，且 2 个 activation `delta` 为负。
  - 两个负 `delta` 均位于 transformer branch 的 attention projection，进一步支持 FFB-S 深层注意力投影是优先诊断区域。
  - 2 样本 smoke 已观察到较高 outlier ratio，`absmax_over_p99_max=44.54`；该值只是 smoke 信号，正式结论需要 64/1024 样本诊断。
- 结论：
  - E001 工具已能把“负 delta”从手动 checkpoint 检查升级为可复现、可结构定位、可扩展的诊断流程。
  - 当前不应把 smoke 的分布统计当成最终实验结论，但它已经证明 E001 能服务于后续正式诊断和 E002 修复验证。
- 下一步：
  - 用默认 `e001_diagnostics.json` 跑 64 样本正式诊断，必要时再跑 1024 样本诊断。
  - 对比 final checkpoint 与 `quantized_scrn_brecq_pre_act_recon.pth`，确认负 `delta` 只在 activation reconstruction 后出现。
  - 进入 E002：实现正 scale 约束最小修复，并用 E001 工具验证 `non_positive_delta_count` 是否归零。

### E001a：补齐 activation diagnostics 指标

- 日期：2026-05-04
- 负责人：Codex
- 代码状态：
  - branch：`main`
  - commit：提交前记录，目标提交信息为 `Complete activation diagnostics metrics`
  - dirty files：`quant/activation_diagnostics.py`、`tests/test_activation_diagnostics.py`、`DEVELOPMENT_LOG.md`、`ACTIVATION_QUANTIZATION_LOG.md`
- 实验目的：补齐 E001 诊断工具的指标覆盖，使其能输出 per-channel 差异、top-k 问题层和结构分组统计。
- 假设：正式 E001 baseline 前应先保证诊断工具字段完整，否则即使跑出 run 产物，也无法满足“哪些层有离群值、哪些层有效 level 极低、CNN vs Transformer 差异如何”的验收标准。
- 相关候选问题：A2、A6、A11。
- 代码/配置改动：
  - 在 activation stats row 中新增：
    - `per_channel_axis`
    - `per_channel_count`
    - `per_channel_absmax_max`
    - `per_channel_absmax_median`
    - `per_channel_absmax_ratio`
    - `per_channel_absmax_skip_reason`
  - 在 summary 中新增：
    - `top_outlier_layers`
    - `lowest_effective_level_layers`
    - `worst_fake_quant_mse_layers`
    - `worst_relative_mse_layers`
    - `top_per_channel_imbalance_layers`
  - 在 summary 中新增结构分组统计：
    - `branch_summary`
    - `stage_summary`
    - `role_summary`
    - `module_type_summary`
- 命令：
  - 本步骤不运行正式 checkpoint 诊断，不生成正式实验产物。
- 输入 checkpoint / packed artifact：
  - 不涉及。
- 输出目录：
  - 不涉及。
- 最小检查：
  - TDD 红灯：新增测试后，`per_channel_count` 和 `top_outlier_layers` 字段缺失。
  - 单元测试转绿：`conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`。
- 关键指标：
  - FP32：不涉及。
  - W-only pre weight recon：不涉及。
  - W-only post weight recon：不涉及。
  - W+A pre act recon：不涉及。
  - W+A post act recon：不涉及。
  - packed/checkpoint 对齐：不涉及。
- Activation quantizer 诊断：
  - 本步骤只补齐诊断字段，不对真实 checkpoint 产生新的诊断结论。
- 现象：
  - 原 E001 工具第一版只能输出部分 top-k 和 tensor-level outlier 指标，缺少 per-channel 差异和结构分组统计。
  - E001a 后，后续 `summary.json` 和 `activation_stats.jsonl` 将能直接支持 CNN branch vs Transformer branch、stage/role/module type 聚合分析。
- 结论：
  - E001a 是 E001 baseline 的前置工具补齐，不等价于正式 E001 实验。
  - 正式实验结论仍需 E001b 在项目内运行 final W4A8 checkpoint 的 64 样本 baseline 后写入。
- 下一步：
  - E001b：使用默认 `e001_diagnostics.json` 在项目内 run 目录运行 final W4A8 checkpoint 64 样本正式 baseline。
  - E001c：用同一诊断工具对 `quantized_scrn_brecq_pre_act_recon.pth` 做对照。

### E001b：final W4A8 checkpoint 64 样本正式 activation diagnostics baseline

- 日期：2026-05-04
- 负责人：Codex
- 代码状态：
  - branch：`main`
  - commit：提交前记录，目标提交信息为 `Record E001b activation diagnostics baseline`
  - dirty files：`quant/activation_diagnostics.py`、`tests/test_activation_diagnostics.py`、`DEVELOPMENT_LOG.md`、`ACTIVATION_QUANTIZATION_LOG.md`
- 实验目的：在项目内正式运行 final W4A8 checkpoint 的 64-sample activation diagnostics baseline，得到可追溯的结构化诊断结果。
- 假设：如果 W4A8 的主要问题来自 activation reconstruction 后的非法 scale 和局部 activation 分布失配，那么 final checkpoint 的正式 64 样本诊断应稳定复现非正 `delta`，并指出 transformer/attention 相关层的量化劣化信号。
- 相关候选问题：A1、A2、A3、A6、A8、A11、A13。
- 代码/配置改动：
  - 首次运行 E001b 时，`torch.quantile` 在 SCRN head 大 activation tensor 上报错：`RuntimeError: quantile() input tensor is too large`。
  - 根因：64 个 `128x128` calibration 输入经过 `dim=64` 的 SCRN head 后，单层 activation 约 67M 元素，超过当前 PyTorch `torch.quantile` 的限制。
  - 修复：`quant/activation_diagnostics.py` 对元素数超过 16,000,000 的 tensor 使用 `torch.kthvalue` fallback 计算分位数；小 tensor 保持原 `torch.quantile` 路径。
  - 测试：`tests/test_activation_diagnostics.py` 新增大 tensor quantile 回归测试。
- 命令：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.diagnose_activation_quantization --config SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/e001_diagnostics.json`
- 输入 checkpoint / packed artifact：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_194908_w4a8_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq.pth`
- calibration 数据：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches`
  - num_samples：64
  - batch_size：16
  - seed：1005
  - device：CPU
- 输出目录：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E001_diagnostics/20260504_203753_e001_diagnostics`
  - `config.json`
  - `summary.json`
  - `summary.md`
  - `quantizers.csv`
  - `activation_stats.jsonl`
  - `offender_layers.json`
- Git 产物规则：
  - run 产物位于项目内，不在 `/tmp`。
  - `git check-ignore -v` 确认 `summary.json` 被 `.gitignore` 的 `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/*` 规则忽略。
- 最小检查：
  - TDD 红灯：新增大 tensor quantile 测试后，单元测试失败于 `RuntimeError: quantile() input tensor is too large`。
  - 单元测试转绿：`conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`，`Ran 6 tests ... OK`。
  - 正式诊断完成：`activation_quantizers=52 non_positive_delta_count=2 activation_stat_count=52 fake_quant_mse_max=0.003323915181681514`。
- 关键指标：
  - FP32：不涉及。
  - W-only pre weight recon：不涉及。
  - W-only post weight recon：不涉及。
  - W+A pre act recon：不涉及，本步骤只读取 final W4A8。
  - W+A post act recon：读取 final W4A8 checkpoint。
  - packed/checkpoint 对齐：不涉及。
- Activation quantizer 总览：
  - activation_quantizers：52
  - activation_delta_count：52
  - activation_zero_point_count：52
  - non_positive_delta_count：2
  - activation_stat_count：52
  - fake_quant_mse_max：`0.003323915181681514`
  - fake_quant_mse_mean：`0.00010568322308295127`
  - effective_int_levels_min：17
  - effective_int_levels_max：256
  - absmax_over_p99_max：`34.627632811323394`
- 非正 delta offender layers：
  - `model.stage4.0.block.trans_branch.attn.proj`
    - stage：`stage4`
    - branch：`transformer`
    - role：`attention_proj`
    - index：37
    - delta：`-0.0031369472853839397`
  - `model.stage5.0.block.trans_branch.attn.proj`
    - stage：`stage5`
    - branch：`transformer`
    - role：`attention_proj`
    - index：47
    - delta：`-0.002051445422694087`
- Top outlier layers，按 `absmax_over_p99`：
  - `model.head`：`34.627632811323394`
  - `model.tail`：`24.92000569877867`
  - `model.stage2.0.block.trans_branch.mlp.0`：`4.598540554106423`
  - `model.stage2.0.block.conv_branch.6`：`4.571042050421725`
  - `model.stage5.0.block.conv_branch.6`：`4.252737760437427`
- Lowest effective int level layers：
  - `model.stage2.0.block.trans_branch.attn.proj`：17
  - `model.stage4.0.block.trans_branch.attn.proj`：23
  - `model.stage5.0.block.trans_branch.attn.proj`：38
  - `model.stage3.0.block.trans_branch.attn.proj`：49
  - `model.stage5.0.block.trans_branch.mlp.0`：143
- Worst fake-quant MSE layers：
  - `model.stage1.0.block.trans_branch.attn.qkv`：`0.003323915181681514`
  - `model.stage4.0.block.trans_branch.attn.qkv`：`0.00038254939136095345`
  - `model.stage2.0.block.conv_branch.6`：`0.0001195866207126528`
  - `model.stage3.0.block.conv_branch.3`：`0.00009796844096854329`
  - `model.stage5.0.block.trans_branch.attn.qkv`：`0.00009043353929882869`
- Worst relative MSE layers：
  - `model.stage4.0.block.trans_branch.attn.qkv`：`0.020646367816721696`
  - `model.tail`：`0.014584982809199652`
  - `model.head`：`0.012885225747812064`
  - `model.stage2.0.block.trans_branch.attn.proj`：`0.007135294640276089`
  - `model.stage4.0.block.trans_branch.attn.proj`：`0.005390653549703427`
- Top per-channel imbalance layers：
  - `model.stage5.0.block.conv_branch.6`：`7.788059729177248`
  - `model.stage2.0.block.conv_branch.6`：`3.3941267855532167`
  - `model.stage1.0.block.trans_branch.mlp.2`：`3.219494744568011`
  - `model.stage1.0.block.split_proj`：`2.803273019473079`
  - `model.stage2.1`：`2.573026767960425`
- Branch summary：
  - `transformer`
    - count：20
    - effective_int_levels_min：17
    - fake_quant_mse_max：`0.003323915181681514`
    - fake_quant_relative_mse_max：`0.020646367816721696`
    - absmax_over_p99_max：`4.598540554106423`
  - `cnn`
    - count：15
    - effective_int_levels_min：231
    - fake_quant_mse_max：`0.0001195866207126528`
    - fake_quant_relative_mse_max：`0.0003560360421608271`
    - absmax_over_p99_max：`4.571042050421725`
  - `head`
    - count：1
    - absmax_over_p99_max：`34.627632811323394`
    - fake_quant_relative_mse_max：`0.012885225747812064`
  - `tail`
    - count：1
    - absmax_over_p99_max：`24.92000569877867`
    - fake_quant_relative_mse_max：`0.014584982809199652`
- Module type summary：
  - `Linear`
    - count：20
    - effective_int_levels_min：17
    - fake_quant_mse_max：`0.003323915181681514`
    - fake_quant_relative_mse_max：`0.020646367816721696`
  - `Conv2d`
    - count：32
    - effective_int_levels_min：188
    - fake_quant_mse_max：`0.0001195866207126528`
    - fake_quant_relative_mse_max：`0.014584982809199652`
- 现象：
  - final W4A8 checkpoint 在 64 样本正式诊断下稳定复现 2 个非法 activation `delta`，位置与早期手动检查和 2-sample smoke 一致。
  - 非正 delta 层均是深层 transformer attention projection，但最低有效 int level 还包括 stage2/stage3/stage4/stage5 的 attention projection，说明问题不只局限于两个负 scale 层。
  - transformer/Linear 的局部量化误差和有效 int level 明显更差：`Linear` effective level min 为 17，而 `Conv2d` min 为 188；`transformer` relative MSE max 为 `0.020646367816721696`，远高于 `cnn` 的 `0.0003560360421608271`。
  - head/tail 的 outlier ratio 很高，但 fake-quant absolute MSE 很小，说明它们更像分布离群诊断信号，不是当前最大局部 MSE 来源。
  - per-channel imbalance 最强的层来自 CNN conv branch，尤其是 stage5/stage2 的 `conv_branch.6`，后续如果只处理 transformer 仍可能遗漏 channel 不均衡问题。
- 结论：
  - E001b 正式 baseline 已完成，且在项目内形成有效实验产物。
  - 当前最确定的首要修复方向仍是 activation `delta` 正值约束，因为 final checkpoint 存在明确非法 scale。
  - 第二优先级是 attention projection 的有效 level 崩塌和 transformer/Linear 局部误差；这可能需要对称/非对称策略、per-channel/per-token 策略或 reconstruction 参数进行后续对照实验。
  - 第三优先级是 head/tail outlier 与 CNN per-channel imbalance，它们更适合作为 E002 修复后的残余误差分析方向。
- 下一步：
  - E001c：用同一诊断工具运行 `quantized_scrn_brecq_pre_act_recon.pth` 对照，确认负 `delta` 是否只由 activation reconstruction 引入。
  - E002：实现 activation scale 正值约束或优化后 clamp 的最小修复，并用 E001 工具验证 `non_positive_delta_count` 是否归零。
  - E003/E004：围绕 attention projection 的有效 int level 过低，继续设计 clipping、asym、per-channel 或重建目标对照实验。

### E001c：pre-act-recon checkpoint 64 样本 activation diagnostics 对照

- 日期：2026-05-04
- 负责人：Codex
- 代码状态：
  - branch：`main`
  - commit：提交前记录，目标提交信息为 `Record E001c pre-act-recon diagnostics baseline`
  - dirty files：`DEVELOPMENT_LOG.md`、`ACTIVATION_QUANTIZATION_LOG.md`
- 实验目的：对 activation reconstruction 前的 W4A8 checkpoint 运行同一套 E001 诊断，与 E001b final checkpoint 对照，判断非法 `delta` 和 transformer/Linear 指标恶化是否由 activation reconstruction 引入。
- 假设：如果 `pre_act_recon` checkpoint 中 activation `delta` 全为正，而 final checkpoint 出现负 `delta`，则 activation reconstruction 优化过程缺少正 scale 约束是确定问题之一。
- 相关候选问题：A1、A2、A3、A8、A11、A13。
- 代码/配置改动：
  - 不修改量化算法。
  - 不修改诊断工具代码。
  - 只运行诊断并更新日志。
- 命令：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.diagnose_activation_quantization --config SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/e001_diagnostics.json --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_194908_w4a8_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq_pre_act_recon.pth --run-name e001c_pre_act_recon`
- 输入 checkpoint / packed artifact：
  - E001c pre-act-recon：
    - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_194908_w4a8_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
  - 对照 E001b final：
    - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_194908_w4a8_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq.pth`
- calibration 数据：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches`
  - num_samples：64
  - batch_size：16
  - seed：1005
  - device：CPU
- 输出目录：
  - E001c：
    - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E001_diagnostics/20260504_205237_e001c_pre_act_recon`
  - E001b 对照：
    - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E001_diagnostics/20260504_203753_e001_diagnostics`
- Git 产物规则：
  - E001c run 产物位于项目内，不在 `/tmp`。
  - `git check-ignore -v` 确认 `summary.json` 被 `.gitignore` 的 `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/*` 规则忽略。
- 最小检查：
  - E001c 正式诊断完成：`activation_quantizers=52 non_positive_delta_count=0 activation_stat_count=52 fake_quant_mse_max=0.00011542496213223785`。
- 关键指标：
  - FP32：不涉及。
  - W-only pre weight recon：不涉及。
  - W-only post weight recon：不涉及。
  - W+A pre act recon：读取 `quantized_scrn_brecq_pre_act_recon.pth`。
  - W+A post act recon：对照 E001b final checkpoint。
  - packed/checkpoint 对齐：不涉及。
- Activation quantizer 总览对比：
  - E001b final：
    - activation_quantizers：52
    - activation_delta_count：52
    - activation_zero_point_count：52
    - non_positive_delta_count：2
    - activation_stat_count：52
    - fake_quant_mse_max：`0.003323915181681514`
    - fake_quant_mse_mean：`0.00010568322308295127`
    - effective_int_levels_min：17
    - effective_int_levels_max：256
    - absmax_over_p99_max：`34.627632811323394`
  - E001c pre-act-recon：
    - activation_quantizers：52
    - activation_delta_count：52
    - activation_zero_point_count：52
    - non_positive_delta_count：0
    - activation_stat_count：52
    - fake_quant_mse_max：`0.00011542496213223785`
    - fake_quant_mse_mean：`0.000033345033807013605`
    - effective_int_levels_min：188
    - effective_int_levels_max：256
    - absmax_over_p99_max：`34.627632811323394`
- 非正 delta offender 对比：
  - E001b final offender：
    - `model.stage4.0.block.trans_branch.attn.proj`
      - delta：`-0.0031369472853839397`
    - `model.stage5.0.block.trans_branch.attn.proj`
      - delta：`-0.002051445422694087`
  - E001c pre-act-recon offender：
    - 无。
- Top outlier layers：
  - E001c 与 E001b 一致，说明该指标主要由 activation tensor 分布决定，不由 activation quantizer scale 优化改变。
  - 前五层：
    - `model.head`：`34.627632811323394`
    - `model.tail`：`24.92000569877867`
    - `model.stage2.0.block.trans_branch.mlp.0`：`4.598540554106423`
    - `model.stage2.0.block.conv_branch.6`：`4.571042050421725`
    - `model.stage5.0.block.conv_branch.6`：`4.252737760437427`
- Lowest effective int level 对比：
  - E001b final：
    - `model.stage2.0.block.trans_branch.attn.proj`：17
    - `model.stage4.0.block.trans_branch.attn.proj`：23
    - `model.stage5.0.block.trans_branch.attn.proj`：38
    - `model.stage3.0.block.trans_branch.attn.proj`：49
    - `model.stage5.0.block.trans_branch.mlp.0`：143
  - E001c pre-act-recon：
    - `model.tail`：188
    - `model.stage5.0.block.trans_branch.mlp.0`：202
    - `model.stage2.0.block.split_proj`：209
    - `model.stage4.0.block.trans_branch.mlp.0`：220
    - `model.stage2.0.block.trans_branch.mlp.0`：224
- Worst fake-quant MSE 对比：
  - E001b final 最大层：
    - `model.stage1.0.block.trans_branch.attn.qkv`：`0.003323915181681514`
  - E001c pre-act-recon 最大层：
    - `model.stage2.0.block.conv_branch.6`：`0.00011542496213223785`
  - 解释：activation reconstruction 后，最大局部 fake-quant MSE 从 CNN conv 层转移到 transformer attention qkv，且最大值显著放大。
- Worst relative MSE 对比：
  - E001b final 最大层：
    - `model.stage4.0.block.trans_branch.attn.qkv`：`0.020646367816721696`
  - E001c pre-act-recon 最大层：
    - `model.tail`：`0.014584982809199652`
  - 但 E001c 的 transformer/Linear relative MSE max 仅为 `0.00021580944014857998`，远低于 E001b final 的 `0.020646367816721696`。
- Branch summary 对比：
  - `transformer`
    - E001b final：effective_int_levels_min `17`，fake_quant_mse_max `0.003323915181681514`，relative_mse_max `0.020646367816721696`
    - E001c pre-act-recon：effective_int_levels_min `202`，fake_quant_mse_max `0.00005239322854322381`，relative_mse_max `0.00021580944014857998`
  - `cnn`
    - E001b final：effective_int_levels_min `231`，fake_quant_mse_max `0.0001195866207126528`，relative_mse_max `0.0003560360421608271`
    - E001c pre-act-recon：effective_int_levels_min `231`，fake_quant_mse_max `0.00011542496213223785`，relative_mse_max `0.00034364585636106383`
  - `head` 和 `tail`
    - 两次诊断的 outlier ratio 与 relative MSE 基本一致，说明它们不是 activation reconstruction 引入负 scale 的主路径。
- Role summary 对比：
  - `attention_proj`
    - E001b final：effective_int_levels_min `17`，relative_mse_max `0.007135294640276089`
    - E001c pre-act-recon：effective_int_levels_min `247`，relative_mse_max `0.00003250767545649528`
  - `attention_qkv`
    - E001b final：fake_quant_mse_max `0.003323915181681514`，relative_mse_max `0.020646367816721696`
    - E001c pre-act-recon：fake_quant_mse_max `0.00005239322854322381`，relative_mse_max `0.0000716945385857808`
- Module type summary 对比：
  - `Linear`
    - E001b final：effective_int_levels_min `17`，fake_quant_mse_max `0.003323915181681514`，relative_mse_max `0.020646367816721696`
    - E001c pre-act-recon：effective_int_levels_min `202`，fake_quant_mse_max `0.00005239322854322381`，relative_mse_max `0.00021580944014857998`
  - `Conv2d`
    - E001b final：effective_int_levels_min `188`，fake_quant_mse_max `0.0001195866207126528`，relative_mse_max `0.014584982809199652`
    - E001c pre-act-recon：effective_int_levels_min `188`，fake_quant_mse_max `0.00011542496213223785`，relative_mse_max `0.014584982809199652`
- 现象：
  - pre-act-recon checkpoint 没有任何非正 activation `delta`，final checkpoint 有 2 个负 `delta`。
  - E001c 的 activation tensor outlier 排名与 E001b 一致，说明 outlier 分布本身没有因为 checkpoint 选择改变；改变的是 activation quantizer 参数及其 fake-quant 行为。
  - activation reconstruction 后，attention projection 的有效 int level 从 pre-act-recon 的最低 247 降到 final 的最低 17。
  - activation reconstruction 后，Linear/transformer 的最坏相对误差从 `0.00021580944014857998` 增加到 `0.020646367816721696`，而 Conv2d/CNN 基本不变。
- 结论：
  - E001c 强化了 E001b 的判断：非法 activation `delta` 是 activation reconstruction 阶段引入的，不是 activation quantizer 初始化阶段固有问题。
  - 当前 W4A8 激活量化失败的第一优先级修复应是 activation reconstruction 中对 `delta` 的正值约束，而不是先更换校准集或直接调 packed deployment。
  - attention projection 和 attention qkv 是 activation reconstruction 后劣化最明显的结构位置；E002 修复正 scale 后，必须继续用 E001 指标确认 transformer/Linear 的 effective level 与 relative MSE 是否恢复。
  - head/tail outlier 与 CNN per-channel imbalance 是残余诊断线索，但不是解释“final checkpoint 负 delta”的主因。
- 下一步：
  - E002：实现 activation reconstruction 的正 scale 约束最小修复，并复跑 E001 final 诊断检查 `non_positive_delta_count=0`。
  - E002 后继续比较 attention projection/qkv 的 effective level 和 relative MSE，确认修复是否只消除非法 scale，还是同步改善 W4A8 输出质量。

### E001d：E001 诊断阶段综合结论

- 日期：2026-05-04
- 负责人：Codex
- 代码状态：
  - branch：`main`
  - commit：提交前记录，目标提交信息为 `Summarize E001 activation diagnostics conclusions`
  - dirty files：`DEVELOPMENT_LOG.md`、`ACTIVATION_QUANTIZATION_LOG.md`
- 实验目的：不再运行新实验，而是把 E001 smoke、E001a、E001b 和 E001c 的证据链整理成可读结论，为 E002 修复提供明确入口。
- 相关候选问题：A1、A2、A3、A8、A11、A13。
- 代码/配置改动：
  - 不修改量化算法。
  - 不修改诊断工具。
  - 不创建新的 tracked 文档。
  - 只在激活量化日志和开发日志中沉淀 E001 总结。
- 命令：
  - 不运行新 checkpoint 诊断。
- 输入 checkpoint / packed artifact：
  - E001b final：
    - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_194908_w4a8_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq.pth`
  - E001c pre-act-recon：
    - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_194908_w4a8_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- 输出目录：
  - 不生成新 run。
  - 本总结引用已有 E001b/E001c 产物：
    - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E001_diagnostics/20260504_203753_e001_diagnostics`
    - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E001_diagnostics/20260504_205237_e001c_pre_act_recon`
- 最小检查：
  - 本步骤只更新 Markdown 日志，提交前使用 `git diff --check` 和 `git diff --check --cached` 检查。

#### 一句话结论

当前 W4A8 激活量化失败的首要确定问题是：activation reconstruction 把部分 activation quantizer 的 `delta` 优化成了非法负 scale，并且这个过程伴随 transformer/Linear 层有效 int level 崩塌和局部 fake-quant 误差显著放大。

#### 证据链

- E001 工具层证据：
  - 已能恢复并扫描 52 个 activation quantizer。
  - 已能输出 `delta/zero_point/inited/disabled`、activation 分布、fake-quant MSE、relative MSE、effective int level、per-channel imbalance 和结构分组统计。
  - 诊断不改模型行为，只读取 checkpoint 和 calibration forward hook 结果。
- E001b final W4A8 证据：
  - `activation_quantizers=52`
  - `non_positive_delta_count=2`
  - `effective_int_levels_min=17`
  - `fake_quant_mse_max=0.003323915181681514`
  - transformer relative MSE max：`0.020646367816721696`
  - attention qkv MSE max：`0.003323915181681514`
  - 两个非法 `delta` 层：
    - `model.stage4.0.block.trans_branch.attn.proj`
    - `model.stage5.0.block.trans_branch.attn.proj`
- E001c pre-act-recon 对照证据：
  - `non_positive_delta_count=0`
  - `effective_int_levels_min=188`
  - transformer effective level min：`202`
  - transformer relative MSE max：`0.00021580944014857998`
  - attention projection effective level min：`247`
- 对照结论：
  - 负 `delta` 在 pre-act-recon 中不存在，在 final 中出现，因此不是 activation quantizer 初始化阶段固有问题。
  - activation reconstruction 后 transformer effective level min 从 202 降到 17。
  - activation reconstruction 后 transformer relative MSE max 从 `0.00021580944014857998` 升到 `0.020646367816721696`。
  - attention projection 的有效 int level 从 pre-act-recon 最低 247 降到 final 最低 17，说明 attention projection 是最直接的异常结构位置。

#### 可以弱化或暂时排除的解释

- activation quantizer state 丢失不是当前主因：
  - E001b/E001c 都恢复了 52 个 activation quantizer、52 个 delta 和 52 个 zero point。
  - 问题不是“没有加载 activation quantizer”，而是 final checkpoint 中部分 loaded `delta` 本身非法。
- packed restore 不是当前 E001 主因：
  - E001 直接诊断 checkpoint，不依赖 packed deployment。
  - 负 `delta` 在 checkpoint 层面已经存在，packed 路径即使正确恢复也会继承这个非法状态。
- activation outlier 是背景风险，但不是“负 delta 出现”的直接证据：
  - E001b 和 E001c 的 top outlier layers 排名一致，说明 outlier 分布主要来自模型 activation 本身。
  - final/pre 的 outlier 排名一致，但只有 final 出现负 `delta`，因此负 scale 更直接指向 activation reconstruction 的 `delta` 优化约束问题。
- CNN/Conv2d 不是当前首要劣化路径：
  - E001b final 的 CNN effective level min 为 231，E001c pre-act-recon 也是 231。
  - CNN relative MSE max 从 `0.00034364585636106383` 到 `0.0003560360421608271`，变化很小。
  - per-channel imbalance 在 CNN 中存在，但它更像 E002 后的残余误差分析方向，而不是解释 final 负 `delta` 的主因。

#### 结构定位

- 最确定的非法 scale 层：
  - `model.stage4.0.block.trans_branch.attn.proj`
  - `model.stage5.0.block.trans_branch.attn.proj`
- 最需要后续跟踪的结构组：
  - `attention_proj`：
    - effective level min 从 pre-act-recon 的 247 降到 final 的 17。
  - `attention_qkv`：
    - final 中 fake-quant MSE max 达到 `0.003323915181681514`。
  - `Linear` / `transformer`：
    - Linear effective level min 从 202 降到 17。
    - Linear relative MSE max 从 `0.00021580944014857998` 升到 `0.020646367816721696`。
- 需要继续观察但不是第一修复对象的结构：
  - `head` / `tail`：
    - outlier ratio 高，但 final/pre 一致，更像固定 activation 分布风险。
  - CNN `conv_branch.6`：
    - per-channel imbalance 高，后续可能需要 channel-wise 或 clipping 分析，但不是负 scale 的直接来源。

#### E002 修复验收方向

- 第一验收指标：正 scale 合法性。
  - activation reconstruction 后重新保存 final checkpoint。
  - 用 E001 诊断复核 `non_positive_delta_count=0`。
  - offender layers 应为空。
- 第二验收指标：transformer/Linear 是否恢复到接近 pre-act-recon。
  - transformer effective level min 不应再接近 17。
  - Linear effective level min 不应再接近 17。
  - attention projection effective level min 应显著高于 final 的 17。
  - transformer/Linear relative MSE max 应显著低于 final 的 `0.020646367816721696`。
- 第三验收指标：输出质量。
  - 在修正正 scale 后，重新评估 W4A8 推理质量。
  - 如果 `non_positive_delta_count=0` 但 dB 仍低，再进入 clipping、asym、per-channel/per-token 或 reconstruction loss 方向。
- E002 的最小优先级：
  - 不先改 packed deployment。
  - 不先改 calibration 数据。
  - 不先扩大正式长实验。
  - 先修 activation reconstruction 中 `delta` 的正值约束，并用 E001 诊断闭环验证。

### E001e：checkpoint 固定指标与 calibration 样本统计指标的区别

- 日期：2026-05-04
- 负责人：Codex
- 记录目的：澄清此前对 E001 诊断指标的一个误解，即“做完一次 W4A8 量化后，每层 activation 分布、fake-quant MSE、effective int level 和 top-k 问题层是否已经完全固定在 checkpoint 文件里”。
- 结论：W4A8 checkpoint 固定的是模型参数和量化器参数，但不固定所有可能输入下的 activation 分布；activation 分布和基于 activation 计算出的误差指标必须依赖具体输入样本统计。

#### checkpoint 中已经固定的内容

- 模型权重。
- weight quantizer 状态。
- activation quantizer 的 `delta`。
- activation quantizer 的 `zero_point`。
- activation quantizer 的 `inited/disabled` 等状态。
- final quant state。

因此，以下诊断结论不依赖 calibration 样本数：

- activation quantizer 总数。
- 每个 quantizer 的结构位置。
- 每个 quantizer 的 `delta/zero_point/inited/disabled`。
- `delta` 是否为负。
- final W4A8 checkpoint 中哪些层存在非正 `delta`。

这也是为什么 E001b 中 `non_positive_delta_count=2` 是 checkpoint 本身的硬错误；即使用 1 个样本、64 个样本、1024 个样本，甚至只读 checkpoint 不跑 forward，也能发现该问题。

#### checkpoint 中没有直接固定的内容

以下指标不是 checkpoint 文件里保存好的固定表格，而是把输入样本送进模型后，在每层 hook 到 activation 再统计得到：

- 每层 activation 分布：`min/max/p99/p99.9/absmax`。
- `absmax/p99` 和 `absmax/p99.9`。
- per-channel absmax 差异。
- fake-quant MSE / MAE / relative MSE。
- effective int level。
- top outlier / worst fake-quant MSE / worst relative MSE 排名。

形式上可以理解为：

```text
activation = layer_output(input_sample, checkpoint_parameters)
```

checkpoint 固定了 `checkpoint_parameters`，但不同 `input_sample` 会触发不同 activation。因此，activation 分布和由 activation 推导出的 fake-quant MSE、effective int level、top-k 排名都和所选 calibration 样本集合有关。

#### 64 / 1024 / full calibration 的区别

- 64-sample：
  - 适合快速诊断和定位明显问题。
  - 足够发现 checkpoint 参数级硬错误，例如 final checkpoint 的负 `delta`。
  - 对极端 outlier、p99.9、top-k 排名的稳定性有限。
- 1024-sample：
  - activation 分布、fake-quant MSE、effective int level 和 top-k 排名更稳定。
  - 更接近原量化 calibration 范围。
  - 适合在 E002 修复后验证趋势是否稳定。
- full calibration：
  - 统计最完整，适合最终报告或论文级补充验证。
  - 成本最高，不适合作为发现非法 scale 后的第一优先级。

#### 对 E001 结论的影响

- 样本数会影响：
  - activation 分布数值。
  - outlier ratio。
  - per-channel imbalance 统计。
  - fake-quant MSE / relative MSE。
  - effective int level。
  - top outlier 和 worst MSE 的具体排序。
- 样本数不会改变：
  - final checkpoint 是否包含负 `delta`。
  - pre-act-recon checkpoint 是否包含负 `delta`。
  - final 有 2 个负 `delta` 而 pre-act-recon 为 0 这一对照事实。
  - 负 `delta` 是 activation reconstruction 后引入这一核心判断。

因此，E001b/E001c 的 64-sample 诊断足够支持 E002 的方向：先修 activation reconstruction 中的 `delta` 正值约束。1024-sample 或 full calibration 更适合放在 E002 修复后，用来验证修复趋势是否稳定，而不是在已经发现 checkpoint 参数非法时继续扩大诊断。

### E002：正 scale 约束最小修复初步计划

- 日期：2026-05-04
- 负责人：Codex
- 阶段定位：E002 是修复阶段的第一步，但仍然保持最小变量控制；先修掉 activation quantizer 的非法状态，再判断负 `delta` 对最终 W4A8 SNR 的影响。

#### 核心判断

直接增加正 scale 约束是必要的，但不一定充分。

- 必要性：
  - activation `delta` 是量化 scale，数学上必须为正。
  - E001b 已经证明 final W4A8 checkpoint 中存在 2 个负 `delta`。
  - E001c 已经证明 pre-act-recon checkpoint 中 `non_positive_delta_count=0`，因此非法 scale 是 activation reconstruction 后引入。
  - 只要 checkpoint 中存在负 scale，后续 fake-quant 行为、effective int level 和输出质量评估都混入了非法量化器状态。
- 不充分性：
  - 如果 optimizer 会把 `delta` 推到负数，说明当前 reconstruction loss、学习率、activation 分布、层结构或参数化方式之间可能存在更深的优化冲突。
  - 正约束只能阻止 scale 进入非法区域；如果优化器持续把某些层推向 0 附近，量化器仍可能出现有效 int level 崩塌。
  - 因此 E002 不能只看 `non_positive_delta_count=0`，还必须继续观察 transformer/Linear 的 effective level、relative MSE 和最终 SNR。

#### E002a：post-step clamp 最小修复

目标：只在 activation reconstruction 的优化步骤后，对 learnable activation `delta` 做正值投影，不改变量化公式、不改 packed deployment、不改 calibration 数据。

计划内容：

- 在 layer/block activation reconstruction 的 `optimizer.step()` 后，对收集到的 activation delta 执行 `clamp_min_(eps)`。
- `eps` 使用一个很小的正数，只用于保证合法性，不主动改变正常正 scale。
- 添加最小测试，验证 activation reconstruction 优化后不能留下非正 `delta`。
- 记录 clamp 前后的最小 `delta`，后续如有必要再增加 clamp hit count 或 delta trajectory 诊断。

验收标准：

- 修复后的 activation reconstruction 产物用 E001 工具诊断时，`non_positive_delta_count=0`。
- offender layers 为空。
- 如果最终 W4A8 SNR 明显恢复，说明负 scale 是主因之一。
- 如果最终 W4A8 SNR 仍很差，说明负 scale 是必须修的合法性问题，但不是唯一主瓶颈。

#### E002b：修复后 W4A8 复现实验

目标：用与失败 run 尽量一致的设置重新生成 W4A8 checkpoint，单独评估“正 scale 约束”这一变量的影响。

计划内容：

- 先跑小规模 smoke，确认新 checkpoint 不再出现非正 activation `delta`。
- smoke 通过后，再考虑复跑与当前失败 run 对齐的 W4A8 activation reconstruction。
- 所有实验产物只能写入项目目录内的 `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002_positive_scale/` 或对应 quant run 目录，不写 `/tmp`。

重点对比：

- final W4A8 SNR。
- `non_positive_delta_count`。
- transformer/Linear effective int level min。
- attention projection/qkv 的 fake-quant MSE 和 relative MSE。
- 是否还有层被 clamp 到接近 `eps`。

#### E002c：如果 clamp 后仍然不好

如果正 scale clamp 消除了非法状态但 SNR 仍明显偏低，下一步不应继续盲目扩大样本，而应定位是否存在更深优化问题。

候选方向：

- 限制 `delta` 相对初始化值的变化幅度，例如 `min_ratio * init_delta <= delta <= max_ratio * init_delta`。
- 对 `log(delta / init_delta)` 增加正则，防止 scale 被优化到极端范围。
- 降低 `activation_lr` 或缩短 `iters_a`，判断当前学习率/迭代数是否过激。
- 改为 log-scale 或 `softplus(raw_delta) + eps` 参数化，从参数空间上保证 scale 为正。
- 对 attention projection/qkv 做选择性冻结或单独策略，判断 transformer/Linear 是否是有害重构路径。
- 重新评估 `asym=False` 的 activation reconstruction 设定是否和真实量化输入分布不一致。

#### E002 的边界

- E002 第一阶段不修改 `SCRN-main/` 和 `BRECQ-main/`。
- E002 第一阶段不先做 packed deployment 修复。
- E002 第一阶段不先扩大到 full calibration。
- E002 第一阶段不把 post-step clamp 直接视为最终算法创新，只把它作为合法性修复和因果验证工具。
- 每次代码或实验变更都必须同步记录在 `DEVELOPMENT_LOG.md` 和本日志中，并按 Git 工作流提交。

### E002a：activation delta post-step clamp 最小修复

- 日期：2026-05-04
- 负责人：Codex
- 实验/开发性质：代码最小修复，不运行正式 W4A8 reconstruction，不生成新 checkpoint。

#### 修改内容

- 在 activation reconstruction 的优化步骤后，对 learnable activation `delta` 执行正值投影。
- 新增最小下界 `ACTIVATION_DELTA_MIN = 1e-8`，与当前 quantizer 初始化中的小 scale 下界保持一致。
- 新增 `_project_activation_delta_params_positive(...)`，只对传入的 activation delta 参数做 `clamp_(min=eps)`。
- 在 block reconstruction 和 layer reconstruction 的 `act_quant=True` 路径中调用该投影。
- `act_quant=False` 的 AdaRound 权重量化路径不受影响。
- 不修改 `UniformAffineQuantizer.forward()`，因此历史 checkpoint 加载和普通推理路径的语义不被额外改变。

#### 设计判断

- E002a 只解决 activation scale 合法性问题：activation reconstruction 后不应留下非正 `delta`。
- 该修复不声称解决所有 W4A8 精度损失；如果 optimizer 仍持续把某些层推向极小 scale，后续还需要 E002c 的 delta ratio 限制、log-scale/softplus 参数化或 attention 层策略。
- 选择 post-step clamp 而不是立即重构 quantizer 参数化，是为了保持最小改动和 checkpoint 兼容性，先验证负 scale 对最终 SNR 的影响。

#### 验证计划

- 新增 `test_activation_scale_constraints.py`，直接验证：
  - 负数和 0 的 activation `delta` 会被 clamp 到正数。
  - 已经大于 `eps` 的正 `delta` 不被改变。
  - `eps <= 0` 会报错。
- 本步骤完成后只运行单元测试、诊断测试和 py_compile。
- E002b 再重新生成 W4A8 checkpoint，并用 E001 诊断验证 `non_positive_delta_count=0` 和最终 SNR 是否恢复。

### E002b：positive-scale W4A8 复现实验

- 日期：2026-05-04
- 负责人：Codex
- 实验目的：在 E002a post-step clamp 已合入后，重新生成 W4A8 checkpoint，验证正 scale 合法性修复是否能显著恢复最终 SNR。
- 实验边界：不修改代码；不切换学习率、样本数或 reconstruction loss；只改变当前代码中已经存在的 activation `delta` 正值投影。

#### 运行记录

- smoke quant run：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260504_221053_e002b_smoke_positive_scale`
  - 命令：W4A8、`num_samples=2`、`iters_w=1`、`iters_a=1`、`device=cuda`、`gpus=0`。
  - run 内 `post_recon_snr=7.8397 dB`。
- smoke diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002_positive_scale/diagnostics/20260504_221230_e002b_smoke_diagnostics`
  - `activation_quantizers=52`
  - `non_positive_delta_count=0`
  - `fake_quant_mse_max=0.0006604313966818154`
- formal quant run：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260504_221242_e002b_w4a8_positive_scale_1024samples_w20000_a5000`
  - 命令：W4A8、`num_samples=1024`、`batch_size=16`、`iters_w=20000`、`iters_a=5000`、`device=cuda`、`gpus=0`。
- formal diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002_positive_scale/diagnostics/20260504_232451_e002b_final_diagnostics`
- formal single-sample reload eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002_positive_scale/eval/20260504_232504_e002b_final_single_eval`

#### 正式 run 指标

- run 内阶段指标：
  - `quant_post_weight_recon_snr_db=11.696099054461113`
  - `quant_pre_act_recon_snr_db=4.9874515693637465`
  - `quant_post_act_recon_snr_db=5.236280200086368`
  - `quant_act_recon_snr_gain_db=0.2488286307226213`
  - `activation_reconstruction_seconds=1323.0121958255768`
- checkpoint reload 单样本评估：
  - `quant_snr_db=5.230110892430229`
  - `quant_ssim=0.6613561048695519`

#### 正式诊断指标

- `activation_quantizers=52`
- `non_positive_delta_count=0`
- offender layers：空。
- `effective_int_levels_min=30`
- `fake_quant_mse_max=0.008150355890393257`
- `fake_quant_mse_mean=0.00018797567968744142`
- transformer / Linear：
  - effective level min：30
  - relative MSE max：`0.010542650171161586`
  - fake-quant MSE max：`0.008150355890393257`
- attention projection：
  - effective level min：30
  - effective level mean：65.4
  - relative MSE max：`0.002503253427257536`
- attention qkv：
  - effective level min：131
  - fake-quant MSE max：`0.008150355890393257`
  - relative MSE max：`0.010542650171161586`
- activation `delta` 最小值：
  - 最小正 delta 为 `0.000576426915358752`，位于 `model.stage4.0.block.trans_branch.mlp.2`。
  - `eps=1e-8` 边界命中数为 0，说明最终 checkpoint 没有 layer 被 clamp 卡在极小下界。

#### 与 E001b final W4A8 baseline 对比

- 非法 scale：
  - E001b final：`non_positive_delta_count=2`
  - E002b final：`non_positive_delta_count=0`
  - 结论：E002a 正 scale 投影成功消除了 final checkpoint 中的负 `delta`。
- 单样本 SNR：
  - E001b final run 内 `quant_post_act_recon_snr_db=5.227702998470372`
  - E002b final run 内 `quant_post_act_recon_snr_db=5.236280200086368`
  - 提升约 `0.0086 dB`，不构成实质性精度恢复。
- transformer / Linear：
  - E001b effective level min：17
  - E002b effective level min：30
  - E001b relative MSE max：`0.020646367816721696`
  - E002b relative MSE max：`0.010542650171161586`
  - 结论：正 scale 修复让部分局部指标改善，但 transformer/Linear 仍然是主要劣化区域。
- fake-quant MSE：
  - E001b fake_quant_mse_max：`0.003323915181681514`
  - E002b fake_quant_mse_max：`0.008150355890393257`
  - E002b worst layer 仍是 `model.stage1.0.block.trans_branch.attn.qkv`。

#### E002b 结论

E002b 证明：E002a 的 post-step clamp 能修掉 final checkpoint 中的非法 activation scale，且正式 checkpoint 中没有任何 activation `delta` 停在 `1e-8` 下界；因此“负 delta”确实是一个必须修复的合法性问题。

但 E002b 同时证明：只修正 scale 正值约束几乎不能恢复 W4A8 的最终 SNR。W4A8 仍从 W4A32 weight recon 的约 11.7 dB 掉到约 5.23 dB，说明当前主要精度瓶颈更可能来自 activation quantization 本身的初始化/范围选择、局部 reconstruction 目标、attention qkv/proj 敏感性、tensor-wise activation scale 或学习率/迭代策略，而不只是负 scale。

下一步应进入 E002c / E003 方向：

- 记录 activation delta 变化幅度，判断是否有 scale 被优化到极端但仍为正。
- 做 `activation_lr` sweep，例如 `4e-4 / 1e-4 / 4e-5`。
- 比较 `asym=False` vs `asym=True` 的 activation reconstruction 输入口径。
- 针对 attention qkv/proj 做冻结或单独策略对照。
- 考虑 delta ratio 约束、log-scale / softplus 参数化，或 activation clipping / range 学习。

### E002c：activation-only 初始化敏感性实验

- 日期：2026-05-05
- 负责人：Codex
- 实验目的：从同一个 E002b W4 weight-recon checkpoint 出发，只重新做 A8 activation 初始化，不重跑 W20000，也默认不跑 A5000 activation reconstruction，验证“小样本 smoke 约 8 dB”是否来自 calibration 子集对单张 eval 图的偶然匹配。
- 起点 checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260504_221242_e002b_w4a8_positive_scale_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq_weight_recon.pth`

#### 工具变更

- 新增 activation-only CLI：
  - `SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py`
- 功能边界：
  - 从 weight-recon checkpoint 重建 QuantModel。
  - 只做 activation quantizer 初始化。
  - 默认保存 `quantized_scrn_brecq_pre_act_recon.pth`。
  - 默认 `--skip-act-recon`，因此不生成 final activation reconstruction checkpoint。
  - 不修改 `UniformAffineQuantizer`、不修改 reconstruction 算法。
- 新增最小测试：
  - `SCRN_BRECQ_app/scrn_brecq/tests/test_activation_only_quantize_scrn.py`

#### 运行产物

- smoke run：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002c_init_sensitivity/quant/20260505_150649_e002c_smoke_init_n0002`
- smoke diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002c_init_sensitivity/diagnostics/20260505_150741_e002c_smoke_diag_n0002`
- 正式 init-only sweep runs：
  - `n=2`：`20260505_150800_e002c_init_n0002`
  - `n=8`：`20260505_150814_e002c_init_n0008`
  - `n=16`：`20260505_150826_e002c_init_n0016`
  - `n=64`：`20260505_150842_e002c_init_n0064`
  - `n=256`：`20260505_150917_e002c_init_n0256`
  - `n=1024`：`20260505_150952_e002c_init_n1024`
- 固定 64-sample diagnostics：
  - 为避免 diagnostics 样本数和 init 样本数混在一起，所有正式 sweep checkpoint 都使用 64 个 calibration 样本诊断。
  - CPU diagnostics 单组耗时过长，因此正式 6 组统一用 `--device cuda`。
  - `n=2`：`20260505_151932_e002c_diag_cuda_n0002`
  - `n=8`：`20260505_152323_e002c_diag_cuda_n0008`
  - `n=16`：`20260505_152708_e002c_diag_cuda_n0016`
  - `n=64`：`20260505_153054_e002c_diag_cuda_n0064`
  - `n=256`：`20260505_153439_e002c_diag_cuda_n0256`
  - `n=1024`：`20260505_153825_e002c_diag_cuda_n1024`

#### 必须澄清的实现细节

activation 初始化并不是自动使用全部 `num_samples`。当前代码路径是：

- calibration loader 先收集 `num_samples` 个样本。
- `initialize_activation_quantization(...)` 再取 `calibration_data[:min(init_batch_size, num_samples)]` 做一次前向初始化。
- 默认 `init_batch_size=64`。

因此，本轮 `num_samples=64/256/1024` 三组在默认设置下实际都只用前 64 个样本初始化 activation scale。这解释了为什么这三组的 SNR、delta、diagnostics 指标完全一致。这个发现也反过来说明：E002b formal 的 A8 初始化不是 1024-sample activation init，而是 1024 个 calibration 样本被收集后，只用前 64 个样本完成 activation scale 初始化。

#### SNR 结果

| `num_samples` | 有效 init 样本数 | `quant_pre_act_recon_snr_db` | A8 init 相对 W4A32 掉点 |
| --- | ---: | ---: | ---: |
| 2 | 2 | 8.380215760145743 | -3.3158832943153704 |
| 8 | 8 | 7.186574348364017 | -4.509524706097096 |
| 16 | 16 | 6.488236236978712 | -5.207862817482401 |
| 64 | 64 | 4.9874515693637465 | -6.708647485097366 |
| 256 | 64 | 4.9874515693637465 | -6.708647485097366 |
| 1024 | 64 | 4.9874515693637465 | -6.708647485097366 |

W4 weight-recon 起点单样本 SNR 固定为：

- `quant_post_weight_recon_snr_db=11.696099054461113`

#### 诊断结果

| `num_samples` | `non_positive_delta_count` | `delta_min` | `delta_max` | `effective_int_levels_min` | `fake_quant_mse_max` | Linear effective min | attention proj effective min | attention qkv effective min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | 0.00023412889277096838 | 0.04403826966881752 | 146 | 0.00009798040264286101 | 177 | 245 | 244 |
| 8 | 0 | 0.0002338363992748782 | 0.04399380832910538 | 154 | 0.00009798632527235895 | 210 | 245 | 246 |
| 16 | 0 | 0.0002460727409925312 | 0.04407128319144249 | 164 | 0.00009808229515329003 | 206 | 245 | 250 |
| 64 | 0 | 0.00025086768437176943 | 0.04417974501848221 | 191 | 0.00009840026177698746 | 208 | 246 | 251 |
| 256 | 0 | 0.00025086768437176943 | 0.04417974501848221 | 191 | 0.00009840026177698746 | 208 | 246 | 251 |
| 1024 | 0 | 0.00025086768437176943 | 0.04417974501848221 | 191 | 0.00009840026177698746 | 208 | 246 | 251 |

局部变化最大的层：

- `delta` 从 2-sample 到 1024-sample 变化最大的是 `model.stage1.0.block.trans_branch.mlp.2`：
  - `0.008632086217403412 -> 0.0038006349932402372`
- effective int level 变化最大的是同一层：
  - `177 -> 256`
- `model.head` 和 `model.tail` 的 effective int level 也随有效 init 样本数增加明显上升：
  - head：`209 -> 254`
  - tail：`146 -> 191`

#### 对 E002b smoke 现象的解释

小样本 activation init 的单样本 SNR 明显更高：

- 2-sample：`8.3802 dB`
- 8-sample：`7.1866 dB`
- 16-sample：`6.4882 dB`
- 64-sample：`4.9875 dB`

但这并不代表小样本量化状态更健康。固定 64-sample diagnostics 显示，随着有效 init 样本数增加，effective int level 通常更高、Linear relative MSE 反而更低，attention qkv/proj 在 pre-act-recon 阶段也保持较健康：

- attention proj effective min：`245 -> 246`
- attention qkv effective min：`244 -> 251`
- Linear relative MSE max：`0.00024304398617962052 -> 0.00020600040279561196`

因此，2-sample 的高 SNR 更可能是 activation scale 对当前单张 eval 图的偶然匹配或小样本 overfit，而不是更好的通用 A8 初始化。E002b smoke 约 8 dB 的现象可以由这一点解释：它不是正式 W4A8 流程变好了，而是有效 activation init 样本数极小。

#### Full init 256/1024 的限制

为了确认 `num_samples=256/1024` 如果真正全部参与 activation 初始化会怎样，尝试运行：

- `--num-samples 256 --init-batch-size 256`

结果：CUDA 0 在 activation MSE scale 初始化阶段 OOM，报错位置在 `UniformAffineQuantizer._init_mse_scale -> quantize(...)`，尝试额外分配约 2 GiB 时失败。

这说明当前 activation 初始化实现不是流式统计，而是直接用整批 activation tensor 做 MSE scale 搜索。若要真正测试 256/1024 activation init，需要先做以下之一：

- 改成分层/分批收集 activation range，再初始化 scale。
- 用更小的候选搜索内存占用。
- 在更大显存设备上重跑。
- 先把 `init_batch_size` 作为独立实验变量，而不是只改 `num_samples`。

#### E002c 结论

E002c 回答了两个关键问题：

1. 小样本 A8 init 的确能让当前单张 eval 的 SNR 显著高于 64-sample init，但这更像 calibration subset 对单张 eval 的偶然匹配，不是健康量化状态。
2. 当前正式 W4A8 的 A8 初始化瓶颈已经在 `init_batch_size=64` 暴露；继续把 `num_samples` 从 64 增大到 1024，在现有代码下不会改变 activation 初始化结果。

这使后续方向更明确：

- 不应把“扩大 `num_samples`”当作下一步主实验，因为 activation init 实际受 `init_batch_size` 控制。
- 下一步应转向 calibration subset 选择、activation clipping/range 学习、分批初始化统计，或直接做 activation reconstruction 的学习率/冻结策略。
- pre-act-recon 阶段 attention qkv/proj 指标并不崩；E002b final 中 qkv/proj 崩坏更像 activation reconstruction 阶段引入的问题。

### E002 阶段收束：进入 E003 前的判断

- 日期：2026-05-05
- 负责人：Codex
- 目的：把 E002 的修复边界、已完成状态和后续优先级记录清楚，避免后续继续在低优先级方向上分散实验资源。

#### 当前已经做到的程度

E002 当前已经完成三个层面的工作：

1. 合法性修复已经完成。
   - E002a 已在 activation reconstruction 的 `optimizer.step()` 后加入 activation `delta` 正值投影。
   - 该修复不改变普通推理公式，也不改变历史 checkpoint restore 语义。
   - E002b formal checkpoint 证明 `non_positive_delta_count=0`，offender layers 为空。
2. 正 scale 修复的收益边界已经验证。
   - E001b old final：`quant_post_act_recon_snr_db=5.227702998470372`
   - E002b positive-scale final：`quant_post_act_recon_snr_db=5.236280200086368`
   - 提升约 `0.0086 dB`，说明负 `delta` 是必须修的合法性问题，但不是当前 W4A8 失败的主要瓶颈。
3. A8 initialization 敏感性已经初步澄清。
   - E002c 证明当前 activation 初始化实际受 `init_batch_size` 控制，而不是单纯受 `num_samples` 控制。
   - 默认 `init_batch_size=64` 下，`num_samples=64/256/1024` 得到完全相同的 A8 init 状态。
   - 2/8/16 样本在当前单张 eval 图上 SNR 更高，但 fixed diagnostics 不支持“小样本状态更健康”的解释。

因此，E002 可以阶段性收束：保留正 scale clamp 作为必要工程修复，但不再把“更复杂的 delta 约束”作为下一阶段主线。

#### 暂时降级的方向

以下方向仍然可能有价值，但当前优先级应低于 E003：

- 限制 `delta` 相对初始化值的变化幅度，例如 `min_ratio * init_delta <= delta <= max_ratio * init_delta`。
- 对 `log(delta / init_delta)` 增加正则，防止 scale 被优化到极端范围。
- 降低 `activation_lr` 或缩短 `iters_a`，判断当前学习率/迭代数是否过激。
- 改为 log-scale 或 `softplus(raw_delta) + eps` 参数化，从参数空间上保证 scale 为正。
- 对 attention projection/qkv 做选择性冻结或单独策略，判断 transformer/Linear 是否是有害重构路径。
- 重新评估 `asym=False` 的 activation reconstruction 设定是否和真实量化输入分布不一致。

降级原因：

- 这些方向大多作用在 activation reconstruction 或 delta 优化空间。
- 但当前最大掉点发生在 A8 init 一打开时：W4 weight recon 约 `11.696 dB`，A8 init 后约 `4.987 dB`。
- activation reconstruction 后只到约 `5.236 dB`，说明在 reconstruction trick 之前，activation range / scale 初始化已经是主问题。

#### 进入 E003 前必须注意的问题

后续处理 W4A8 激活量化时，需要特别注意以下约束：

- 不要再把 `num_samples=1024` 直接理解为 “activation init 使用了 1024 个样本”。
  - 现有代码实际使用 `min(num_samples, init_batch_size)` 做 activation init。
  - 若不显式调整 `init_batch_size`，64/256/1024 的 activation init 状态不会变化。
- 不要只看单张 eval SNR。
  - E002c 的 2-sample init 在当前单张 eval 图上更高，但 diagnostics 说明这更可能是 calibration subset 对该图的偶然匹配。
  - E003 应优先建立 multi-sample eval 或至少多张固定 eval set。
- 不要直接启动长时间 A5000 reconstruction sweep。
  - 在 A8 init 已经明显崩坏的前提下，先调 reconstruction 学习率或冻结层容易把问题定位复杂化。
- 不要把 pre-act-recon 阶段 attention qkv/proj 视为已经崩坏。
  - E002c fixed diagnostics 中 attention qkv/proj 的 effective levels 很高。
  - E002b final 中 qkv/proj 恶化更像 activation reconstruction 阶段进一步引入的问题。
- 若要真正测试 256/1024 activation init，必须先解决显存/初始化策略问题。
  - 直接 `--init-batch-size 256` 已在 CUDA 0 上 OOM。
  - 需要考虑分批统计 range、降低 MSE scale 搜索内存、或使用更大显存设备。

#### E003 建议入口

E003 不应从 delta ratio/log-scale/softplus 开始，而应优先解决 activation initialization 与评估协议：

1. 建立 multi-sample evaluation，降低单张 eval 图带来的误导。
2. 把 `init_batch_size` 作为真实实验变量，而不是只改 `num_samples`。
3. 对比不同 calibration subset 对 A8 init SNR 和 fixed diagnostics 的影响。
4. 调查 activation clipping/range 初始化策略，尤其是 tensor-wise activation scale 是否过度受 calibration 分布影响。
5. 在确认 A8 init 的行为后，再决定是否回到 activation reconstruction 的学习率、冻结 qkv/proj 或 delta 参数化实验。

### E003 初步计划：初始化覆盖与优化稳定性

- 日期：2026-05-05
- 负责人：Codex
- 阶段目标：在 E002 已完成正 scale 合法性修复后，系统验证 W4A8 的主要瓶颈到底来自 activation 初始化覆盖不足、calibration subset 偶然匹配，还是 activation reconstruction 优化不稳定。

#### E003 要回答的问题

E003 不是为了继续证明负 `delta` 的问题。负 `delta` 已经由 E002a 作为合法性问题修掉。E003 要回答的是：

1. A8 init 为什么会让 W4 weight-recon 从约 `11.696 dB` 直接掉到约 `4.987 dB`？
2. E002c 中 2/8/16 样本 init 的单张 eval SNR 更高，究竟是有效策略，还是对当前 eval 图的偶然匹配？
3. 当前 `init_batch_size=64` 是否已经足够，还是 activation range 初始化需要更大的覆盖、更好的 subset 或 clipping/range 策略？
4. activation reconstruction 的学习率/迭代是否进一步破坏了 pre-act-recon 阶段还算健康的 attention qkv/proj？

#### E003 分阶段设计

E003 不建议直接做一个大矩阵 sweep。建议拆成三个阶段，逐步减少混淆变量。

##### E003a：固定多样本评估口径

目的：先解决单张 eval 图误导问题。

需要做的事：

- 使用已有评估入口或新增最小 multi-sample eval 工作流，对同一 checkpoint 在固定 eval subset 上计算平均 SNR/SSIM。
- eval subset 必须固定 seed、固定样本列表，并把样本清单写入 run 目录。
- 至少记录：
  - mean / median / min / max SNR
  - 每张样本 SNR
  - mean / median SSIM
  - 输入 degraded baseline SNR
- 第一批对比对象：
  - W4 weight-recon checkpoint
  - E002c 2-sample A8 init checkpoint
  - E002c 8-sample A8 init checkpoint
  - E002c 16-sample A8 init checkpoint
  - E002c 64-sample A8 init checkpoint

验收标准：

- 如果 2/8/16-sample init 在多样本 eval 上仍明显高于 64-sample init，则小样本 range 不是单张图偶然，需要进入 calibration subset / clipping 方向。
- 如果多样本 eval 上差距消失或反转，则 E002c 的高 SNR 主要是单张 eval 图匹配，后续不能以单张图结果选择 activation scale。

##### E003b：`init_batch_size` 作为真实变量

目的：验证 activation 初始化覆盖本身的影响。

推荐变量：

- `init_batch_size=2/8/16/32/64`
- 暂不把 256/1024 纳入正式必跑项，因为 E002c 已证明 `--init-batch-size 256` 在 CUDA 0 上 OOM。
- 若要测试 256/1024，必须先单独解决显存或初始化实现问题，不能和主实验混在一起。

固定条件：

- 起点统一为 E002b W4 weight-recon checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260504_221242_e002b_w4a8_positive_scale_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq_weight_recon.pth`
- `num_samples` 至少等于 `init_batch_size`。
- 默认不跑 activation reconstruction，只做 A8 init。
- 每个 checkpoint 都跑：
  - E003a multi-sample eval
  - 固定 64-sample E001 diagnostics

记录指标：

- A8 init multi-sample SNR/SSIM
- single-sample SNR 只作为兼容参考，不作为主结论
- `non_positive_delta_count`
- `delta_min/delta_max`
- `effective_int_levels_min`
- Linear / transformer relative MSE max
- attention qkv/proj effective level 与 fake-quant MSE
- top outlier / worst MSE layers

验收标准：

- 明确判断 `init_batch_size` 增大是否真的改善泛化评估。
- 明确判断 64-sample init 是过宽 range、过窄 range，还是 calibration subset 不匹配。

##### E003c：activation reconstruction 学习率 sweep

目的：在 A8 init 行为被 E003a/E003b 固定后，再验证 activation reconstruction 是否优化过激。

候选变量：

- `activation_lr=4e-4`
- `activation_lr=1e-4`
- `activation_lr=4e-5`

建议做法：

- 不要一开始就重跑 W20000。
- 从同一个 W4 weight-recon checkpoint 开始，使用 E003b 选定的 activation init 设置，然后只跑 activation reconstruction。
- 如果工具支持，应保存：
  - pre-act-recon checkpoint
  - final act-recon checkpoint
  - activation reconstruction 前后 multi-sample eval
  - activation reconstruction 前后 E001 diagnostics
- 若每个 A5000 run 成本过高，可先做 smoke/short run：
  - `iters_a=500/1000`
  - 确认趋势后再跑 `iters_a=5000`

记录指标：

- `quant_pre_act_recon_snr_db`
- `quant_post_act_recon_snr_db`
- `quant_act_recon_snr_gain_db`
- multi-sample eval mean/median/min SNR
- `non_positive_delta_count`
- offender layers
- attention qkv/proj 的 effective level、relative MSE 和 fake-quant MSE
- final `delta` 是否比 pre-act-recon 发生极端变化

验收标准：

- 如果低学习率能明显提升 multi-sample final SNR，并避免 qkv/proj 指标崩坏，则 activation reconstruction 确实存在优化稳定性问题。
- 如果低学习率仍不能恢复，而 A8 init 本身已低，则主线继续转向 activation range/clipping，而不是继续调 reconstruction。

#### 关于 `init_batch_size=64/256/1024` 的处理

原始设想中包含：

- `init_batch_size=64/256/1024`

但 E002c 已经给出新的约束：

- `init_batch_size=256` 在 CUDA 0 上直接 OOM。
- 当前 activation MSE scale 初始化不是流式统计，会在较大 batch 上产生显存压力。

因此 E003 不应把 256/1024 当作第一轮必跑变量。更稳妥的处理是：

1. 第一轮只跑当前显存可承受的 `2/8/16/32/64`，建立趋势。
2. 如果趋势显示更大 init 覆盖可能有价值，再单开 E003d 或 E004 解决 full-init 内存问题。
3. 解决方式可以是分批统计 activation range、降低 MSE scale 搜索内存、使用更大显存 GPU，或把 scale 初始化改成更轻量的 percentile/max 方案。

#### E003 的优先级判断

E003 的优先级应高于以下方向：

- delta ratio clamp
- log-scale / softplus 参数化
- `log(delta / init_delta)` 正则
- attention qkv/proj 选择性冻结
- `asym=False` vs `asym=True` activation reconstruction 对照

原因：

- 这些方向主要解释 reconstruction 阶段的问题。
- 但目前最大损失在 A8 init 阶段已经发生。
- 如果不先建立多样本评估和 activation init 变量控制，后续 reconstruction sweep 可能会优化到单张 eval 或某个 calibration subset 上，结论不可靠。

#### E003 推荐执行顺序

1. E003a：建立或确认 multi-sample eval 工作流。
2. E003a-baseline：评估 W4 weight-recon、E002c 2/8/16/64 init checkpoints。
3. E003b：正式做 `init_batch_size=2/8/16/32/64` init-only sweep。
4. E003b-diagnostics：每个 checkpoint 跑固定 64-sample E001 diagnostics。
5. 根据 E003a/E003b 结果决定：
   - 若小 init batch 在 multi-sample 上仍更好：进入 calibration subset / clipping / range 学习。
   - 若 64 init 更稳或差距消失：进入 activation reconstruction 学习率 sweep。
6. E003c：只在评估口径稳定后做 `activation_lr=4e-4/1e-4/4e-5`。

### E003a：multi-sample eval 口径建立与 baseline

- 日期：2026-05-05
- 负责人：Codex
- 目的：用固定多样本 eval subset 复核 E002c 的单张 eval 结论，判断小样本 A8 init 的高 SNR 是否能泛化。

#### 工具状态

已有 `evaluate_quantized_scrn_multi.py` 可以复用：

- 固定 seed 从 clean patch 目录抽样。
- 在线生成 degraded 输入。
- 输出 `metrics.json`、`summary.md`、`per_sample_metrics.jsonl`、`config.json`。
- `config.json` 记录 `selected_sample_paths`。

E003a 只做了一个小工具补齐：

- `build_aggregate_metrics(...)` 新增 median 聚合字段。
- `summary.md` 追加主要 mean/median 指标。
- 新增 `test_evaluate_quantized_scrn_multi.py`，验证 median 聚合和旧字段 alias。

#### 评估设置

- eval dataset：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches`
- `num_eval_samples=128`
- `batch_size=16`
- `seed=20260427`
- `device=cuda`
- `--no-save-figures`
- run root：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E003_multi_sample_eval`
- selected sample list hash：
  - `cf3b4fe1a094`
  - 所有正式 run 的 `selected_sample_paths` 完全一致。

#### 正式 run 产物

- W4 weight-recon：
  - `20260505_165402_e003a_median_w4_weight_recon_eval128`
- A8 init, effective init samples = 2：
  - `20260505_165418_e003a_median_a8_init_n0002_eval128`
- A8 init, effective init samples = 8：
  - `20260505_165438_e003a_median_a8_init_n0008_eval128`
- A8 init, effective init samples = 16：
  - `20260505_165459_e003a_median_a8_init_n0016_eval128`
- A8 init, effective init samples = 64：
  - `20260505_165516_e003a_median_a8_init_n0064_eval128`
- E002b positive-scale final：
  - `20260505_165531_e003a_median_e002b_final_eval128`

#### 128-sample eval 结果

| checkpoint | SNR mean | SNR median | SNR min | SNR max | SSIM mean | mean SNR - FP32 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W4 weight-recon | 4.79725691949495 | 4.309259459875541 | 1.3343390090950384 | 10.249259240602004 | 0.7049305604844021 | -1.2928323404783364 |
| A8 init n=2 | -7.0231250370839735 | -7.857011917587894 | -14.339920378397583 | 8.000126045067118 | 0.18979444123250055 | -13.113214297057262 |
| A8 init n=8 | -7.023006163652582 | -7.7914053078130605 | -14.271765951371972 | 7.704190278141572 | 0.2021710648360309 | -13.113095423625868 |
| A8 init n=16 | -7.047440279252236 | -7.8041388297714125 | -14.294644861268328 | 7.493808244600226 | 0.196979314851416 | -13.137529539225525 |
| A8 init n=64 | -7.102088710746793 | -7.850034466981217 | -14.370487804609258 | 6.784886648518827 | 0.19452354877811634 | -13.192177970720081 |
| E002b final | -7.071334905403255 | -7.811254783170435 | -14.321177469597867 | 6.931303675318263 | 0.20218104685008775 | -13.161424165376543 |

同一 128-sample eval subset 上：

- input degraded SNR mean：`0.9709 dB`
- FP32 SNR mean：`6.0901 dB`
- W4 weight-recon SNR mean：`4.7973 dB`

#### E003a 结论

E003a 直接推翻了 “2/8/16-sample A8 init 可能更好” 这个解释。

E002c 中单张 eval 的 A8 init SNR：

- 2-sample：`8.3802 dB`
- 8-sample：`7.1866 dB`
- 16-sample：`6.4882 dB`
- 64-sample：`4.9875 dB`

但在固定 128-sample eval 上：

- 2/8/16/64-sample A8 init 全部约 `-7 dB`。
- 它们之间差距极小，且全部远低于 W4 weight-recon 的 `4.7973 dB`。
- E002b final activation reconstruction 也只有 `-7.0713 dB`，没有在多样本上恢复 A8 init 崩坏。

因此，E002c 的小样本高 SNR 基本可以判定为对当前单张 eval 图的偶然匹配或过拟合，不应作为选择 activation init 策略的依据。

#### 对后续 E003 的影响

E003a 之后，E003b 不应再以单张 eval SNR 为主指标。后续必须使用：

- multi-sample eval mean/median/min/max SNR
- fixed diagnostics
- activation quantizer 合法性
- attention qkv/proj 和 Linear 的局部误差

E003a 同时说明：

- W4 weight-recon 在多样本上已有明显泛化掉点，但仍远好于 A8。
- A8 init 是当前最主要的多样本崩坏点。
- activation reconstruction 没有在多样本上修复 A8 崩坏。
- 下一步 E003b 应继续围绕 activation range / init_batch_size / calibration subset 设计，而不是立即做长时间 A5000 学习率 sweep。

### 后续实验资源使用原则：优先充分利用 GPU

- 日期：2026-05-05
- 负责人：用户 / Codex
- 原则：后续激活量化相关实验应尽可能充分利用 GPU 资源，避免在 CPU 上运行本可用 GPU 加速的量化、评估或诊断任务。

#### 具体要求

- 默认优先使用 `--device cuda`。
- 单卡实验默认明确指定 `--gpus 0` 或当前计划中的目标 GPU。
- 长时间实验、正式 sweep、multi-sample eval、activation reconstruction 不应默认落到 CPU。
- 如果 CUDA 不可用、显存不足或 GPU 任务失败，不自动悄悄切回 CPU；应先记录失败原因，再决定是否改 CPU 或换 GPU。
- 如果为了和历史结果对齐必须使用 CPU，应在日志中明确写出原因和口径差异。
- 实验前应尽量检查 GPU 可用性和显存占用，避免因已有进程占用导致 OOM 或结论中断。

#### 对 E003 后续的影响

- E003b 的 init-only sweep、multi-sample eval 和 fixed diagnostics 应优先使用 GPU。
- E003c 如涉及 activation reconstruction 或 learning-rate sweep，应优先使用 CUDA，且在命令中明确 GPU 选择。
- 只有当某个工具在 GPU 上存在确定性、显存或兼容性问题时，才把 CPU 作为有记录的例外方案。

### 后续实验资源使用原则补充：多卡并行

- 日期：2026-05-05
- 负责人：用户 / Codex
- 原则补充：项目环境存在多张 GPU。后续实验不只是优先使用单张 GPU，而应在实验设计允许时充分利用多卡资源，缩短 sweep、eval 和 diagnostics 的总耗时。

#### 多卡使用方式

- 对互相独立的实验配置，优先做 job-level 多卡并行：
  - 例如 E003b 中不同 `init_batch_size` 的 init-only run 可以分别分配到不同 GPU。
  - 不同 checkpoint 的 multi-sample eval 可以分别分配到不同 GPU。
  - 不同 diagnostics run 可以分别分配到不同 GPU，前提是显存充足且输出目录/`run-name` 不冲突。
- 如果 CLI 原生支持分布式并且该任务适合分布式，再使用内部多卡分布式。
  - 目前已知 `quantize_scrn.py --distributed` 主要适用于 W-only reconstruction。
  - activation reconstruction 路径此前明确不支持 distributed，因此这类任务应优先考虑多进程多卡并行跑不同配置，而不是强行单任务分布式。
- 对不支持内部多卡的脚本，使用显式 GPU 绑定：
  - `CUDA_VISIBLE_DEVICES=<gpu_id>` 或 CLI 的 `--gpus <gpu_id>`。
  - 每个并行 run 必须使用独立 `--run-name`，避免输出覆盖。
- 正式并行实验前应检查 GPU 占用：
  - `nvidia-smi`
  - 记录可用 GPU、显存余量和选择理由。

#### 对后续 E003 的具体影响

- E003b sweep 应优先按配置拆分到多张 GPU 并行执行，而不是串行使用单卡。
- E003a/E003b 的 multi-sample eval 可以按 checkpoint 维度并行。
- 如果需要固定 diagnostics，也可以按 checkpoint 维度并行，但要留意 E001 diagnostics 的 hook/quantile 统计显存和耗时。
- 若某个单 run 因显存限制无法使用大 `init_batch_size`，不能简单认为“多卡能自动解决”；除非脚本支持模型/数据分布式，否则仍需分批初始化策略或更大单卡显存。

### E003 阶段性总结：当前已完成内容、暂缓内容与后续入口

- 日期：2026-05-05
- 负责人：用户 / Codex
- 状态：阶段性收束，准备从 E003 转入 E004 activation range / clipping 主线。

#### 已完成内容

1. E003a 已完成 multi-sample eval 口径建立。

   - `evaluate_quantized_scrn_multi.py` 已补齐 mean / median / min / max 聚合能力。
   - 已用固定 128-sample eval subset 对 W4 weight-recon、A8 init n=2/8/16/64、E002b final 做同口径评估。
   - 所有正式 run 使用相同 sample list，hash 为 `cf3b4fe1a094`。

2. E003a 给出了关键结论。

   - W4 weight-recon 128-sample SNR mean 为 `4.7973 dB`。
   - A8 init n=2/8/16/64 在 128-sample eval 上全部约 `-7 dB`。
   - E002b positive-scale final activation reconstruction 也约 `-7.0713 dB`。
   - 因此，E002c 单张 eval 中小样本 A8 init 的 `6-8 dB` 表现不具备泛化意义，更像 calibration subset 与单张 eval 图的偶然匹配。

3. E003 已明确主要瓶颈位置。

   - 主要掉点不是负 `delta` 合法性问题；E002a/E002b 已把 final `non_positive_delta_count` 修到 0，但多样本 SNR 没有恢复。
   - 主要掉点也不应优先归因于 activation reconstruction 学习率；因为 A8 init 打开后在多样本上已经崩坏。
   - 当前最可信的主瓶颈是 activation quantizer 的 range / clipping / 初始化策略不适合 SCRN 激活分布。

#### 未完成但已决定暂缓的内容

1. E003b 原计划的 `init_batch_size=2/8/16/32/64` sweep 暂缓。

   - 原因：E003a 已经证明低样本 init 在 multi-sample eval 上全部很差。
   - 用户判断低样本继续测试价值有限；Codex 同意。
   - 继续测试 `2/8/16/32/64` 不太可能改变主结论，只会重复证明小样本不稳定。

2. E003b 中 `init_batch_size=256/1024` 暂不直接执行。

   - 原因：E002c 已尝试 `num_samples=256, init_batch_size=256`，在 CUDA 上 OOM。
   - 当前 activation init 是单进程、单设备一次性前向，不会因为给脚本多个 GPU 就自动把一个 batch 分摊到多卡。
   - 现有 `--gpus` 主要控制可见 GPU；现有 distributed 路径不支持 activation quant / activation reconstruction。
   - 因此，多卡当前适合并行多个独立 run，不能直接解决单个 256 init 的显存峰值。

3. E003c activation reconstruction 学习率 sweep 暂缓。

   - 原因：E003a 显示 A8 init 本身已经在多样本 eval 上崩坏。
   - 学习率 sweep 只能回答 activation reconstruction 是否进一步破坏，不能解决 A8 init 已低的问题。
   - 若后续需要保留证据链，可以只做 short sweep：
     - 固定 `init_batch_size=64`
     - `iters_a=500/1000`
     - `activation_lr=4e-4/1e-4/4e-5`
     - 只看 multi-sample final SNR 是否有显著恢复。
   - 如果 short sweep 仍无明显恢复，则不再跑 A5000。

#### 后续再回到 E003 时的注意事项

- 不能再使用单张 eval SNR 作为主要判断依据。
- 所有结论必须基于固定 multi-sample eval subset，并记录 sample list hash。
- 如果要测试 `init_batch_size=256/1024`，需要先解决 memory-safe activation init：
  - 分批统计 activation range。
  - 改写 MSE scale 初始化，避免在大 batch 上保留巨大临时张量。
  - 或实现真正的跨卡 activation range / MSE score 聚合。
- 不能假设多卡会自动降低单个 run 的显存峰值；除非脚本明确支持数据/模型分布式。
- 后续 sweep 应充分利用多 GPU 做 job-level 并行，但每个 run 必须有独立 `run-name` 和输出目录。
- 若 CUDA 不可用、OOM 或驱动异常，先记录原因，不要无记录切换 CPU。

#### 下一步建议

E003 当前不再继续扩展。下一阶段建议进入 E004：

- activation range / clipping 策略。
- `scale_method=max`、`max_scale`、`mse` 的对比。
- percentile / outlier clipping calibration。
- 针对 transformer / Linear / attention qkv-proj 的局部分布诊断和可选分层策略。

E004 的核心问题应是：如何让 A8 init 在 multi-sample eval 上不从 W4 weight-recon 的约 `4.8 dB` 直接跌到约 `-7 dB`。

### E004 计划深度分析：插入位置敏感性图谱的合理性与改进

- 日期：2026-05-05
- 负责人：用户 / Codex
- 状态：计划评审与改进，不运行实验。

#### 为什么 E004 有必要

E001-E003 的证据链已经把问题范围收窄：

- E001 证明 final W4A8 中存在非法负 `delta`，但 pre-act-recon 没有该问题。
- E002a/E002b 修复了负 `delta` 合法性，final `non_positive_delta_count=0`，但 SNR 没有实质恢复。
- E003a 用 128-sample eval 证明 A8 init 本身已经崩坏，2/8/16/64 init checkpoints 全部约 `-7 dB`。
- activation reconstruction 在多样本上没有修复 A8 init 崩坏。

因此，下一步不应继续优先调 reconstruction 学习率或 delta 参数化，而应回答一个更基础的问题：

> 52 个 activation quantizer 中，是否存在少数结构位置一打开就造成大部分 SNR 崩坏？

如果答案是肯定的，后续 E005/E006 才能从“全模型统一 A8”转向更合理的选择性策略，例如：

- 保留某些 attention / Linear activation 为 FP32。
- 对特定 stage 使用 clipping 或 percentile calibration。
- 对 CNN branch 和 transformer branch 采用不同 activation quantization 策略。
- 在精度和资源收益之间做可解释取舍。

#### 原计划的合理性

原始 E004 包含：

- 单点开启或单点关闭 activation quantizer。
- 按 stage / branch / module type 分组开启或关闭。
- 输出 sensitivity ranking、activation-volume proxy ranking 和 sensitivity vs resource benefit 表。

这个方向是合理的，原因如下：

1. 它能把“全局 A8 崩坏”拆成可定位的结构问题。

   当前只有全模型 A8 结果，很难判断是所有层普遍不适合 A8，还是少数层拖垮整体。单点和分组开关能把这个问题拆开。

2. 它能验证 E001 中的结构怀疑。

   E001 显示 stage4/stage5 `attention_proj` 曾出现负 `delta`，transformer / Linear 的 relative MSE 明显更差。E004 可以验证这些位置是否也是 A8 init 崩坏的关键位置。

3. 它能为后续 mixed activation precision 提供依据。

   如果少数 quantizer 敏感但 activation volume 不大，保留 FP32 的成本可能很低，却能恢复大部分 SNR。这比盲目全局 clipping 更有工程价值。

4. 它能避免把所有问题都归因于 range/clipping。

   如果单点/分组开关发现 CNN branch 也全部高度敏感，则问题可能是量化公式、输入分布或 eval pipeline 更基础；如果只集中在 attention/Linear，则后续策略应更有针对性。

#### 原计划需要改进的地方

原计划仍有几个风险，需要在 E004 执行前修正。

1. 不能一开始就跑完整 52 层大 sweep。

   单点关闭 52 次、单点开启 52 次、再加分组实验，如果每次都跑 128-sample eval，会消耗较多 GPU 时间。应先做工具 smoke 和小规模验证，确认开关语义正确。

2. 单点关闭和单点开启解释不同，不能混为一谈。

   - 单点关闭：从全 A8 出发，关闭某个 quantizer，看能恢复多少。
   - 单点开启：从 W4A32 出发，只开启某个 quantizer，看单独造成多少损失。

   单点关闭更接近“修当前全 A8 checkpoint”，但可能受层间耦合影响；单点开启更适合看独立危害，但可能低估多层累积误差。两者应分阶段做。

3. 应优先使用 A8 init checkpoint，而不是 final act-recon checkpoint。

   E003a 已证明主要崩坏发生在 A8 init。E004 第一轮如果直接用 final checkpoint，容易混入 activation reconstruction 对 `delta` 的二次影响。第一轮应以 A8 init n=64 / pre-act-recon checkpoint 为主。

4. sensitivity ranking 必须绑定同一 multi-sample eval subset。

   单张 eval 已被 E003a 证明会误导判断。E004 所有正式 ranking 必须使用固定 128-sample eval subset，并记录 sample list hash。

5. activation-volume proxy 不能误读为真实部署收益。

   activation volume 只能作为资源收益代理，例如输出元素数、shape、通道数。真实部署收益还取决于算子融合、缓存、内存带宽、硬件 kernel 和 packed runtime。E004 先输出 proxy，不直接声称真实速度或显存收益。

6. 多卡只能用于 job-level 并行。

   E004 中每个开关实验相互独立，适合分配到多张 GPU 并行；但单个 eval run 不应假设可以自动分布式。每个并行 job 必须有独立 `run-name` 和输出目录。

#### 改进后的 E004 分阶段计划

##### E004a：选择性 activation quantizer 开关工具

目标：先建立可追踪、可复现的开关能力，不直接跑完整 sweep。

建议实现：

- 新增 CLI：
  - `SCRN_BRECQ_app/scrn_brecq/cli/evaluate_activation_sensitivity.py`
- 复用：
  - checkpoint 加载逻辑。
  - `activation_diagnostics.py` 中的 quantizer 结构标签。
  - `evaluate_quantized_scrn_multi.py` 的 multi-sample eval 逻辑。
- 支持开关模式：
  - `all_on`
  - `all_off`
  - `disable_one`
  - `enable_one`
  - `disable_group`
  - `enable_group`
- 支持筛选字段：
  - quantizer index
  - module name
  - stage
  - branch
  - role
  - module type
- 输出：
  - `config.json`
  - `metrics.json`
  - `summary.md`
  - `selected_quantizers.csv`
  - `per_sample_metrics.jsonl`

验收标准：

- 2-4 sample smoke 能证明指定 quantizer 的 `disable_act_quant` 或等价开关确实生效。
- `all_off` 应接近 W4A32 / weight-recon 口径。
- `all_on` 应接近 A8 init 或 final W4A8 口径。
- 不生成或提交 run 产物。

##### E004b：小规模 sentinel 敏感性验证

目标：先验证 E004 方法是否能产生有意义差异，再决定是否跑 52 层完整 sweep。

建议选择 8-12 个 sentinel quantizers：

- E001 worst relative MSE 前列。
- E001 lowest effective level 前列。
- stage4/stage5 attention projection。
- attention qkv。
- 若干 CNN Conv2d 对照层。
- 若干 early stage / late stage 对照层。

评估口径：

- checkpoint：优先 A8 init n=64 / pre-act-recon checkpoint。
- eval：固定 128-sample subset，沿用 E003a seed 和 sample list hash。
- device：CUDA，多卡按 job-level 并行。

验收标准：

- 如果关闭某些 sentinel quantizer 能明显恢复 mean/median SNR，说明完整 sensitivity map 有价值。
- 如果所有 sentinel 变化都很小，则需要先检查开关工具是否正确，或说明问题是多层累积误差而非单点主导。

##### E004c：完整单点关闭 ranking

目标：从全 A8 出发，评估每个 activation quantizer 被关闭后的恢复收益。

输出 ranking：

- `delta_snr_mean = snr_mean_disable_one - snr_mean_all_on`
- `delta_snr_median`
- `delta_ssim_mean`
- quantizer index / name / stage / branch / role / module type
- activation shape / element count / channel count

解释：

- 排名前列是“当前全 A8 中最值得保留高精度或单独处理”的候选。
- 如果 top quantizers 集中在 transformer / Linear / attention，则后续策略应结构化。
- 如果 top quantizers 分散，说明可能需要全局 range/clipping 策略。

##### E004d：分组关闭实验

目标：验证结构级规律，避免只凭单点 ranking 过拟合。

建议分组：

- branch：
  - CNN
  - transformer
- module type：
  - Conv2d
  - Linear
- role：
  - attention qkv
  - attention proj
  - MLP / FFN
  - stem / head
- stage：
  - stage1-stage5

输出：

- 每组关闭后的 SNR/SSIM。
- 关闭 quantizer 数量。
- activation-volume proxy 总量。
- group sensitivity per quantizer。
- group sensitivity per activation-volume proxy。

验收标准：

- 能明确判断是否存在“某个 branch / module type / role 主导崩坏”。

##### E004e：单点开启补充验证

目标：从 W4A32 出发，只开启某个 activation quantizer，评估其独立破坏力。

执行条件：

- E004c/E004d 已经显示存在明确敏感层或敏感组。
- 不建议在 E004a 之后立刻跑完整单点开启，因为成本高，且解释要结合单点关闭结果。

用途：

- 验证 E004c 的 top sensitive quantizers 是否确实单独有害。
- 找出可以安全 A8 的 quantizer 集合。

##### E004f：sensitivity vs resource benefit 策略表

目标：把 E004c/E004d/E004e 转成可执行策略。

输出表字段：

- quantizer index
- name
- stage
- branch
- role
- module type
- sensitivity score
- activation-volume proxy
- sensitivity / volume ratio
- recommendation：
  - keep A8
  - keep FP32
  - test percentile clipping
  - test higher precision
  - needs group-level treatment

验收标准：

- 能提出 E005 的具体候选策略，而不是只给排名。

#### E004 与 E004/E005 边界

E004 只回答“哪里敏感”和“保留高精度是否值得”。

E004 不应直接混入以下修复策略：

- percentile clipping 实现。
- scale_method 改造。
- per-group calibration。
- mixed precision final deployment。

这些应放到 E005 或 E006，否则实验变量会混乱，无法判断收益来自位置选择还是量化算法变化。

#### 当前推荐的下一步

下一步应执行 E004a：

1. 先实现选择性 activation quantizer 开关工具。
2. 用 2-4 sample smoke 验证：
   - `all_off` 接近 W4A32。
   - `all_on` 接近 A8。
   - 指定 index/name/group 的开关生效。
3. 通过测试和 smoke 后再进入 E004b sentinel 实验。

暂时不要直接运行完整 52 层 sweep；在工具语义没有验证前，完整 sweep 的结果风险较高。

### E004a：选择性 activation quantizer 开关工具

- 日期：2026-05-05
- 负责人：Codex
- 状态：工具实现与 smoke 完成；未运行完整 52 层 sensitivity sweep。

#### 实现内容

- 新增 `quant/activation_sensitivity.py`：
  - 支持按 index、name substring、stage、branch、role、module type 选择 activation quantizer。
  - 支持 `all_on`、`all_off`、`disable_one`、`enable_one`、`disable_group`、`enable_group`。
  - 默认排除最后一个网络输出 activation quantizer，并在 sensitivity 模式下保持其关闭。
  - 使用 context manager 保存并恢复每个 `QuantModule.disable_act_quant`，避免一次评估污染后续模型状态。
  - 为 selected rows 补充 `weight_shape` 和 `activation_numel` proxy 字段；当前 `activation_numel` 先保留为 `null`。
- 新增 `cli/evaluate_activation_sensitivity.py`：
  - 默认 checkpoint 使用 E002c A8 init n=64 / pre-act-recon：
    `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002c_init_sensitivity/quant/20260505_150842_e002c_init_n0064/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
  - 默认 eval 口径沿用 E003a：
    `num_eval_samples=128`、`batch_size=16`、`seed=20260427`。
  - 输出到：
    `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E004_sensitivity/e004a_tool_smoke/`
  - 每次 run 写入 `config.json`、`metrics.json`、`summary.md`、`selected_quantizers.csv`、`per_sample_metrics.jsonl`。
- 新增 `tests/test_activation_sensitivity.py`：
  - 覆盖 selector、默认排除 output quantizer、`all_on/all_off/disable_one/enable_one`、group mode 和状态恢复。

#### Smoke 口径

- checkpoint：E002c A8 init n=64 / pre-act-recon checkpoint。
- eval：4 samples，batch size 2。
- device：CUDA。
- figures：disabled。
- run root：
  `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E004_sensitivity/e004a_tool_smoke/`

#### Smoke 结果

| mode | run | selected | quant SNR mean | quant SNR median | quant SSIM mean |
| --- | --- | ---: | ---: | ---: | ---: |
| `all_on` | `20260505_201035_e004a_smoke_all_on_cuda` | 51 | -9.4503 | -11.7509 | 0.1586 |
| `all_off` | `20260505_201050_e004a_smoke_all_off_cuda` | 51 | 3.2151 | 1.7027 | 0.7134 |
| `disable_one --index 1` | `20260505_201050_e004a_smoke_disable_one_idx1_cuda` | 1 | -9.3920 | -11.7291 | 0.1567 |
| `disable_group --branch transformer` | `20260505_201049_e004a_smoke_disable_transformer_cuda` | 20 | -9.4168 | -11.7108 | 0.1681 |

#### 验收结论

- `all_on` 选中 51 个候选 quantizer，符合默认排除最终输出 activation quantizer 的设计。
- `all_off` 明显恢复到正 SNR，并接近 W4A32 方向，说明开关链路有效。
- `disable_one --index 1` 的 `selected_quantizers.csv` 只有 1 个目标 quantizer。
- `disable_group --branch transformer` 的 `selected_quantizers.csv` 中 20 个目标均为 transformer branch。
- E004a 只验证工具语义和 smoke，不把 4-sample 指标当作正式 sensitivity 结论。

#### 下一步入口

进入 E004b 前，应先选择 8-12 个 sentinel quantizers：

- E001 worst relative MSE 前列。
- E001 lowest effective level 前列。
- stage4/stage5 attention projection。
- attention qkv。
- 若干 CNN Conv2d 对照层。
- 若干 early / late stage 对照层。

E004b 应继续使用 CUDA、多卡 job-level 并行和固定 128-sample eval subset；不要直接运行完整 52 层 sweep。

### E004b：小规模 sentinel 单点关闭敏感性验证

- 日期：2026-05-05
- 负责人：Codex
- 状态：完成 128-sample sentinel 验证；未运行完整 52 层 sweep。

#### 目标

验证 E004a 的选择性开关工具在正式 128-sample eval 口径下，是否能通过关闭少数关键 activation quantizer 产生明显恢复，从而判断完整 sensitivity map 是否值得立即展开。

#### 评估口径

- checkpoint：
  `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002c_init_sensitivity/quant/20260505_150842_e002c_init_n0064/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- eval dataset：
  `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches`
- `num_eval_samples=128`
- `batch_size=16`
- `seed=20260427`
- device：CUDA
- run root：
  `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E004_sensitivity/e004b_sentinel/`

#### GPU 资源说明

- PyTorch 可见 4 张 CUDA 设备：
  - `NVIDIA GeForce RTX 5080` x4
- `nvidia-smi` 在当前 shell 中无法与驱动通信。
- 尝试用 `CUDA_VISIBLE_DEVICES=<id>` 做 job-level 绑定时，子进程内 `torch.cuda.is_available()` 变为 `False`，因此本轮没有使用显式 GPU 绑定。
- 为避免无记录 CPU fallback，本轮仍使用 `--device cuda`，但不绑定单个 GPU。后续如要稳定多卡并行，应先给相关 CLI 增加 `--gpus` 或早期 CUDA 环境设置能力。

#### Sentinel 选择

本轮选择 12 个 sentinel quantizers，覆盖 E001/E002c 诊断中的高风险位置和结构对照：

| index | name | 选择理由 |
| ---: | --- | --- |
| 0 | `model.head` | high outlier / high relative MSE |
| 11 | `model.stage2.0.block.split_proj` | low effective level / per-channel imbalance / fusion |
| 12 | `model.stage2.0.block.merge_proj` | high relative MSE / fusion |
| 15 | `model.stage2.0.block.conv_branch.6` | worst fake-quant MSE / outlier / CNN control |
| 18 | `model.stage2.0.block.trans_branch.mlp.0` | transformer low effective level / outlier |
| 20 | `model.stage2.1` | low effective level / high relative MSE / downsample-like control |
| 24 | `model.stage3.0.block.conv_branch.3` | CNN high MSE / per-channel imbalance |
| 36 | `model.stage4.0.block.trans_branch.attn.qkv` | late transformer attention qkv |
| 37 | `model.stage4.0.block.trans_branch.attn.proj` | late transformer attention proj; E001 final negative-delta related family |
| 46 | `model.stage5.0.block.trans_branch.attn.qkv` | late transformer attention qkv |
| 47 | `model.stage5.0.block.trans_branch.attn.proj` | late transformer attention proj; E001 final negative-delta related family |
| 48 | `model.stage5.0.block.trans_branch.mlp.0` | lowest non-output effective levels / late transformer MLP |

#### Baseline

| mode | selected | SNR mean | SNR median | SSIM mean |
| --- | ---: | ---: | ---: | ---: |
| `all_on` | 51 | -7.1021 | -7.8500 | 0.1945 |
| `all_off` | 51 | 4.7973 | 4.3093 | 0.7049 |

该 baseline 与 E003a 对齐：

- `all_on` 等价于 A8 init n=64，多样本 SNR mean 约 `-7.10 dB`。
- `all_off` 等价于 W4 weight-recon，多样本 SNR mean 约 `4.80 dB`。
- 全部候选 activation quantizer 关闭可恢复约 `+11.90 dB`，再次确认 A8 activation 是主要崩坏来源。

#### Sentinel 单点关闭结果

以 `all_on` 为 baseline，记录 `disable_one` 后的变化：

| index | structure | SNR mean | ΔSNR mean | ΔSNR median | ΔSSIM mean |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20 | stage2 unknown Conv2d | -6.8101 | +0.2920 | +0.3740 | +0.0050 |
| 12 | stage2 fusion merge_proj Conv2d | -6.9704 | +0.1317 | +0.1833 | +0.0025 |
| 0 | head Conv2d | -6.9833 | +0.1188 | -0.0239 | +0.0015 |
| 11 | stage2 fusion split_proj Conv2d | -7.0505 | +0.0515 | +0.0041 | -0.0060 |
| 15 | stage2 cnn conv Conv2d | -7.0750 | +0.0271 | +0.0218 | -0.0027 |
| 24 | stage3 cnn conv Conv2d | -7.0828 | +0.0193 | +0.0417 | +0.0004 |
| 36 | stage4 transformer attention_qkv Linear | -7.0994 | +0.0027 | -0.0168 | -0.0005 |
| 18 | stage2 transformer mlp Linear | -7.1011 | +0.0010 | +0.0171 | -0.0011 |
| 37 | stage4 transformer attention_proj Linear | -7.1013 | +0.0008 | +0.0091 | -0.0001 |
| 48 | stage5 transformer mlp Linear | -7.1028 | -0.0007 | +0.0125 | +0.0003 |
| 46 | stage5 transformer attention_qkv Linear | -7.1035 | -0.0014 | -0.0015 | +0.0005 |
| 47 | stage5 transformer attention_proj Linear | -7.1039 | -0.0018 | +0.0138 | +0.0002 |

#### 结论

E004b 没有发现“关闭单个 sentinel quantizer 即可显著恢复 W4A8”的强信号。

关键观察：

- 最大单点恢复来自 `index=20 model.stage2.1`，SNR mean 只提升约 `+0.292 dB`。
- 第二梯队是 stage2 fusion / head，恢复约 `+0.05` 到 `+0.13 dB`。
- stage4/stage5 attention qkv/proj 的单点关闭几乎没有恢复，变化接近 0。
- 全部 activation quantizer 关闭可恢复约 `+11.90 dB`，但任一 sentinel 单点关闭都远不能解释该差距。

因此，当前更可信的解释是：

- W4A8 崩坏不是由某一个明显的 activation quantizer 单独主导。
- 也不是简单由 stage4/stage5 attention projection 单点主导；这些层更可能与 activation reconstruction 阶段的负 `delta` 有关，而不是 A8 init 崩坏的单点源头。
- A8 崩坏更像多层 activation quantization 误差累积、早中期 Conv/Fusion 传播误差、或全局 range/clipping 策略不适配造成的系统性问题。

#### 对后续 E004 的影响

- 不建议立刻跑完整 52 层 `disable_one` sweep；预期收益有限，因为 sentinel 已覆盖多个高风险结构但恢复都很小。
- 下一步更合理的是做分组关闭 E004c：
  - `branch=fusion`
  - `branch=cnn`
  - `branch=transformer`
  - `module_type=Conv2d`
  - `module_type=Linear`
  - `stage2`
  - `stage1+stage2` 或 early stages
- 如果分组关闭能显著恢复，说明问题是多层/结构组累积；再决定是否做完整 52 层 ranking。
- 如果分组关闭仍不能恢复，则应转向 E005 activation range / clipping / scale_method，而不是继续做单点位置筛选。

### E004d：结构分组关闭敏感性验证

- 日期：2026-05-05
- 负责人：Codex
- 状态：完成 128-sample group disable 验证；未运行完整 52 层单点 sweep。

#### 目标

在 E004b 未发现单点主导层之后，验证 W4A8 A8 init 崩坏是否由某些结构组的 activation quantization 累积导致。

#### 评估口径

- checkpoint：
  `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002c_init_sensitivity/quant/20260505_150842_e002c_init_n0064/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- eval dataset：
  `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches`
- `num_eval_samples=128`
- `batch_size=16`
- `seed=20260427`
- device：CUDA
- run root：
  `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E004_sensitivity/e004d_group/`

Baseline 复用 E004b：

| mode | selected | SNR mean | SNR median | SSIM mean |
| --- | ---: | ---: | ---: | ---: |
| `all_on` | 51 | -7.1021 | -7.8500 | 0.1945 |
| `all_off` | 51 | 4.7973 | 4.3093 | 0.7049 |

#### 分组关闭结果

以 `all_on` 为 baseline，按 `ΔSNR mean` 排序：

| group | selected | SNR mean | ΔSNR mean | ΔSNR median | ΔSSIM mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `module_type=Conv2d` | 31 | 4.5293 | +11.6314 | +11.8975 | +0.3190 |
| `role=unknown` | 5 | -4.2108 | +2.8913 | +3.1173 | +0.0693 |
| `stage=stage5` | 10 | -5.0406 | +2.0615 | +2.1606 | +0.0434 |
| `branch=fusion` | 10 | -5.4061 | +1.6960 | +1.8476 | -0.0412 |
| `role=merge_proj` | 5 | -5.7532 | +1.3489 | +1.4754 | -0.0320 |
| `stage=stage4` | 10 | -6.3032 | +0.7989 | +0.8264 | +0.0144 |
| `stage=stage1` | 10 | -6.3761 | +0.7259 | +0.8040 | +0.0147 |
| `stage=stage2` | 10 | -6.5257 | +0.5764 | +0.5934 | +0.0205 |
| `branch=cnn` / `role=conv` | 15 | -6.6676 | +0.4345 | +0.4457 | +0.0054 |
| `stage=stage3` | 10 | -6.7709 | +0.3312 | +0.3057 | +0.0020 |
| `role=split_proj` | 5 | -6.7977 | +0.3044 | +0.2704 | -0.0140 |
| `role=head` | 1 | -6.9833 | +0.1188 | -0.0239 | +0.0015 |
| `branch=transformer` / `module_type=Linear` | 20 | -7.0812 | +0.0209 | +0.0234 | +0.0079 |
| `role=mlp` | 10 | -7.0892 | +0.0129 | +0.0019 | +0.0002 |
| `role=attention_qkv` | 5 | -7.0964 | +0.0057 | +0.0195 | +0.0077 |
| `role=tail` | 1 | -7.1021 | +0.0000 | +0.0000 | +0.0000 |
| `role=attention_proj` | 5 | -7.1075 | -0.0054 | -0.0369 | -0.0009 |

#### 关键结构定位

1. Conv2d activation quantization 是当前 A8 init 崩坏的主导问题。

   - 关闭全部 Conv2d activation quantizers 后，SNR mean 从 `-7.1021 dB` 恢复到 `4.5293 dB`。
   - 这已经接近 all_off / W4 weight-recon 的 `4.7973 dB`。
   - 说明 31 个 Conv2d activation quantizers 几乎解释了全 A8 崩坏的大部分损失。

2. Transformer / Linear 不是当前 A8 init 崩坏主因。

   - 关闭 transformer branch 或全部 Linear activation quantizers 只恢复约 `+0.0209 dB`。
   - attention qkv / attention proj / MLP role 关闭后也几乎没有恢复。
   - 因此，E001 中 stage4/stage5 attention projection 的负 `delta` 更像 activation reconstruction 阶段的合法性/优化问题，不是 A8 init 崩坏的主来源。

3. Fusion 和 stage-level Conv2d 结构值得继续细分。

   - `branch=fusion` 恢复 `+1.6960 dB`。
   - `role=merge_proj` 恢复 `+1.3489 dB`，明显高于 `split_proj` 的 `+0.3044 dB`。
   - `role=unknown` 包含 `model.stage1.1` 到 `model.stage5.1` 五个 Conv2d stage transition / downsample-like modules，恢复 `+2.8913 dB`。
   - `stage5` 恢复 `+2.0615 dB`，比 stage1-stage4 更强。

4. CNN branch alone 不是全部 Conv2d 问题。

   - `branch=cnn` / `role=conv` 只恢复 `+0.4345 dB`。
   - 大部分 Conv2d 贡献来自 fusion、unknown stage modules、head 以及各 stage 的组合，而不是单纯 CNN branch。

#### 结论

E004d 给出了比 E004b 更强的定位结论：

- W4A8 A8 init 崩坏主要由 Conv2d activation quantization 引起。
- 这不是单点层主导，而是 Conv2d 结构组的累积误差。
- Transformer / Linear / attention qkv/proj 在 A8 init 崩坏中影响很小，不应作为下一步主修复对象。
- 后续修复方向应从“attention/Linear 特殊处理”转向“Conv2d activation range / clipping / calibration 策略”，尤其关注：
  - merge projection
  - stage transition / downsample-like Conv2d modules
  - stage5 Conv2d activations
  - fusion branch

#### 对后续实验的影响

不建议继续做完整 52 层单点关闭 sweep作为下一步主线。更合理的下一阶段是 E005：

- 对 Conv2d activation quantizers 做 `scale_method=max/max_scale/mse` 对照。
- 对 Conv2d activation 做 percentile / outlier clipping calibration。
- 优先在 fusion / merge_proj / unknown stage transition / stage5 上试局部 clipping。
- 保留 Linear / transformer 当前策略作为对照，避免引入无效变量。

如果仍要在 E004 内继续，应只做更细的 Conv2d group 切分，而不是全层单点 sweep。

### E004 资源修正：支持显式 CUDA device index

- 日期：2026-05-05
- 负责人：用户 / Codex
- 状态：完成工具修正，后续可避开 0 卡运行 E004/E005 eval。

#### 背景

E004d 期间发现：

- 不绑定 GPU 时，PyTorch 能看到 4 张 `NVIDIA GeForce RTX 5080`。
- `nvidia-smi` 在当前 shell 中无法与驱动通信。
- 使用外部 `CUDA_VISIBLE_DEVICES=<id>` 绑定时，子进程内 `torch.cuda.is_available()` 会变为 `False`。
- 因此，不能依赖 `CUDA_VISIBLE_DEVICES` 做 job-level GPU 绑定。

用户指出 0 卡已有其他任务占用，因此后续实验需要能显式选择非 0 卡。

#### 修改策略

新增内部 CUDA device index 选择，而不是修改 `CUDA_VISIBLE_DEVICES`：

- `evaluate_activation_sensitivity.py` 新增：
  - `--cuda-device-index`
- `evaluate_quantized_scrn_multi.py` 新增：
  - `--cuda-device-index`

使用方式：

```bash
--device cuda --cuda-device-index 1
```

这会直接选择 `torch.device("cuda:1")`，不改变进程可见 GPU 列表。

#### Smoke 验证

命令口径：

- CLI：`evaluate_activation_sensitivity.py`
- mode：`all_on`
- samples：2
- batch size：1
- device：`cuda`
- cuda device index：1
- run：
  `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E004_sensitivity/cuda_device_index_smoke/20260505_205010_e004_cuda_index_1_smoke`

结果：

- run 成功完成。
- `config.json` 记录：
  - `device`: `cuda:1`
  - `cuda_device_index`: `1`
- SNR mean：`-6.8132 dB`
- SSIM mean：`0.1401`
- run 产物位于 `.gitignore` 保护的 `runs/activation_quantization/*` 下，不纳入 Git。

#### 后续使用原则

- 如果 0 卡被占用，后续 E004/E005 eval 优先使用：

```bash
--device cuda --cuda-device-index 1
```

或根据可用情况改为 `2/3`。

- 这仍然是“单个 run 单卡”，但可以让多个独立 run 分别指定 `cuda:1/2/3` 并行执行。
- 后续若要稳定多进程并行，需要在调度脚本中按 run 分配不同 `--cuda-device-index`，不要再用 `CUDA_VISIBLE_DEVICES=<id>`。

### E004 后续路线重规划：从全层 sweep 转向 Conv2d 子组定位

- 日期：2026-05-05
- 负责人：用户 / Codex
- 状态：规划完成，准备执行 E004e。

#### 重新判断

结合 E001-E003 与 E004b/E004d，E004 的重点已经发生变化。

原始 E004 设想是通过单点开启/关闭 52 个 activation quantizers，寻找最敏感层。但当前证据显示：

- E004b 的 12 个 sentinel 单点关闭中，最大恢复只有 `+0.2920 dB`。
- E004d 关闭全部 `module_type=Conv2d` activation quantizers 后，SNR mean 从 `-7.1021 dB` 恢复到 `4.5293 dB`，恢复 `+11.6314 dB`。
- E004d 关闭全部 `module_type=Linear` 或 `branch=transformer` 后，只恢复 `+0.0209 dB`。

因此，当前 W4A8 A8 init 崩坏不是单个 quantizer 主导，也不是 transformer/Linear 主导，而是 Conv2d activation quantization 的结构组累积误差。

这意味着：E004 后续不应继续把完整 52 层单点关闭 sweep 作为主线。完整 sweep 成本高，而且大概率只会重复 E004b 的结论：单层效应弱、组效应强。

#### E004e：Conv2d 子组关闭细分

目标：把 E004d 的“Conv2d 是主因”拆成更可操作的结构结论。

固定口径：

- checkpoint：继续使用 A8 init n=64 / pre-act-recon checkpoint。
- eval：128 samples，seed `20260427`，batch size 16。
- baseline：沿用 E004b/E004d 的 `all_on` 与 `all_off`。
- device：优先 CUDA，使用 `--device cuda --cuda-device-index 1/2/3` 避开 0 卡。

建议测试的 Conv2d 子组：

- `stage=stage1/2/3/4/5 + module_type=Conv2d`
- `branch=fusion + module_type=Conv2d`
- `branch=cnn + module_type=Conv2d`
- `role=split_proj`
- `role=merge_proj`
- `role=unknown`
- `name contains=head`
- `stage5 Conv2d`
- stage transition / downsample-like Conv2d modules，即当前 `role=unknown` 的五个模块。

记录指标：

- selected quantizer count
- selected quantizer names
- SNR mean / median / min / max
- SSIM mean
- `delta_snr_mean` vs `all_on`
- recovery ratio：`delta_snr_mean / (all_off_snr_mean - all_on_snr_mean)`

验收标准：

- 如果少数 Conv2d 子组解释大部分恢复，E005 优先对这些子组做 range / clipping / calibration。
- 如果所有 Conv2d 子组都只能恢复一小段，而全部 Conv2d 关闭才恢复，说明是 Conv2d activation 量化的全局累积问题，E005 应优先改 Conv2d activation range 策略，而不是只挑层保留 FP32。

#### E004f：Conv2d-only reopen / leave-one-out 验证

目标：验证 E004e 找到的 Conv2d 子组是否真的决定恢复能力。

建议优先做“从 Conv2d 全关状态重新打开某些子组”，而不是继续从 all-on 状态单点关闭。

原因：

- E004b 已经说明单点关闭效果很弱。
- E004d 说明全部 Conv2d 关闭几乎恢复 W4A32。
- 从“全部 Conv2d 关闭”的强恢复状态出发，重新开启某个 Conv2d 子组，可以更清楚地看到哪个子组一量化就把 SNR 拉低。

候选实验：

- 全部 Conv2d 关闭，只重新开启 `merge_proj`。
- 全部 Conv2d 关闭，只重新开启 `role=unknown`。
- 全部 Conv2d 关闭，只重新开启 `stage5 Conv2d`。
- 全部 Conv2d 关闭，只重新开启 `fusion Conv2d`。
- 全部 Conv2d 关闭，只重新开启 `cnn branch Conv2d` 作为对照。

如果当前工具暂不支持“全关 Conv2d 后重新开启子组”的组合模式，可先记录为 E004f-tooling 或 E005 前置工具需求，不强行用现有 `enable_group` 语义替代。

#### E004g：策略表收束

目标：把 E004 变成后续 E005/E006 的输入，而不是无限延长 sensitivity 实验。

需要输出一张策略表，至少包含：

- 结构组
- quantizer 数量
- 关闭后 SNR 恢复
- recovery ratio
- activation-volume proxy
- 是否建议优先修复
- 候选策略

预期策略方向：

- 优先修复：
  - fusion Conv2d
  - merge projection
  - stage transition / downsample-like Conv2d
  - stage5 Conv2d
- 暂不主攻：
  - Linear / transformer / attention qkv/proj 的 A8 init
- 保留对照：
  - Linear / transformer 仍作为后续 E005/E006 的 sanity check，避免修复 Conv2d 后引入新的 transformer 劣化。

#### E004 与 E005 的边界

E004 只回答“哪里敏感、哪里值得保留高精度或单独处理”。

E004 不直接修改量化算法，也不直接引入：

- percentile clipping
- MSE scale calibration
- per-channel activation quantization
- mixed precision deployment 策略
- Conv2d 专用 quantizer 参数化

这些修复实验应进入 E005。

#### 当前下一步

下一步执行 E004e。

E004e 应先做 Conv2d 子组关闭细分，而不是完整 52 层单点关闭 sweep。若 E004e 形成明确子组 ranking，再决定是否需要 E004f reopen / leave-one-out；如果 E004e 已足够清楚，可直接进入 E004g 策略表并转入 E005。

### E004e：Conv2d 子组关闭细分

- 日期：2026-05-05
- 负责人：用户 / Codex
- 状态：完成。

#### 实验目的

E004d 已经证明 W4A8 A8 init 崩坏主要来自 Conv2d activation quantization 的结构组累积误差。

E004e 的目的不是再次证明 Conv2d 有问题，而是把 Conv2d 进一步拆开，回答：

- 哪些 Conv2d 子组最值得优先修复？
- 这个问题是否由某个 stage、fusion projection、stage transition / downsample-like module 主导？
- 是否还需要继续完整 52 层单点 sweep？

#### 实验口径

- checkpoint：A8 init n=64 / pre-act-recon checkpoint。
- eval：128 samples。
- seed：`20260427`。
- batch size：16。
- device：CUDA，使用 `--cuda-device-index 1/2/3` 分批并行，避开 0 卡。
- run root：
  `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E004_sensitivity/e004e_conv2d_subgroups/`
- 所有 run 产物位于 `.gitignore` 保护的 `runs/activation_quantization/*` 下，不纳入 Git。

Baseline：

- `all_on`：SNR mean `-7.1021 dB`，SSIM mean `0.1945`。
- `all_off`：SNR mean `4.7973 dB`，SSIM mean `0.7049`。
- 可恢复总 gap：`11.8993 dB`。

#### 结果表

`delta_snr_mean` 均相对 `all_on` 计算；`recovery` 为相对 `all_on -> all_off` gap 的恢复比例。

| group | selected | SNR mean | SNR median | delta SNR | recovery | SSIM mean |
|---|---:|---:|---:|---:|---:|---:|
| all_off | 51 | 4.7973 | 4.3093 | +11.8993 | 100.0% | 0.7049 |
| `role=unknown + Conv2d` | 5 | -4.2108 | -4.7328 | +2.8913 | 24.3% | 0.2638 |
| `stage5 + Conv2d` | 6 | -5.0410 | -5.6906 | +2.0611 | 17.3% | 0.2378 |
| `branch=fusion + Conv2d` | 10 | -5.4061 | -6.0024 | +1.6960 | 14.3% | 0.1533 |
| `role=merge_proj + Conv2d` | 5 | -5.7532 | -6.3747 | +1.3489 | 11.3% | 0.1626 |
| `model.stage5.1` | 1 | -6.0764 | -6.7730 | +1.0256 | 8.6% | 0.2114 |
| `stage4 + Conv2d` | 6 | -6.3013 | -7.0370 | +0.8008 | 6.7% | 0.2091 |
| `stage1 + Conv2d` | 6 | -6.3991 | -7.1354 | +0.7030 | 5.9% | 0.2024 |
| `stage2 + Conv2d` | 6 | -6.5327 | -7.2929 | +0.5693 | 4.8% | 0.2140 |
| `branch=cnn + Conv2d` | 15 | -6.6676 | -7.4043 | +0.4345 | 3.7% | 0.1999 |
| `model.stage1.1` | 1 | -6.7106 | -7.4556 | +0.3915 | 3.3% | 0.2084 |
| `model.stage4.1` | 1 | -6.7239 | -7.4885 | +0.3782 | 3.2% | 0.1983 |
| `stage3 + Conv2d` | 6 | -6.7667 | -7.5072 | +0.3354 | 2.8% | 0.1968 |
| `role=split_proj + Conv2d` | 5 | -6.7977 | -7.5796 | +0.3044 | 2.6% | 0.1805 |
| `model.stage2.1` | 1 | -6.8101 | -7.4760 | +0.2920 | 2.5% | 0.1995 |
| `model.stage3.1` | 1 | -6.9486 | -7.6839 | +0.1535 | 1.3% | 0.1942 |
| `role=head + Conv2d` | 1 | -6.9833 | -7.8740 | +0.1188 | 1.0% | 0.1961 |
| all_on | 51 | -7.1021 | -7.8500 | +0.0000 | 0.0% | 0.1945 |

#### 关键子组内容

`role=unknown + Conv2d` 包含：

- `model.stage1.1`
- `model.stage2.1`
- `model.stage3.1`
- `model.stage4.1`
- `model.stage5.1`

该组是 E004e 最强子组，关闭 5 个 quantizers 即恢复 `+2.8913 dB`，占全部可恢复 gap 的 `24.3%`。

`stage5 + Conv2d` 包含：

- `model.stage5.0.block.split_proj`
- `model.stage5.0.block.merge_proj`
- `model.stage5.0.block.conv_branch.0`
- `model.stage5.0.block.conv_branch.3`
- `model.stage5.0.block.conv_branch.6`
- `model.stage5.1`

该组恢复 `+2.0611 dB`，其中单独关闭 `model.stage5.1` 恢复 `+1.0256 dB`，说明 stage5 的 stage transition / downsample-like Conv2d 是一个强敏感点。

`branch=fusion + Conv2d` 包含 5 个 `split_proj` 和 5 个 `merge_proj`。

- fusion overall 恢复 `+1.6960 dB`。
- `merge_proj` 恢复 `+1.3489 dB`。
- `split_proj` 只恢复 `+0.3044 dB`。

因此 fusion 中的主要问题更偏向 merge projection，而不是 split projection。

#### 结论

E004e 进一步确认：Conv2d activation quantization 的问题不是普通 CNN branch 单独主导。

更具体地说：

1. 最优先关注 `role=unknown` 的 stage transition / downsample-like Conv2d modules。

   - 5 个模块恢复 `+2.8913 dB`。
   - 单个最强是 `model.stage5.1`，恢复 `+1.0256 dB`。

2. stage5 Conv2d 是最敏感 stage。

   - `stage5 + Conv2d` 恢复 `+2.0611 dB`，明显强于 stage1-stage4。
   - stage5 既包含 late-stage fusion/conv branch，又包含强敏感的 `stage5.1`。

3. fusion branch 中 `merge_proj` 明显比 `split_proj` 更敏感。

   - `merge_proj` 恢复 `+1.3489 dB`。
   - `split_proj` 恢复 `+0.3044 dB`。

4. `branch=cnn + Conv2d` 不是主因。

   - 关闭 15 个 CNN branch Conv2d 只恢复 `+0.4345 dB`。
   - 说明问题更集中在 fusion/merge/stage transition/stage5，而不是普通 CNN conv branch。

5. 仍然不存在单点完全主导。

   - 单点最强 `model.stage5.1` 只恢复 `+1.0256 dB`。
   - 全部 Conv2d 关闭恢复 `+11.6314 dB`。
   - 说明当前 A8 init 崩坏依然是 Conv2d 多点累积误差，而不是单个层可以解释。

#### 对后续路线的影响

E004e 已经足够支持进入 E005，不建议再做完整 52 层单点关闭 sweep。

如果继续在 E004 内补充，应只做一个很小的 E004f 工具/验证：

- 从“全部 Conv2d 关闭”的强恢复状态出发，重新开启 `role=unknown`、`stage5`、`merge_proj`、`fusion` 等子组。
- 观察哪个子组重新 A8 化后会显著拉低 SNR。
- 当前工具尚不支持“Conv2d 全关后 reopen 子组”的组合语义，因此这属于 E004f-tooling，而不是必须立即完成的实验。

下一阶段主线建议进入 E005：

- 优先对 `role=unknown`、`stage5`、`merge_proj`、`fusion` 的 Conv2d activation 做 range / clipping / calibration 策略。
- `branch=cnn` 和 Linear / transformer 继续作为对照，不作为第一优先修复对象。

### E004f 取舍决策：不按完整 reopen 计划继续，先做 E004g 收束

- 日期：2026-05-05
- 负责人：用户 / Codex
- 状态：决策完成，不执行完整 E004f。

#### 原计划 E004f

E004f 原本考虑做 Conv2d-only reopen / leave-one-out：

- 从“全部 Conv2d activation quantizers 关闭”的强恢复状态出发。
- 重新开启 `role=unknown`、`stage5`、`merge_proj`、`fusion` 等 Conv2d 子组。
- 观察哪个子组重新 A8 化后最明显拉低 SNR。

#### 为什么不作为主线继续

结合 E004b/E004d/E004e，当前证据链已经足够清楚：

- E004b：sentinel 单点关闭最大只恢复 `+0.2920 dB`，没有单点主导层。
- E004d：关闭全部 Conv2d activation quantizers 恢复 `+11.6314 dB`，而关闭 Linear / transformer 只恢复 `+0.0209 dB`。
- E004e：Conv2d 内部的 `role=unknown`、`stage5`、`fusion`、`merge_proj` 更敏感，但任何单个子组都远不能解释全部恢复。

因此，E004 已经回答了核心问题：

- 当前 A8 init 崩坏主要是 Conv2d activation quantization 的系统性/累积性误差。
- 更敏感位置是 stage transition / downsample-like modules、stage5、fusion 和 merge projection。
- 不是某个单层或单个子结构独立导致失败。

完整 E004f 的边际收益有限，且当前 E004a 工具尚不支持“全部 Conv2d 关闭后重新开启某个子组”的组合语义。若严谨实现，需要新增组合 selector / mode，这会把工作重心从修复问题转回工具扩展。

#### 当前决策

- 不执行完整 E004f。
- E004f 只作为可选补充保留：
  - 如果后续论文/报告需要更强的反事实证据，再实现 `disable_group + reopen_group` 组合模式。
  - 只需跑 3-4 个关键组合：`unknown`、`stage5`、`merge_proj`、`fusion`。
- 当前主线改为：
  1. E004g：汇总 E004 sensitivity vs resource benefit 策略表。
  2. E005：开始 Conv2d activation range / clipping / calibration 修复实验。

#### 进入 E005 前需要保留的注意事项

- E005 第一优先级不应是 transformer / Linear / attention qkv/proj。
- E005 第一优先级应是 Conv2d activation 的 range 选择问题。
- E005 重点结构：
  - `role=unknown` stage transition / downsample-like Conv2d
  - `stage5 Conv2d`
  - `merge_proj`
  - `fusion Conv2d`
- E005 每次修复都必须用 E003a 的 128-sample eval 口径复核，而不能回到单样本 SNR。
