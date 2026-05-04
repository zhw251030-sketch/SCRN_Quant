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
| A1 | Activation `delta` 缺少正值约束，优化后出现非法 scale | final W4A8 checkpoint 有 2 个负 delta | 最高 | 已观察，待修复验证 |
| A2 | Transformer/Swin attention 分支存在 activation outlier，tensor-wise A8 被极值主导 | 负 delta 位于 stage4/stage5 attention proj | 高 | 待统计分布 |
| A3 | 只优化 `delta`，`zero_point` 固定导致 asymmetric range 不稳定 | 当前 activation reconstruction 只收集 delta 参数 | 高 | 待验证 |
| A4 | Activation reconstruction 局部 MSE 目标与最终 SNR/SSIM 不一致 | A8 init 后大幅掉点，act recon 只恢复约 0.207 dB | 高 | 待验证 |
| A5 | Calibration 数据覆盖不足或与 eval/部署输入分布不一致 | activation scale 依赖输入分布 | 中 | 待验证 |
| A6 | 全网 A8 一刀切，敏感层需要混合精度或跳过 activation quant | stage4/stage5 attention proj 已暴露异常 | 中 | 待验证 |
| A7 | 激活量化插入位置和 SCRN 实际部署边界不完全一致 | block-level act quantizer 当前不参与 forward | 中 | 待审查 |
| A8 | Activation 初始化只用 `init_batch_size=64`，可能不足以覆盖深层 activation range | `zero_point` 固定，初始化样本对最终范围影响大 | 高 | 待验证 |
| A9 | Activation reconstruction 使用 `asym=False`，可能没有模拟前序量化误差累积 | 代码中 activation 阶段硬编码 `asym=False` | 高 | 待验证 |
| A10 | 教师模型使用 FP32 可能与 W4A8 阶段目标不匹配 | 激活量化叠加在 W4 权重模型之后 | 中 | 待验证 |
| A11 | 当前量化器位置过密，可能在 FFB 内部多次量化造成误差累积 | QuantModel 递归替换每个 Conv/Linear | 高 | 待绘制位置图 |
| A12 | 当前 2-8 bit 限制不支持直接 A16 fallback | `UniformAffineQuantizer` 限制 `2 <= n_bits <= 8` | 中 | 待设计 fallback |
| A13 | Activation LR 可能过大，导致 scale 参数越界或震荡 | `activation_lr=4e-4`，最终出现负 delta | 中 | 待 sweep |

## 实验索引

| 实验编号 | 日期 | 目的 | 主要改动或配置 | 关键结果 | 结论 |
| --- | --- | --- | --- | --- | --- |
| E000 | 2026-05-04 | 建立激活量化研究日志 | 新增本文档 | 不涉及实验 | 后续实验从 E001 开始记录 |
| A000 | 2026-05-04 | 整理用户初步总结并形成实验路线 | 追加 Codex 深度分析 | 不涉及运行 | 后续实验按 E001-E006 推进 |
| P000 | 2026-05-04 | 正式实验前建立目录规范 | 新增激活量化配置目录和运行产物目录 | 不涉及运行 | 后续配置和实验产物分开存放 |

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
