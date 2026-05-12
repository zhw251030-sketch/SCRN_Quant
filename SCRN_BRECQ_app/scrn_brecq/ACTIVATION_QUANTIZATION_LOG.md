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

### E004g：E004 策略表收束与原目标完成度

- 日期：2026-05-05
- 负责人：用户 / Codex
- 状态：完成。

#### 收束目标

E004g 的目标是停止继续扩展 sensitivity 实验，把 E004b/E004d/E004e 已有证据转化为 E005/E006 的执行输入。

当前使用的基准：

- `all_on`：SNR mean `-7.1021 dB`，SSIM mean `0.1945`。
- `all_off`：SNR mean `4.7973 dB`，SSIM mean `0.7049`。
- `all_on -> all_off` gap：`11.8993 dB`。

注意：E004a 当前 CSV 中 `activation_numel` 仍为空，因此 E004g 的 resource benefit 不能声称是真实 runtime activation-volume。这里使用三个粗代理：

- `selected_count`：该组关闭的 activation quantizer 数量。
- `count_share`：`selected_count / 51`，51 是默认排除 output quantizer 后的 E004 activation sensitivity candidate 数。
- `benefit_per_quantizer`：`delta_snr_mean / selected_count`，只作为“单位 quantizer 粗收益”，不是硬件内存收益。

后续若需要严格 memory / activation-volume ranking，需要新增 hook 采集每个 quantizer 的 runtime output shape / numel；这不是当前 E004 已完成工具的能力。

#### E004g 策略表

| 结构组 | selected count | count share | delta SNR | recovery ratio | benefit / quantizer | 是否优先修复 | 候选策略 |
|---|---:|---:|---:|---:|---:|---|---|
| all Conv2d | 31 | 60.8% | +11.6314 | 97.7% | +0.3752 | 是，主方向 | Conv2d activation range / clipping / calibration 总策略 |
| `role=unknown` stage transition / downsample-like Conv2d | 5 | 9.8% | +2.8913 | 24.3% | +0.5783 | 是，最高优先级 | percentile clipping、MSE range、局部 high precision 对照 |
| `stage5 + Conv2d` | 6 | 11.8% | +2.0611 | 17.3% | +0.3435 | 是，高优先级 | stage5 Conv2d clipping / scale_method 对照 |
| `branch=fusion + Conv2d` | 10 | 19.6% | +1.6960 | 14.3% | +0.1696 | 是，高优先级 | fusion Conv2d range calibration |
| `role=merge_proj + Conv2d` | 5 | 9.8% | +1.3489 | 11.3% | +0.2698 | 是，高优先级 | merge_proj 单独 clipping / calibration |
| `model.stage5.1` | 1 | 2.0% | +1.0256 | 8.6% | +1.0256 | 是，单点强对照 | stage5.1 FP32 / clipping ablation |
| `stage4 + Conv2d` | 6 | 11.8% | +0.8008 | 6.7% | +0.1335 | 中 | 作为 late-stage 对照 |
| `stage1 + Conv2d` | 6 | 11.8% | +0.7030 | 5.9% | +0.1172 | 中 | early-stage 对照 |
| `stage2 + Conv2d` | 6 | 11.8% | +0.5693 | 4.8% | +0.0949 | 中低 | stage 对照 |
| `branch=cnn + Conv2d` | 15 | 29.4% | +0.4345 | 3.7% | +0.0290 | 暂不主攻 | 保留为 sanity check |
| `role=split_proj + Conv2d` | 5 | 9.8% | +0.3044 | 2.6% | +0.0609 | 暂不主攻 | fusion 内部低优先级对照 |
| `role=head + Conv2d` | 1 | 2.0% | +0.1188 | 1.0% | +0.1188 | 暂不主攻 | 输入头部对照 |
| all Linear / transformer | 20 | 39.2% | +0.0209 | 0.2% | +0.0010 | 不作为 A8 init 主修复 | 仅做后续 sanity check |
| attention qkv/proj / MLP | 5-10 | 9.8%-19.6% | 约 0 | 约 0 | 约 0 | 不作为 A8 init 主修复 | 只在 reconstruction 或 E006 中复查 |

#### sensitivity ranking

按 `delta_snr_mean` 排名，E004 当前最重要的结构顺序为：

1. all Conv2d：`+11.6314 dB`
2. `role=unknown` stage transition / downsample-like Conv2d：`+2.8913 dB`
3. `stage5 + Conv2d`：`+2.0611 dB`
4. `branch=fusion + Conv2d`：`+1.6960 dB`
5. `role=merge_proj + Conv2d`：`+1.3489 dB`
6. `model.stage5.1`：`+1.0256 dB`
7. `branch=cnn + Conv2d`：`+0.4345 dB`
8. Linear / transformer groups：约 `+0.0209 dB` 或更低

这个 ranking 说明：问题首先是 Conv2d activation quantization 的系统性累积误差，其次才是在 Conv2d 内部定位更敏感子组。

#### resource-benefit 粗 ranking

按 `benefit_per_quantizer` 粗排：

1. `model.stage5.1`：`+1.0256 dB / quantizer`
2. `role=unknown`：`+0.5783 dB / quantizer`
3. all Conv2d：`+0.3752 dB / quantizer`
4. `stage5 + Conv2d`：`+0.3435 dB / quantizer`
5. `merge_proj`：`+0.2698 dB / quantizer`
6. `fusion`：`+0.1696 dB / quantizer`
7. `cnn branch`：`+0.0290 dB / quantizer`
8. Linear / transformer：约 `+0.0010 dB / quantizer`

这不是精确 memory ranking，但足以说明 E005 如果要做小范围策略，`stage5.1`、`role=unknown`、`stage5`、`merge_proj` 的收益密度更高。

#### 对 E005/E006 的输入

E005 第一阶段建议：

- 不从 transformer / Linear 开始。
- 不从完整 mixed precision 搜索开始。
- 先做 Conv2d activation range / clipping / calibration。
- 第一批重点结构：
  - `role=unknown`
  - `stage5 Conv2d`
  - `merge_proj`
  - `fusion Conv2d`
  - `model.stage5.1`

E005 的候选策略：

- Conv2d-only percentile clipping calibration。
- Conv2d-only MSE range calibration。
- 对 `role=unknown` / `stage5` / `merge_proj` 采用更保守 clipping。
- 保持 Linear / transformer A8 策略不变作为 sanity check。
- 每次结果必须用 E003a 128-sample eval 口径对比 all_on baseline，不以单样本 SNR 做结论。

E006 如果后续需要部署策略，可以基于 E004g 表做 mixed precision / selective FP32：

- 首先考虑只保留 `stage5.1` 或 `role=unknown` 为 FP32 的小策略。
- 再考虑 `stage5 + merge_proj` 等组合。
- 不建议优先保留全 transformer / Linear FP32，因为 E004 对 A8 init 崩坏几乎没有支持。

#### E004 原始目标完成度检查

原始 E004 目标：

> 知道哪些 activation quantizer 最该保留高精度或单独处理。

完成状态：基本完成。

- 已明确：最该单独处理的是 Conv2d activation quantization，特别是 `role=unknown`、`stage5`、`fusion`、`merge_proj` 和 `model.stage5.1`。
- 已明确：Linear / transformer / attention qkv/proj 不是当前 A8 init 崩坏主因。

原始实验要求：

1. 单点开启或单点关闭 activation quantizer。

   - 已完成单点关闭 sentinel：E004b。
   - 已补充若干 Conv2d 单点：E004e 中的 `head`、`stage1.1` 到 `stage5.1`。
   - 未做完整 52 层单点关闭 sweep，原因是 E004b/E004d/E004e 已显示单点不是主导，完整 sweep 边际收益低。
   - 未做完整单点开启 sweep，原因是当前问题是累积误差，不适合作为主线。

2. 按 stage / branch / module type 分组开启或关闭。

   - 已完成分组关闭：E004d/E004e。
   - 覆盖 module type、branch、stage、role、Conv2d 子组。
   - 未做复杂组合 reopen，已决定不作为当前主线。

3. 记录最终 SNR / SSIM 变化和结构位置。

   - 已完成。
   - E004b/E004d/E004e 每个 run 均保存 `metrics.json`、`selected_quantizers.csv`、`per_sample_metrics.jsonl` 和 `summary.md`。
   - 日志已汇总关键 SNR / SSIM / selected names。

原始输出要求：

1. sensitivity ranking。

   - 已完成，见 E004g sensitivity ranking。

2. memory / activation-volume proxy ranking。

   - 部分完成。
   - 当前使用 `selected_count`、`count_share`、`benefit_per_quantizer` 作为粗 proxy。
   - 精确 runtime activation-volume 尚未完成，因为 E004a 当前未采集 `activation_numel`。
   - 若后续论文/部署需要严谨资源排名，需要新增 forward hook 采集 runtime activation shape / numel。

3. sensitivity vs resource benefit 二维图或表。

   - 表格已完成，见 E004g 策略表。
   - 二维图未生成，因为当前 proxy 是粗代理，生成图的价值不高；如果后续补齐精确 activation-volume，可再生成正式图。

#### E004 总结论

E004 可以收束。

当前最可信结论是：

- W4A8 A8 init 崩坏主要来自 Conv2d activation quantization 的系统性/累积性误差。
- 不是单点层主导。
- 不是 transformer / Linear / attention qkv/proj 主导。
- Conv2d 内部最值得优先处理的是：
  - stage transition / downsample-like modules (`role=unknown`)
  - stage5 Conv2d
  - fusion Conv2d
  - merge projection
  - `model.stage5.1`

下一阶段应进入 E005，直接验证 Conv2d activation range / clipping / calibration 是否能恢复 128-sample W4A8 SNR。

### E005 计划重构：Conv2d activation 离群值、范围和粒度实验

- 日期：2026-05-06
- 负责人：用户 / Codex
- 状态：计划完成，尚未执行。

#### 背景与问题重定义

原始 E005 目标是：

> 验证 tensor-wise A8 是否被 outlier 和通道差异主导。

在 E004 完成后，E005 的问题需要进一步具体化：

- 当前主瓶颈不是 activation `delta` 负值；E002a/E002b 已修复合法性，但多样本 SNR 没有恢复。
- 当前主瓶颈不是 transformer / Linear / attention qkv/proj 的 A8 init；E004d 显示关闭 Linear / transformer 只恢复约 `+0.0209 dB`。
- 当前主瓶颈是 Conv2d activation quantization 的系统性/累积性误差；关闭全部 Conv2d activation quantizers 恢复 `+11.6314 dB`。
- Conv2d 内部最值得优先处理的是 `role=unknown`、`stage5`、`fusion`、`merge_proj` 和 `model.stage5.1`。

因此，E005 应从“泛泛验证 tensor-wise A8 是否受 outlier 影响”调整为：

> 验证当前 Conv2d tensor-wise A8 range 初始化是否被 outlier / channel imbalance / stage-specific activation 分布主导，并判断 percentile / MSE clipping 或更细粒度策略能否恢复 128-sample W4A8 SNR。

#### E005 在整个激活量化实验中的作用

E001-E004 已经完成“定位问题”：

- E001：建立 activation diagnostics，确认负 `delta` 和局部误差现象。
- E002：修复正 scale 合法性，但证明这不是主要精度瓶颈。
- E003：建立 128-sample multi-sample eval，证明 A8 init 本身已经崩坏。
- E004：定位到 Conv2d activation quantization 的结构组累积误差。

E005 是从“定位”进入“修复”的第一阶段。

它要回答三个核心问题：

1. Conv2d A8 崩坏是否主要由 min/max range 被 outlier 拉宽导致？
2. 如果使用更稳健的 range calibration，W4A8 A8 init 是否能从约 `-7 dB` 明显恢复？
3. 修复是否需要全局 Conv2d 策略，还是只需要针对 `unknown/stage5/fusion/merge_proj` 的结构化局部策略？

如果 E005 中 percentile / MSE clipping 能显著恢复 SNR，说明后续创新点可以集中在 SCRN 结构感知 activation range calibration。

如果 E005 仍无法恢复，说明问题可能不是简单 outlier range，而需要进入 E006 方向，例如 activation per-channel / group-wise / structure-wise quantization 或 mixed precision。

#### 固定评估口径

E005 所有正式结论必须沿用 E003a/E004 的多样本口径：

- checkpoint 起点：E002c A8 init n=64 / pre-act-recon 或同等 W4 weight-recon 起点。
- eval：128 samples。
- seed：`20260427`。
- batch size：16。
- device：CUDA，优先 `--cuda-device-index 1/2/3`。
- baseline：
  - all_on A8 init：SNR mean `-7.1021 dB`。
  - all_off / W4A32 近似：SNR mean `4.7973 dB`。
  - E005 的目标是让 all_on A8 init 明显远离 `-7 dB`，向 W4A32 靠近。

不得用单样本 SNR 作为 E005 成败结论。

#### E005a：Conv2d range diagnostics 增强

目的：在修复前先确认 Conv2d activation 的 range 问题形态。

应做内容：

- 基于 E001 diagnostics 或新增轻量统计，重点输出 Conv2d 子组：
  - min / max
  - p99 / p99.9 / p99.99
  - absmax / p99
  - absmax / p99.9
  - per-channel absmax ratio
  - fake-quant MSE / relative MSE
  - effective int levels
- 分组汇总：
  - all Conv2d
  - `role=unknown`
  - `stage5`
  - `fusion`
  - `merge_proj`
  - `cnn branch`
  - Linear / transformer sanity check

预期结果：

- 如果 outlier 主导，敏感 Conv2d 子组应表现出高 `absmax/p99` 或 `absmax/p99.9`。
- 如果 channel imbalance 主导，敏感 Conv2d 子组应表现出高 `per_channel_absmax_ratio`。
- 如果两者都不明显，E005 后续应更谨慎，避免盲目 clipping。

验收标准：

- 明确指出 Conv2d 问题更像 outlier range、channel imbalance，还是二者混合。
- 明确 `role=unknown/stage5/fusion/merge_proj` 与普通 `cnn branch` 的统计差异。

#### E005b：Conv2d-only percentile clipping

目的：验证简单稳健 range 是否能缓解 Conv2d tensor-wise A8 崩坏。

优先测试：

- all Conv2d clipping：
  - p99.9
  - p99.99
  - p99.995
  - p99.999
- 局部 Conv2d clipping：
  - `role=unknown`
  - `stage5`
  - `fusion`
  - `merge_proj`
  - `model.stage5.1`

实验方式：

- 不重跑 W20000。
- 从同一个 W4 weight-recon checkpoint 出发。
- 只重新做 A8 init / range calibration。
- 默认不跑 activation reconstruction，先看 A8 init 是否恢复。

需要记录：

- percentile 参数。
- clipping 作用范围。
- clipped range 与原 min/max range 的比例。
- 128-sample SNR mean / median / min。
- SSIM mean。
- E001 diagnostics 中 effective int levels、relative MSE、outlier ranking 是否改善。

预期结果：

- 如果 min/max outlier 是主因，合理 percentile clipping 应显著提升 all_on SNR。
- 过强 clipping 会牺牲真实大幅值信号，可能让 SNR 变差。
- 最可能有效的不是极低 percentile，而是较温和的 p99.99/p99.995/p99.999。

验收标准：

- 至少找到一个 Conv2d clipping 设置，使 128-sample A8 init SNR 明显高于 `-7.1021 dB`。
- 如果提升超过 `+2 dB`，说明 clipping 是主修复方向。
- 如果能接近或超过 `0 dB`，说明 E005 有强信号继续扩展。

#### E005c：Conv2d-only MSE range calibration

目的：验证比 percentile 更自适应的 range search 是否更稳。

应做内容：

- 对 Conv2d activation quantizer 初始化时，用 calibration activation 搜索使 fake-quant MSE 最小的 clipping range。
- 先只做 tensor-wise MSE clipping，不引入 per-channel。
- 范围候选可以基于 absmax shrink ratio，例如：
  - 1.0
  - 0.999
  - 0.995
  - 0.99
  - 0.98
  - 0.95
  - 0.90
  - 0.80

优先作用范围：

- all Conv2d。
- `role=unknown`。
- `stage5 + Conv2d`。
- `merge_proj`。
- `fusion Conv2d`。

预期结果：

- 如果 percentile clipping 对不同层不稳定，MSE calibration 可能更稳。
- 如果 MSE range 选择过于偏向局部 reconstruction MSE，也可能不等价于最终 SNR，需要 multi-sample eval 验证。

验收标准：

- 与 E005b 最佳 percentile clipping 对比。
- 如果 MSE calibration 更稳或更高，后续 E005 主线转向 MSE range。
- 如果 MSE 与 percentile 都无效，说明问题可能不是简单 range shrink。

#### E005d：结构化 clipping 对照

目的：判断修复是否必须覆盖全 Conv2d，还是可以只处理 E004g 高敏感子组。

建议对照：

- all Conv2d clipping。
- only `role=unknown` clipping。
- only `stage5` clipping。
- only `merge_proj` clipping。
- only `fusion` clipping。
- `role=unknown + stage5 + merge_proj` 组合。
- 普通 `cnn branch` clipping 作为低优先级对照。

预期结果：

- 如果只处理高敏感子组就有明显恢复，说明可形成结构感知低成本策略。
- 如果只处理局部子组恢复有限，而 all Conv2d 有效，说明需要全 Conv2d range 策略。
- 如果 `cnn branch` clipping 无效，可进一步排除普通 CNN conv branch 是主因。

验收标准：

- 给出“全 Conv2d 策略 vs 高敏感子组策略”的明确选择。
- 为 E006 mixed precision / selective FP32 提供候选结构组合。

#### E005e：粒度实验的进入条件

原始 E005 中包含 activation per-channel、group-wise、branch-wise / structure-wise。

这些不应一开始就做，进入条件如下：

- percentile / MSE clipping 有明显但不足的恢复：进入 per-channel / group-wise，判断剩余误差是否来自 channel imbalance。
- clipping 几乎无效，但 E005a 发现 per-channel absmax ratio 很高：直接进入 per-channel / group-wise。
- clipping 已经能大幅恢复：per-channel / group-wise 降为后续优化，不作为当前必要项。

粒度实验优先级：

1. Conv2d activation per-channel。
2. Conv2d group-wise activation。
3. FFB structure-wise：fusion / cnn / unknown / stage5 分结构策略。
4. branch-wise：CNN branch vs fusion vs transformer。

注意：activation per-channel 可能显著增加实现和部署复杂度。除非 tensor-wise clipping 无法恢复，否则不应优先把它作为 E005 第一阶段。

#### SmoothQuant 类缩放的优先级

原始计划中提到对 attention proj 试 SmoothQuant 类缩放。

结合 E004 结果，该方向优先级降低：

- attention qkv/proj 在 E001/E002 中与 activation reconstruction 劣化有关。
- 但在 A8 init 崩坏中，E004 证明 Linear / transformer 不是主因。
- 因此 SmoothQuant 类缩放不应进入 E005 第一阶段。

保留条件：

- 后续如果 Conv2d range 修复后，Linear / transformer 成为新的主要瓶颈，再开 E006/E007 单独处理。
- 或者如果 activation reconstruction 再次导致 attention qkv/proj 崩坏，再回到 SmoothQuant / reconstruction 稳定性问题。

#### E005 成功/失败判据

核心成功指标：

- 128-sample A8 init SNR mean 相对 `-7.1021 dB` 明显恢复。
- SNR median 同步提升，而不是只改善少数样本。
- SSIM mean 不明显恶化。
- E001 diagnostics 中 Conv2d effective int levels / relative MSE / outlier ratio 有改善。
- Linear / transformer sanity check 不应变差。

阶段性判断：

- 恢复 `< +1 dB`：弱信号，clipping 可能不是主因。
- 恢复 `+1 ~ +2 dB`：有信号，但可能不足，需要结构化组合或粒度实验。
- 恢复 `> +2 dB`：强信号，E005 主线成立。
- 恢复到 `0 dB` 以上：非常强，说明 Conv2d range calibration 是主要创新方向。
- 接近 W4A32 的 `4.7973 dB`：说明 W4A8 A8 init 主要可由 range calibration 修复。

#### E005 推荐执行顺序

1. E005a：Conv2d range diagnostics 增强。
2. E005b：all Conv2d percentile clipping sweep。
3. E005c：all Conv2d MSE range calibration sweep。
4. E005d：对 E005b/E005c 最佳策略做结构化局部对照。
5. E005e：仅在必要时进入 per-channel / group-wise / structure-wise。

当前不建议一开始做：

- 完整 per-channel activation 实现。
- 完整 group-wise activation 实现。
- SmoothQuant 类 attention scaling。
- activation reconstruction 学习率大 sweep。

#### E005 预期产出

E005 应输出：

- 一张 Conv2d range calibration 策略表。
- 一张 percentile / MSE clipping 对比表。
- 一张结构化作用范围对比表。
- 每个候选策略的 128-sample SNR / SSIM。
- 每个候选策略的 E001 diagnostics 对比。
- 明确结论：
  - outlier clipping 是否有效；
  - MSE range 是否优于 percentile；
  - 是否需要全 Conv2d；
  - 是否需要 per-channel / group-wise；
  - 是否可以进入 E006 mixed precision / deployment 策略。

#### 当前 E005 总体判断

E005 最合理的主线是：

> 先做 Conv2d tensor-wise activation range 修复，而不是立即做复杂粒度或 transformer smoothing。

原因：

- E004 已经把问题定位到 Conv2d activation 组。
- clipping / MSE range 是验证 outlier 假设的最小修复。
- 如果最小修复有效，它更容易形成清晰、可解释、可部署的创新点。
- 如果最小修复无效，再进入 per-channel / group-wise，证据链也更清楚。

### 2026-05-06 E005a：Conv2d range diagnostics 基线

#### 目标

E005a 只做诊断增强和基线报告，不修改量化算法、不做 clipping、不跑 activation reconstruction。

核心问题是确认 E004 定位出的 Conv2d A8 崩坏更像：

- outlier range 主导；
- channel imbalance 主导；
- 或者二者混合。

同时，把旧诊断中的 `role=unknown` 改成更准确的结构标签，避免后续把 stage 输出卷积误解成未知结构。

#### 工具改动

- activation diagnostics 新增高分位字段：
  - `p99_99`
  - `p99_999`
  - `abs_p99_99`
  - `abs_p99_999`
  - `absmax_over_p99_99`
  - `absmax_over_p99_999`
- 新增 `conv2d_range_summary`：
  - overall Conv2d 汇总；
  - by stage；
  - by branch；
  - by role；
  - by module type。
- `model.stage1.1` 到 `model.stage5.1` 统一标记为：
  - `branch=stage_output`
  - `role=stage_output_conv`
- `diagnose_activation_quantization.py` 新增 `--cuda-device-index`，用于显式选择 GPU，例如 `cuda:1`。
- 大张量诊断说明：
  - 对超过阈值的大张量，高分位和 fake-quant 局部误差使用确定性 stride sampling。
  - 这样做是为了让 128-sample diagnostics 能稳定跑完。
  - 该采样只影响诊断统计成本和近似程度，不改变模型行为、量化参数或 checkpoint。

#### E005a 配置

配置文件：

`SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/e005a_conv2d_range_diagnostics.json`

默认口径：

- checkpoint：E002c A8 init n=64 / pre-act-recon checkpoint
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002c_init_sensitivity/quant/20260505_150842_e002c_init_n0064/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- samples：128
- batch size：16
- seed：`20260427`
- device：`cuda`
- cuda device index：`1`
- run root：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005a_diagnostics/`

#### 运行记录

smoke run：

- run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005a_diagnostics/20260506_141020_e005a_smoke_conv2d_range_diagnostics_fast/`
- `activation_quantizers=52`
- `non_positive_delta_count=0`
- `activation_stat_count=52`

formal 128-sample run：

- run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005a_diagnostics/20260506_141901_e005a_conv2d_range_diagnostics/`
- device：`cuda:1`
- `activation_quantizers=52`
- `non_positive_delta_count=0`
- `activation_stat_count=52`
- `fake_quant_mse_max=9.82716228463687e-05`
- `effective_int_levels_min=124`
- `summary.json` 包含 `conv2d_range_summary`
- run 产物位于 `.gitignore` 保护目录，不纳入 Git。

#### Conv2d vs Linear / transformer

module type 汇总：

| module type | count | effective level min | effective level mean | relative MSE max | absmax/p99 max | absmax/p99.99 max | per-channel ratio max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conv2d | 32 | 124 | 210.34 | 0.2044677242 | 137.7856 | 29.3499 | 8.5134 |
| Linear | 20 | 139 | 216.85 | 0.0002060855 | 6.6914 | 6.0692 | 5.2560 |

可读结论：

- Conv2d 的 worst relative MSE 比 Linear 高三个数量级左右。
- Conv2d 的极端 outlier ratio 明显更强，尤其 `head/tail`。
- Linear / transformer 仍有若干低 effective level 层，但从 E004 sensitivity 看，A8 init 崩坏主因不是 Linear / transformer。

#### Conv2d branch / role 统计

branch 汇总：

| branch | count | effective level min | relative MSE max | absmax/p99.99 max | per-channel ratio max |
|---|---:|---:|---:|---:|---:|
| cnn | 15 | 200 | 0.0002860603 | 2.0014 | 8.5134 |
| fusion | 10 | 184 | 0.0011164239 | 8.0258 | 3.0885 |
| stage_output | 5 | 189 | 0.0010821137 | 1.0404 | 2.5581 |
| head | 1 | 124 | 0.0874347530 | 29.3499 | 2.5129 |
| tail | 1 | 140 | 0.2044677242 | 13.6724 | 1.0000 |

role 汇总：

| role | count | effective level min | relative MSE max | absmax/p99.99 max | per-channel ratio max |
|---|---:|---:|---:|---:|---:|
| conv | 15 | 200 | 0.0002860603 | 2.0014 | 8.5134 |
| merge_proj | 5 | 190 | 0.0011164239 | 1.0524 | 2.5511 |
| split_proj | 5 | 184 | 0.0009059406 | 8.0258 | 3.0885 |
| stage_output_conv | 5 | 189 | 0.0010821137 | 1.0404 | 2.5581 |
| head | 1 | 124 | 0.0874347530 | 29.3499 | 2.5129 |
| tail | 1 | 140 | 0.2044677242 | 13.6724 | 1.0000 |

解释：

- `fusion/split_proj` 更像 outlier range 问题：`absmax/p99.99` 最高可到 `8.0258`。
- `merge_proj/stage_output_conv` 的 extreme percentile ratio 不算最夸张，但 relative MSE 是内部 Conv2d 中最高的一批，说明它们可能需要 MSE range 或结构化 clipping，而不只是盯着最大 outlier。
- `stage5/cnn conv_branch.6` 有最强 per-channel imbalance：`per_channel_absmax_ratio=8.5134`，但 `cnn branch` 整体 sensitivity 不高，因此它更像后续粒度实验的候选，而不是 E005 第一优先级。
- `head/tail` 的 outlier 和 relative MSE 极强，但 E004 sensitivity 中输出 activation quantizer 默认不参与候选，且 head 单点关闭恢复有限。因此它们是诊断警报，不是当前首要修复对象。

#### Top 问题层

top outlier layers：

1. `model.head`：Conv2d，`absmax/p99=137.7856`
2. `model.tail`：Conv2d，`absmax/p99=86.2140`
3. `model.stage1.0.block.split_proj`：Conv2d，`absmax/p99=8.2788`
4. `model.stage1.0.block.trans_branch.mlp.2`：Linear，`absmax/p99=6.6914`
5. `model.stage2.0.block.trans_branch.mlp.0`：Linear，`absmax/p99=6.0676`

top per-channel imbalance：

1. `model.stage5.0.block.conv_branch.6`：Conv2d，ratio `8.5134`
2. `model.stage1.0.block.trans_branch.mlp.2`：Linear，ratio `5.2560`
3. `model.stage2.0.block.conv_branch.6`：Conv2d，ratio `3.4124`
4. `model.stage1.0.block.split_proj`：Conv2d，ratio `3.0885`
5. `model.stage2.1`：stage output Conv2d，ratio `2.5581`

worst relative MSE：

1. `model.tail`：Conv2d，`0.2044677242`
2. `model.head`：Conv2d，`0.0874347530`
3. `model.stage2.0.block.merge_proj`：Conv2d，`0.0011164239`
4. `model.stage2.1`：stage output Conv2d，`0.0010821137`
5. `model.stage3.0.block.merge_proj`：Conv2d，`0.0010705261`

lowest effective levels：

1. `model.head`：Conv2d，`124`
2. `model.stage2.0.block.trans_branch.mlp.0`：Linear，`139`
3. `model.tail`：Conv2d，`140`
4. `model.stage5.0.block.trans_branch.mlp.0`：Linear，`154`
5. `model.stage1.0.block.trans_branch.mlp.0`：Linear，`172`

#### E005a 结论

E005a 支持下面判断：

- 当前 A8 init 崩坏不是单纯 attention/Linear 问题。
- Conv2d activation 的 range 问题是真实强信号。
- 但 Conv2d 内部不是一种问题：
  - `fusion/split_proj` 偏 outlier range；
  - `merge_proj/stage_output_conv` 偏高 relative MSE；
  - `stage5/cnn conv_branch.6` 偏 channel imbalance；
  - `head/tail` 是高 outlier 警报，但暂不作为主修复目标。
- 因此 E005b 不应直接上 per-channel/group-wise，而应先验证 Conv2d tensor-wise percentile clipping 是否能恢复 128-sample SNR。

#### 对后续 E005 的影响

下一步建议：

1. E005b：先做 all Conv2d percentile clipping sweep。
2. 同步保留结构化作用范围：
   - all Conv2d；
   - `fusion/split_proj`；
   - `merge_proj`；
   - `stage_output_conv`；
   - `stage5`。
3. 如果 percentile clipping 对 `fusion/split_proj` 有效，但对 `merge_proj/stage_output_conv` 不足，再进入 E005c MSE range calibration。
4. 如果 clipping 有明显恢复但 stage5 仍拖后腿，再考虑 per-channel/group-wise。
5. Linear / transformer 继续作为 sanity check，不作为 E005 第一阶段主线。

### 2026-05-06 E004-E005a：可读结论整理

#### 总体判断

E004 和 E005a 合起来给出的结论不是简单的“Conv2d 有问题”，而是：

> 在 SCRN 的 W4A8 A8 init 阶段，主要失真来自 Conv2d activation quantization 的多点累积误差。这个误差不是单个 quantizer 主导，也不是 attention/Transformer 在 A8 init 阶段主导；Conv2d 内部又同时存在 outlier range、局部相对量化误差偏高和部分通道尺度不均衡三类信号。

这条结论目前比“优先修 attention”更符合已有实验，因为 E004 的敏感性实验直接测了关闭不同 activation quantizer 后的 SNR 恢复。

#### E004 说明了什么

E004 的核心作用是回答“哪些 activation quantizer 对最终多样本 SNR 最敏感”。

统一评估口径：

- checkpoint：A8 init n=64 / pre-act-recon checkpoint。
- eval：128 samples，seed `20260427`。
- `all_on`：所有候选 activation quantizers 开启，SNR mean `-7.1021 dB`，SSIM mean `0.1945`。
- `all_off`：候选 activation quantizers 关闭，近似 W4A32 / weight-recon 口径，SNR mean `4.7973 dB`，SSIM mean `0.7049`。
- `all_on -> all_off` gap：约 `11.8993 dB`。

E004b 的 sentinel 单点关闭说明：

- 单点关闭无法解释整体崩坏。
- 单个 quantizer 的最大恢复远小于 `all_on -> all_off` gap。
- stage4/stage5 attention qkv/proj 单点关闭几乎不恢复 SNR。

E004d 的分组关闭给出强结论：

| 关闭对象 | quantizer 数 | SNR mean | 恢复 |
|---|---:|---:|---:|
| all Conv2d | 32 | 约 `4.5293` | `+11.6314 dB` |
| Linear / transformer | 20 | `-7.0812` | `+0.0209 dB` |

解释：

- 关闭全部 Conv2d activation quantizers 后，SNR 几乎恢复到 W4A32 / all_off 水平。
- 关闭全部 Linear / transformer activation quantizers 几乎没有恢复。
- 因此，A8 init 崩坏主因不是 Linear / transformer，而是 Conv2d activation quantization。
- E001 中 attention projection 的负 `delta` 仍然重要，但它更像 activation reconstruction 阶段的合法性/优化问题，不是当前 A8 init 崩坏的主来源。

E004e 的 Conv2d 子组细分说明：

| 关闭对象 | quantizer 数 | SNR mean | 恢复 | recovery ratio |
|---|---:|---:|---:|---:|
| stage output Conv2d，旧 `role=unknown` | 5 | `-4.2108` | `+2.8913 dB` | 24.3% |
| stage5 Conv2d | 6 | `-5.0410` | `+2.0611 dB` | 17.3% |
| fusion Conv2d | 10 | `-5.4061` | `+1.6960 dB` | 14.3% |
| merge_proj Conv2d | 5 | `-5.7532` | `+1.3489 dB` | 11.3% |
| `model.stage5.1` | 1 | `-6.0764` | `+1.0256 dB` | 8.6% |

解释：

- Conv2d 内部也不是平均贡献。
- 最值得优先处理的是 stage output Conv2d、stage5、fusion 和 merge projection。
- 单个 `model.stage5.1` 恢复超过 `+1 dB`，是很强的单点对照。
- 但任何单点或单个子组都不能完全解释 `+11.6314 dB`，所以问题仍然是多点累积。

#### `role=unknown` 的命名修正

E004 中的 `role=unknown` 后来确认并不是未知结构，而是每个 stage 末尾的 3x3 Conv2d：

- `model.stage1.1`
- `model.stage2.1`
- `model.stage3.1`
- `model.stage4.1`
- `model.stage5.1`

从 E005a 开始，这些层统一标记为：

- `branch=stage_output`
- `role=stage_output_conv`

因此，后续不再把这组结构称为 unknown，而应称为 stage output Conv2d。

#### E005a 说明了什么

E005a 的核心作用是解释“为什么 E004 指向 Conv2d”。

module type 级别对比：

| 类型 | 数量 | effective level min | relative MSE max | absmax/p99 max | absmax/p99.99 max | per-channel ratio max |
|---|---:|---:|---:|---:|---:|---:|
| Conv2d | 32 | 124 | `0.2044677` | `137.7856` | `29.3499` | `8.5134` |
| Linear | 20 | 139 | `0.0002061` | `6.6914` | `6.0692` | `5.2560` |

可读解释：

- Conv2d 的最坏 relative MSE 比 Linear 高约三个数量级。
- Conv2d 的 outlier ratio 明显更强，说明少数极值会把 tensor-wise activation range 拉得很宽。
- Linear / transformer 虽然也有低 effective level 和部分 outlier，但 E004 已经证明它们不是当前 A8 init SNR 崩坏的主导因素。

#### Conv2d 内部的三类问题

E005a 显示 Conv2d 内部不是一种统一故障，而是至少有三类信号。

第一类：outlier range 更明显的结构。

- `fusion/split_proj` 的高分位 outlier ratio 更突出。
- `split_proj` 的 `absmax/p99.99` 最高到 `8.0258`。
- 这类结构适合优先验证 percentile clipping，因为问题很像少数极值撑大 min/max range。

第二类：relative MSE 更突出的结构。

- `merge_proj` relative MSE max：`0.0011164239`。
- `stage_output_conv` relative MSE max：`0.0010821137`。
- 它们的 extreme percentile ratio 不一定最高，但 fake-quant 后局部相对误差偏高。
- 这类结构可能需要 MSE range calibration 或更结构化的 clipping，而不只是固定 percentile。

第三类：channel imbalance 更明显的结构。

- `model.stage5.0.block.conv_branch.6` 的 per-channel absmax ratio：`8.5134`。
- 这说明某些 Conv2d 输出通道的幅值明显大于其他通道。
- 但 E004 中普通 `cnn branch` 整体 sensitivity 不高，所以 per-channel/group-wise 不应作为第一步，而应作为 clipping 之后的后续候选。

#### head/tail 的解释边界

E005a 中 `model.head` 和 `model.tail` 的 outlier 与 relative MSE 很强：

- `model.head`：`absmax/p99=137.7856`
- `model.tail`：`absmax/p99=86.2140`
- `model.tail` relative MSE：`0.2044677242`

但当前不把 head/tail 作为第一修复对象，原因是：

- E004 sensitivity 默认排除最终输出 activation quantizer，以对齐现有 eval 口径。
- head 单点关闭恢复有限。
- head/tail 更像需要持续监控的诊断警报，而不是当前最确定的主修复入口。

#### 对 attention / Transformer 的重新定位

不能说 attention / Transformer 没问题。

更准确的说法是：

- 在 activation reconstruction 阶段，attention projection 曾经出现负 `delta`，这是确定的非法状态和优化不稳定问题。
- 但在 A8 init 的 128-sample sensitivity 口径下，关闭 Linear / transformer 几乎不恢复 SNR。
- 因此，当前 E005 第一阶段不应优先做 attention SmoothQuant 或 transformer-specific 策略。
- Linear / transformer 应继续保留为 sanity check，防止修 Conv2d 后出现新的瓶颈。

#### 当前最可信的问题链条

现阶段可以把 W4A8 激活量化失败拆成两条链：

1. activation reconstruction 链：
   - 曾经把 stage4/stage5 attention projection 的 `delta` 优化成负数。
   - E002a 已用正 scale clamp 修掉合法性问题。
   - 但修掉负 `delta` 不能恢复整体 W4A8 SNR，说明它不是唯一主因。

2. A8 init / activation range 链：
   - 128-sample all_on A8 init SNR mean 为 `-7.1021 dB`。
   - 关闭全部 Conv2d activation quantizers 恢复 `+11.6314 dB`。
   - Conv2d 的 outlier ratio、relative MSE 和 per-channel imbalance 均强于 Linear。
   - 因此当前主线应转向 Conv2d activation range calibration。

#### 后续实验决策

E005b 应优先做：

1. all Conv2d tensor-wise percentile clipping。
2. `fusion/split_proj` clipping。
3. `merge_proj` clipping。
4. `stage_output_conv` clipping。
5. `stage5` clipping。

判断标准：

- 如果 percentile clipping 明显提升 128-sample SNR，说明 outlier/range 是主修复方向。
- 如果只对 `fusion/split_proj` 有效，而 `merge_proj/stage_output_conv` 仍差，则进入 E005c MSE range calibration。
- 如果 clipping 有一定恢复但 stage5 仍拖后腿，再考虑 per-channel/group-wise。
- 如果 Conv2d range 修复后 Linear / transformer 变成新瓶颈，再重新考虑 attention/Transformer 专项策略。

当前不建议立刻做：

- 完整 per-channel activation 实现。
- 完整 group-wise activation 实现。
- attention SmoothQuant。
- activation reconstruction 学习率大 sweep。

原因是 E004-E005a 的证据链已经把第一优先级指向了 Conv2d tensor-wise range 修复。直接跳到更复杂策略会让变量过多，难以判断真正有效的机制。

### 2026-05-06 E005b：Conv2d percentile clipping 工具与 smoke

#### 目标

E005b 开始验证 Conv2d activation tensor-wise range 是否被 outlier 拉坏。

本阶段只做：

- 从 W4 weight-recon checkpoint 出发。
- A8 init。
- 对选中的 activation quantizer 执行 percentile range calibration。
- 保存 pre-act-recon checkpoint。

本阶段不做：

- activation reconstruction。
- per-channel / group-wise。
- attention SmoothQuant。
- W20000 weight reconstruction 重跑。

#### 工具实现

新增 `quant/activation_range.py`：

- 支持 two-sided percentile range：
  - `lower_q=(1-p/100)/2`
  - `upper_q=1-lower_q`
  - `clipped_min=min(q(lower_q), 0)`
  - `clipped_max=max(q(upper_q), 0)`
- 复用 E004 selector 语义：
  - `index`
  - `name_contains`
  - `stage`
  - `branch`
  - `role`
  - `module_type`
- 默认排除最后一个 output activation quantizer。
- 只写选中 activation quantizer 的 `delta/zero_point`。
- 不修改 `UniformAffineQuantizer.forward()`。
- 大张量每层最多采样 `500000` 个 activation 值，使用确定性 stride sampling。

扩展 `activation_only_quantize_scrn.py`：

- 新增 `--config`。
- 新增 `--cuda-device-index`。
- 新增：
  - `--activation-range-method {none,percentile}`
  - `--activation-percentile`
  - `--range-index`
  - `--range-name-contains`
  - `--range-stage`
  - `--range-branch`
  - `--range-role`
  - `--range-module-type`
  - `--range-max-values-per-layer`
  - `--include-output-quantizer`
- 在 metrics 中记录 `activation_range_summary`：
  - method
  - percentile
  - selected count
  - selected names
  - per-layer original range / clipped range / shrink ratio
  - sampled 与 sample count

新增配置：

- `SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/e005b_conv2d_percentile_clipping.json`

默认设置：

- checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260504_221242_e002b_w4a8_positive_scale_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq_weight_recon.pth`
- run root：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005b_percentile/quant`
- `num_samples=64`
- `init_batch_size=64`
- `batch_size=16`
- `seed=1005`
- `activation_range_method=percentile`
- `activation_percentile=99.99`
- `range_module_type=Conv2d`
- `include_output_quantizer=false`

#### TDD 与最小验证

已新增/更新测试：

- `tests/test_activation_range.py`
- `tests/test_activation_only_quantize_scrn.py`

TDD red：

- `test_activation_range` 初始失败于缺少 `activation_range` 模块。
- `test_activation_only_quantize_scrn` 初始失败于缺少新增 CLI/config 参数。

已通过检查：

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_range`
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn`
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/activation_range.py SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.activation_only_quantize_scrn --help`

#### E005b-0 smoke

命令口径：

- `num_samples=2`
- `init_batch_size=2`
- `batch_size=1`
- `activation_percentile=99.9`
- `range_module_type=Conv2d`
- `device=cuda`
- `cuda_device_index=1`
- `skip_act_recon=true`

run dir：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005b_percentile/quant/20260506_163005_e005b_smoke_percentile_p999/`

结果：

- device：`cuda:1`
- `activation_quantizers=52`
- `non_positive_delta_count=0`
- selected Conv2d quantizers：31
- first selected names：
  - `model.head`
  - `model.stage1.0.block.split_proj`
  - `model.stage1.0.block.merge_proj`
  - `model.stage1.0.block.conv_branch.0`
  - `model.stage1.0.block.conv_branch.3`
- single-sample `quant_pre_act_recon_snr_db=0.4362 dB`

解释：

- smoke 证明 E005b percentile clipping workflow 可以完成 checkpoint 生成。
- 默认排除 output quantizer 后，Conv2d selector 选中 31 个 quantizers。
- `non_positive_delta_count=0`，说明 percentile 写入没有引入非法 scale。
- single-sample SNR 只用于 smoke sanity check，不作为 E005b 正式结论。

#### 下一步

进入 E005b-1 all Conv2d percentile sweep：

- `99.9`
- `99.99`
- `99.995`
- `99.999`

每个 run 必须做：

- A8 init + percentile range。
- 128-sample multi eval。
- 128-sample E005a diagnostics。
- 记录相对 `all_on=-7.1021 dB` 的 SNR 恢复和 diagnostics 指标变化。

### 2026-05-06 E005b：Conv2d percentile clipping 正式结果

#### 实验口径

本轮只验证 Conv2d tensor-wise percentile clipping 是否能修复 A8 init 崩坏。

固定设置：

- 起点：同一个 W4 weight-recon checkpoint。
- activation init：`num_samples=64`、`init_batch_size=64`、`seed=1005`。
- 不跑 activation reconstruction。
- 输出 checkpoint：`quantized_scrn_brecq_pre_act_recon.pth`。
- 正式评价：128-sample multi eval，`seed=20260427`、`batch_size=16`、CUDA。
- diagnostics：128-sample E005a diagnostics。
- baseline：
  - all_on A8 init：SNR mean `-7.1021 dB`。
  - all_off / W4A32 近似：SNR mean `4.7973 dB`。
  - all_on 到 all_off gap 约 `11.8993 dB`。

#### E005b-1：all Conv2d percentile sweep

| run | selected | single-sample SNR | 128-sample SNR mean | median | min | SSIM mean | delta vs all_on | recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all Conv2d p99.9 | 31 | 0.6913 | -11.5661 | -12.3617 | -18.9702 | 0.3979 | -4.4640 | -0.375 |
| all Conv2d p99.99 | 31 | 1.7552 | -8.0514 | -8.7076 | -15.2199 | 0.2250 | -0.9493 | -0.080 |
| all Conv2d p99.995 | 31 | 2.3028 | -8.0202 | -8.6771 | -15.2424 | 0.2247 | -0.9181 | -0.077 |
| all Conv2d p99.999 | 31 | 3.0883 | -7.9668 | -8.6962 | -15.2510 | 0.2272 | -0.8647 | -0.073 |

结果解释：

- all Conv2d percentile clipping 没有带来恢复，反而全部低于原始 all_on A8 init。
- percentile 越宽松，结果越接近 all_on，但即使 p99.999 仍比 all_on 低约 `0.8647 dB`。
- p99.9 过强 clipping 明显伤害 SNR，说明简单截断 outlier 会引入严重饱和误差。
- 单样本 SNR 随 percentile 放宽从 `0.6913 dB` 上升到 `3.0883 dB`，但 128-sample mean 仍没有恢复；因此后续仍以 multi-sample 结果为准，不能用单图 SNR 判断有效。

diagnostics 结果：

- 所有 all Conv2d run 均保持 `activation_quantizers=52`、`non_positive_delta_count=0`。
- p99.999 all Conv2d run 中，selected Conv2d quantizers 为 31 个。
- p99.999 的 per-layer range shrink ratio：
  - min `0.2427`
  - mean `0.8891`
  - max `0.9983`
- p99.999 all Conv2d diagnostics：
  - Conv2d `fake_quant_relative_mse_max=0.204468`
  - Conv2d `fake_quant_relative_mse_mean=0.010738`
  - Conv2d `effective_int_levels_min=140`
  - Conv2d `effective_int_levels_mean=219.0`
  - Conv2d `absmax_over_p99.99_max=29.3499`
  - Conv2d `absmax_over_p99.99_mean=2.5891`
- 对比 E005a 原始 A8 init：原始 Conv2d `fake_quant_mse_max=9.827e-05`，all Conv2d p99.999 后 `fake_quant_mse_max=0.007526`。这说明简单 tensor-wise percentile clipping 通过缩小 scale 换来更多饱和误差，局部 fake-quant MSE 反而放大。

#### E005b-2：结构组对照

使用 E005b-1 中相对最不差的 `p99.999`，只对局部 Conv2d 结构组应用 percentile range。

| run | selected | single-sample SNR | 128-sample SNR mean | median | min | SSIM mean | delta vs all_on | recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fusion p99.999 | 10 | 4.8361 | -7.1144 | -7.8704 | -14.3532 | 0.2050 | -0.0123 | -0.001 |
| split_proj p99.999 | 5 | 4.8327 | -7.0859 | -7.8377 | -14.3553 | 0.1943 | +0.0162 | 0.001 |
| merge_proj p99.999 | 5 | 4.9909 | -7.1121 | -7.8484 | -14.4247 | 0.2044 | -0.0100 | -0.001 |
| stage_output_conv p99.999 | 5 | 4.9830 | -7.0469 | -7.7355 | -14.3601 | 0.2019 | +0.0552 | 0.005 |
| stage5 Conv2d p99.999 | 6 | 4.9873 | -8.1822 | -9.0382 | -15.5562 | 0.2408 | -1.0801 | -0.091 |

结果解释：

- 局部结构组没有出现足够强的恢复信号。
- `stage_output_conv` 是本轮相对最好的局部组，但只恢复 `+0.0552 dB`，recovery ratio 约 `0.5%`，不足以证明 percentile clipping 是有效修复。
- `split_proj` 只有 `+0.0162 dB`，可以视为噪声级别弱信号。
- `fusion`、`merge_proj` 基本持平或略差。
- `stage5 Conv2d` 明显变差 `-1.0801 dB`，说明后期结构对 clipping 饱和更敏感，不能简单扩大到 stage5。
- 结构组单样本 SNR 都在 `4.83-4.99 dB` 附近，但 128-sample mean 仍接近 all_on；再次说明单样本结果存在很强偶然性，不应作为策略选择依据。

#### E005b 结论

E005b 没有验证“Conv2d A8 崩坏主要可由 tensor-wise percentile clipping 修复”。

更准确的判断是：

- Conv2d activation 仍是 W4A8 A8 init 崩坏的主要敏感区域，这一点由 E004/E005a 支持。
- 但问题不是简单的“outlier 把 range 拉大，所以截掉 outlier 就恢复”。
- 简单 percentile clipping 会缩小 selected Conv2d quantizer 的 range，但也会带来饱和误差，尤其 all Conv2d 和 stage5 Conv2d 会明显伤害 multi-sample SNR。
- 当前 evidence 不支持继续扩大 percentile sweep 或把 percentile clipping 作为主修复策略。

#### 后续判断

E005b 后，优先级应调整为：

1. 进入 E005c：MSE range calibration。原因是 percentile 只按分位数截断，不直接优化 fake-quant error；当前结果显示“截断 outlier”容易转化为饱和误差。
2. 如果 E005c 仍无恢复，再考虑 per-channel / group-wise activation quantization。原因是 E005a 显示 Conv2d 存在 channel imbalance，但 E005b 的 tensor-wise range 修复不足。
3. attention / Linear 仍保留为 sanity check，不作为 E005b 后的第一修复对象。E005b 未推翻 E004 的结构敏感性结论。
4. 后续所有策略仍必须用 128-sample eval 判断，不接受单样本 SNR 作为成功证据。

### 2026-05-06 E005c：Conv2d MSE range calibration 正式结果

#### 实验动机

E005b 说明 percentile clipping 没有恢复 W4A8 A8 init，反而容易把 outlier 问题转成饱和误差。

E005c 因此测试更自适应的 tensor-wise range search：

- `max`：对选中 Conv2d activation quantizer 写 full min/max range，作为“不裁剪”对照。
- `mse_grid`：在一组 shrink ratio 上搜索 fake-quant Lp/MSE loss 最小的 range。
- conservative grid：`1.0,0.999,0.995,0.99,0.98,0.97,0.96,0.95`。
- standard grid：`1.0,0.999,0.995,0.99,0.98,0.97,0.96,0.95,0.925,0.9,0.875,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5`。

重要前提：

- 当前 all_on A8 init 本身已经使用 `scale_method=mse`。
- 因此 E005c 不是“首次启用 MSE”，而是审计和细化 Conv2d-only MSE range。

#### 工具改动

扩展 `quant/activation_range.py`：

- 新增通用入口 `apply_activation_ranges()`。
- 保留 `apply_percentile_activation_ranges()` 作为兼容接口。
- 新增 `max` method。
- 新增 `mse_grid` method。
- 新增 `parse_mse_shrink_ratios()`。
- 每层记录：
  - `chosen_min/chosen_max`
  - `chosen_range`
  - `range_shrink_ratio`
  - `best_shrink_ratio`
  - `best_score`
  - `candidate_scores`
  - `sampled/sample_count`

扩展 `activation_only_quantize_scrn.py`：

- `--activation-range-method` 支持 `{none,percentile,max,mse_grid}`。
- 新增 `--range-mse-shrink-ratios`。
- 新增 `--range-loss-p`，默认 `2.4`。

新增配置：

- `SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/e005c_conv2d_mse_range.json`

#### 验证

已通过：

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_range`
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn`
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/activation_range.py SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.activation_only_quantize_scrn --help`

E005c-0 smoke：

- run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005c_mse_range/quant/20260506_190919_e005c_smoke_mse_grid/`
- device：`cuda:1`
- `activation_quantizers=52`
- `non_positive_delta_count=0`
- selected Conv2d quantizers：31
- single-sample `quant_pre_act_recon_snr_db=8.7106 dB`
- 该 smoke 只证明工具链有效，不作为正式结论。

#### E005c-1：all Conv2d range controls

baseline：

- all_on A8 init：SNR mean `-7.1021 dB`。
- all_off / W4A32 近似：SNR mean `4.7973 dB`。
- all_on 到 all_off gap 约 `11.8993 dB`。

| run | method | selected | single-sample SNR | 128-sample SNR mean | median | min | SSIM mean | delta vs all_on | recovery | best shrink min/mean/max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all Conv2d max | max | 31 | 9.1001 | -7.2963 | -8.1817 | -14.6397 | 0.2000 | -0.1942 | -0.016 | 1.000/1.000/1.000 |
| all Conv2d MSE conservative | mse_grid | 31 | 8.8912 | -7.1224 | -7.9550 | -14.4597 | 0.1909 | -0.0203 | -0.002 | 0.950/0.971/1.000 |
| all Conv2d MSE standard | mse_grid | 31 | 5.7910 | -7.7571 | -8.6296 | -15.0790 | 0.2130 | -0.6550 | -0.055 | 0.500/0.902/0.995 |

诊断对比：

| run | Conv2d fake-quant MSE max | Conv2d fake-quant MSE mean | Conv2d effective level min | Conv2d effective level mean |
|---|---:|---:|---:|---:|
| E005a original A8 init | 0.00009827 | 0.00005048 | 124 | 210.34 |
| all Conv2d max | 0.00012242 | 0.00005360 | 63 | 202.72 |
| all Conv2d MSE conservative | 0.00014805 | 0.00005735 | 68 | 205.94 |
| all Conv2d MSE standard | 0.00450799 | 0.00041749 | 109 | 213.72 |

解释：

- 三个 all Conv2d range control 都没有超过 all_on baseline。
- `max` 单样本 SNR 很高，但 128-sample mean 仍比 all_on 低 `0.1942 dB`，再次说明单样本偶然性很强。
- conservative MSE 最接近 all_on，但仍低 `0.0203 dB`，可以视为基本持平或轻微变差，不是有效恢复。
- standard MSE grid 允许缩到 0.5，平均 best shrink ratio 为 `0.902`，正式 SNR 反而下降 `0.6550 dB`，说明局部 fake-quant loss 搜索更激进时会伤害最终输出。
- `max` 和 conservative MSE 都把 Conv2d effective level min 从原始 `124` 降到 `63/68`，说明 range 变宽或局部改写并没有改善有效 int level。

#### E005c-2：结构组对照

使用 E005c-1 中最接近 all_on 的 conservative MSE grid。

| run | selected | single-sample SNR | 128-sample SNR mean | median | min | SSIM mean | delta vs all_on | recovery | best shrink min/mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fusion MSE conservative | 10 | 4.9992 | -7.1603 | -7.8697 | -14.4085 | 0.1955 | -0.0582 | -0.005 | 0.950/0.979/0.995 |
| split_proj MSE conservative | 5 | 4.9969 | -7.1614 | -7.8870 | -14.4011 | 0.1895 | -0.0593 | -0.005 | 0.970/0.978/0.990 |
| merge_proj MSE conservative | 5 | 4.9902 | -7.0997 | -7.8480 | -14.3643 | 0.2010 | +0.0024 | 0.000 | 0.950/0.979/0.995 |
| stage_output_conv MSE conservative | 5 | 4.9944 | -7.1444 | -7.8779 | -14.4226 | 0.1957 | -0.0423 | -0.004 | 0.960/0.980/0.990 |
| stage5 Conv2d MSE conservative | 6 | 4.9881 | -7.1600 | -7.9289 | -14.4799 | 0.1908 | -0.0580 | -0.005 | 0.950/0.975/1.000 |

解释：

- 结构组 MSE range 也没有有效恢复。
- `merge_proj` 是唯一略高于 all_on 的组，但只提升 `+0.0024 dB`，属于噪声级别。
- 其他结构组都轻微变差。
- 相比 E005b percentile 的 `stage_output_conv +0.0552 dB`，E005c conservative MSE 没有提供更强结构信号。

#### E005c 结论

E005c 没有验证“Conv2d A8 init 崩坏可以由 tensor-wise MSE range search 修复”。

更准确的判断：

- 当前 all_on A8 init 已经使用 MSE scale，额外的 Conv2d-only `max` 或 `mse_grid` 改写没有带来 128-sample 恢复。
- `max` 明显优于 aggressive standard MSE，但仍不如 all_on，说明不是简单的“当前 MSE shrink 过强，换 full range 就好”。
- conservative MSE 基本贴近 all_on，说明当前默认 MSE 初始化已经接近这个 tensor-wise range search 的上限。
- standard MSE grid 的强 shrink 会进一步伤害 SNR，说明局部 fake-quant loss 最优不等于最终恢复质量最优。
- E005b/E005c 合起来基本否定了“Conv2d tensor-wise range/clipping 是主要可修复瓶颈”。

#### 后续判断

E005 继续扩大 tensor-wise percentile/MSE sweep 的价值很低。

下一步更合理方向：

1. 进入 activation per-channel / group-wise feasibility。E005a 已显示 Conv2d 存在 channel imbalance，且 tensor-wise range search 无效。
2. 保留 Linear / transformer sanity check，确保 Conv2d 粒度策略不会把瓶颈转移到 attention。
3. 如果 per-channel/group-wise 仍无效，再转入 E006：mixed precision / selective FP32 activation quantizer / reconstruction objective。
4. 后续仍必须以 128-sample eval 为主，single-sample SNR 只能做 smoke。

### 2026-05-06 E005D：结构化 Conv2d clipping 组合实验

#### 目的

E005D 继续结构化 clipping 对照，但不重复 E005b/E005c 已完成的单组实验。核心问题是：

- 单组 percentile / MSE clipping 基本无效后，多个高敏感 Conv2d 子组联合处理是否会产生恢复。
- 排除 stage5 或 head 等明显风险结构后，再扩大 Conv2d clipping 覆盖面是否更稳。
- 如果组合仍无效，是否可以停止 tensor-wise clipping 主线。

成功阈值：

- `+0.2 dB`：弱有效，值得继续分析。
- `+1.0 dB`：强信号，tensor-wise clipping 可继续作为主线。
- 低于 `+0.2 dB`：视为噪声级或无效。

baseline：

- all_on A8 init：128-sample SNR mean `-7.1021 dB`。
- all_off / W4A32 近似：128-sample SNR mean `4.7973 dB`。
- all_on 到 all_off gap：`11.8993 dB`。

#### 工具扩展

新增 selector 组合能力：

- `range_selector_groups`：多个 selector 取并集。
- `range_exclude_selector_groups`：从候选集合中排除指定结构。
- 每个 selector 支持：
  - `index`
  - `name_contains`
  - `stage`
  - `branch`
  - `role`
  - `module_type`
- `range_selector_groups` 与旧单 selector 字段互斥，避免同时出现“并集”和“单条件 AND”的语义混乱。

新增 CLI 参数：

- `--range-selector-groups-json`
- `--range-exclude-selector-groups-json`

新增配置：

- `SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/e005d_structured_clipping.json`

#### Smoke

- run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005d_structured_clipping/quant/20260506_195451_e005d_smoke_p99999_merge_stageout/`
- method：`percentile`
- percentile：`99.999`
- selector union：
  - `role=merge_proj + module_type=Conv2d`
  - `role=stage_output_conv + module_type=Conv2d`
- device：`cuda:1`
- selected Conv2d quantizers：10
- `activation_quantizers=52`
- `non_positive_delta_count=0`
- single-sample `quant_pre_act_recon_snr_db=8.3784 dB`

该 smoke 只证明 selector union、生效路径、checkpoint 保存和正 scale 状态正常，不作为正式效果结论。

#### 正式实验结果

| run | method | selected | single-sample SNR | 128-sample SNR mean | median | min | SSIM mean | delta vs all_on | recovery |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p99.999 merge + stage_output | percentile | 10 | 4.9862 | -7.0700 | -7.8076 | -14.3644 | 0.2147 | +0.0321 | 0.003 |
| p99.999 split + merge + stage_output | percentile | 15 | 4.8327 | -7.0633 | -7.8080 | -14.3794 | 0.2144 | +0.0388 | 0.003 |
| p99.999 all Conv2d except stage5 | percentile | 25 | 3.0929 | -7.5204 | -8.1969 | -14.7713 | 0.2039 | -0.4183 | -0.035 |
| p99.999 all Conv2d except head + stage5 | percentile | 24 | 4.8333 | -7.4515 | -8.2368 | -14.6998 | 0.2082 | -0.3494 | -0.029 |
| MSE conservative merge + stage_output | mse_grid | 10 | 4.9797 | -7.1323 | -7.9026 | -14.4646 | 0.2017 | -0.0302 | -0.003 |
| MSE conservative all Conv2d except stage5 | mse_grid | 25 | 8.8828 | -7.0689 | -7.9072 | -14.3853 | 0.1948 | +0.0332 | 0.003 |

诊断摘要：

- 所有正式 run 均保持 `activation_quantizers=52`。
- 所有正式 run 均保持 `non_positive_delta_count=0`。
- 组合 clipping 没有降低全局最坏 Conv2d relative MSE；最坏层仍由 head/tail 等极端层主导。
- p99.999 all Conv2d except stage5 / except head+stage5 的 diagnostics 中，`fake_quant_mse_max` 升至 `0.00388185`，说明大范围 percentile clipping 仍会引入明显饱和误差。
- MSE conservative all Conv2d except stage5 的 Conv2d effective level min 为 `68`，低于 E005a 原始 A8 init 的 `124`，说明该策略没有改善有效 int level。

#### 结论

E005D 没有验证“结构组合 tensor-wise clipping 能恢复 W4A8 A8 init”。

更准确的判断：

- 最好的组合是 `p99.999 split+merge+stage_output`，只恢复 `+0.0388 dB`，远低于 `+0.2 dB` 弱有效阈值。
- `merge+stage_output` 与 `MSE all Conv2d except stage5` 都只有约 `+0.03 dB`，属于噪声级弱信号。
- 扩大覆盖面到 all Conv2d except stage5 / except head+stage5 反而明显变差，说明不是“去掉有害结构后大范围 clipping 就有效”。
- MSE all Conv2d except stage5 单样本 SNR 很高，但 128-sample mean 仍接近 all_on，进一步确认单样本不能作为策略判断依据。

#### E005D 后的路线判断

E005b、E005c、E005D 三组实验已经覆盖：

- all Conv2d percentile clipping。
- Conv2d 子组 percentile clipping。
- all Conv2d MSE/max range calibration。
- Conv2d 子组 MSE range calibration。
- Conv2d 组合与排除结构 clipping。

这些 tensor-wise 方案都没有产生有效恢复。因此 E005 后续不应继续扩展 tensor-wise clipping sweep。

下一步优先级：

1. E005E：activation per-channel / group-wise feasibility。
   - 原因：E005a 已显示 Conv2d 存在 channel imbalance，tensor-wise range 已基本排除。
   - 注意不能直接复用 weight `channel_wise=True`，必须定义 activation channel 维度和广播 shape。
2. E006：mixed precision / selective FP32 activation quantizer。
   - 如果 E005E 实现成本过高或效果仍弱，则根据 E004g sensitivity 表保留高敏感 Conv2d activation quantizer 为 FP32。
3. Linear / transformer 继续作为 sanity check。
   - 当前不作为第一修复对象，但后续任何 Conv2d 粒度或混精度策略都要确认没有把瓶颈转移到 attention / Linear。

### 2026-05-06 E005 / E006 编号修订：E005 收束为离群值实验，E006 定义为粒度实验

为避免后续实验线索混乱，从本节点开始重新明确编号语义：

- E005：离群值、range、clipping 相关实验。
- E006：activation 量化粒度相关实验。

因此，前文中提到的 “E005E：activation per-channel / group-wise feasibility” 不再作为 E005 的延续实验执行，统一重命名为：

- E006a：Conv2d activation per-channel feasibility。

#### E005 的最终边界

E005 的研究问题是：W4A8 A8 init 崩坏是否主要由 tensor-wise activation range 被 outlier 拉坏导致，并且能否通过 tensor-wise clipping / range calibration 修复。

目前已经完成：

- E005a：Conv2d range diagnostics。
- E005b：Conv2d percentile clipping。
- E005c：Conv2d max / MSE range calibration。
- E005D：结构化 Conv2d tensor-wise clipping 组合与排除实验。

E005 的结论：

- Conv2d activation 是敏感区域。
- Conv2d 的 outlier、relative MSE、per-channel imbalance 均明显强于 Linear。
- 但 tensor-wise percentile clipping、MSE/max range calibration、结构化组合 clipping 都没有恢复 128-sample SNR。
- 因此 E005 不应继续扩展 tensor-wise clipping sweep。

E005 的作用不是“修复成功”，而是排除了一个重要方向：单纯修 tensor-wise range / outlier clipping 不足以解决当前 W4A8 激活量化失败。

#### E006 的新定位

E006 的研究问题是：W4A8 A8 init 崩坏是否来自 tensor-wise activation 量化粒度过粗，尤其是 Conv2d activation 的通道间尺度差异。

E006 不再以 outlier clipping 为主线，而以量化粒度为主线：

- Conv2d activation per-channel。
- Conv2d activation group-wise。
- structure-wise granularity，例如只对 fusion / merge_proj / stage_output_conv / stage5 使用更细粒度。
- Linear / transformer 作为 sanity check，避免修复 Conv2d 后把瓶颈转移到 attention 或 Linear。

#### E006a 进入前注意事项

- 不能直接把现有 `channel_wise=True` 用到 activation 上。
  - 当前 `channel_wise=True` 是权重量化路径，默认按权重第 0 维做输出通道。
  - Conv2d activation 通常是 `[N, C, H, W]`，per-channel 应该按 `C` 维，即 dim=1。
- E006a 应先做 feasibility，不应一开始就做完整部署策略。
- E006a 的最小目标：
  - 给 Conv2d activation quantizer 写入 `[1, C, 1, 1]` 形状的 `delta/zero_point`。
  - 验证 forward 广播正确。
  - 验证 checkpoint 保存/恢复正确。
  - 验证 E001/E005 diagnostics 能读取 per-channel activation quantizer。
  - 跑小样本 smoke，再跑 128-sample eval。
- 如果 E006a 有明显恢复，再进入 group-wise 和结构化粒度；如果仍无效，再转入 E007 / mixed precision / selective FP32 activation quantizer。

### E006：Activation 量化粒度实验总体计划

#### 总目标

E006 用来验证一个新的核心假设：

> 当前 W4A8 激活量化失败，不是因为 tensor-wise range 没调好，而是因为 tensor-wise activation 量化粒度过粗，无法处理 Conv2d activation 的通道间尺度差异和结构差异。

E006 的目标不是继续调 outlier clipping，而是回答：

- Conv2d activation per-channel 是否能显著恢复 128-sample SNR。
- 如果 per-channel 有效，group-wise 是否能以较低复杂度获得主要收益。
- 是否必须全 Conv2d 使用细粒度，还是只处理 fusion / merge_proj / stage_output_conv / stage5 等结构即可。
- 细粒度策略是否会把瓶颈转移到 Linear / transformer。

统一 baseline：

- all_on A8 init：128-sample SNR mean `-7.1021 dB`。
- all_off / W4A32 近似：128-sample SNR mean `4.7973 dB`。
- 评估口径：128-sample eval，`seed=20260427`，`batch_size=16`，优先 CUDA 多卡 job-level 并行。
- 起点 checkpoint：E002b W4 weight-recon checkpoint，不重跑 W20000。
- 默认不跑 activation reconstruction；先只看 A8 init 粒度本身能否恢复。

#### E006a：Conv2d activation per-channel feasibility

目的：先验证 activation per-channel 在当前代码体系中能不能正确工作，而不是立刻追求最优效果。

要做到：

- 新增或扩展 activation range helper，使其支持 Conv2d activation per-channel range。
- 对 4D activation `[N, C, H, W]` 按 `C` 维统计 range。
- 写入 `delta/zero_point` 形状为 `[1, C, 1, 1]`。
- 验证 `UniformAffineQuantizer.forward()` 中 `x / delta` 和反量化广播正确。
- 验证 checkpoint 保存/恢复后仍保留 per-channel activation `delta/zero_point` shape。
- 验证 E001/E005 diagnostics 能读取和报告 per-channel activation quantizer。

实验：

- E006a-0：单元测试 + toy model。
- E006a-1：2-sample smoke。
- E006a-2：all Conv2d per-channel A8 init，128-sample eval。
- E006a-3：E005a diagnostics 复核。

验收：

- smoke 中 `activation_quantizers=52`。
- `non_positive_delta_count=0`。
- checkpoint reload 后推理可运行。
- 如果 all Conv2d per-channel 相对 all_on 提升 `> +1 dB`，说明粒度假设成立，应继续 E006b/E006c。
- 如果提升 `< +0.2 dB`，说明单纯 per-channel 也不足，应尽快转向 mixed precision / selective FP32。

#### E006b：Conv2d activation group-wise feasibility

目的：如果 per-channel 有效，验证是否能用更低复杂度的 group-wise 方案获得主要收益。

候选 group 设置：

- group size = 4 channels。
- group size = 8 channels。
- group size = 16 channels。
- 如果通道数不能整除，最后一组单独处理。

要做到：

- 对 Conv2d activation 的 `C` 维分组。
- 每组共享一组 `delta/zero_point`。
- 广播 shape 可以设计为 `[1, C, 1, 1]`，其中同组通道使用相同值，先保证 forward 和 checkpoint 兼容。
- 记录 group size、group count、metadata proxy。

实验：

- all Conv2d group-wise，分别测试 group size 4/8/16。
- 使用与 E006a 相同 128-sample eval 和 diagnostics。

验收：

- 如果 group-wise 接近 per-channel 收益，例如达到 per-channel 恢复量的 70% 以上，则 group-wise 是更有部署价值的候选。
- 如果 group-wise 明显弱于 per-channel，则 E006 后续可以保留 per-channel 作为上限实验，部署策略转向结构化细粒度或 mixed precision。

#### E006c：结构化粒度对照

目的：判断是否必须全 Conv2d 使用细粒度，还是只处理高敏感结构即可。

候选结构：

- all Conv2d。
- `branch=fusion + module_type=Conv2d`。
- `role=merge_proj + module_type=Conv2d`。
- `role=split_proj + module_type=Conv2d`。
- `role=stage_output_conv + module_type=Conv2d`。
- `stage=stage5 + module_type=Conv2d`。
- 组合：`split_proj + merge_proj + stage_output_conv`。

实验：

- 使用 E006a/E006b 中效果最好的粒度策略。
- 对上述结构分别启用细粒度，其余 activation quantizer 保持 tensor-wise。
- 记录 128-sample SNR/SSIM、diagnostics、selected quantizer count。

验收：

- 如果少数组结构能恢复大部分收益，则 E006 可以形成结构感知低成本策略。
- 如果只有 all Conv2d 有效，则说明粒度问题是广泛累积误差，不是少数结构主导。
- 如果所有结构化粒度都无效，但 all Conv2d 有效，则后续部署需要权衡精度和 metadata 成本。

#### E006d：Linear / transformer sanity check

目的：确认 Conv2d 细粒度修复不会隐藏新的 transformer / Linear 瓶颈。

要做到：

- 对 E006 最优 Conv2d 策略运行 E001 diagnostics。
- 对比 Linear / transformer：
  - effective int levels min/mean。
  - fake-quant relative MSE max/mean。
  - attention qkv/proj 是否进入 worst layer。
- 必要时补一个 Linear per-channel / per-token 的只诊断实验，但不作为 E006 第一主线。

验收：

- 如果 Conv2d 修复后 SNR 明显恢复，而 Linear / transformer 指标仍稳定，则主瓶颈确实在 Conv2d activation 粒度。
- 如果 Conv2d 修复后 Linear / transformer 成为新的 worst group，则下一阶段需要单独设计 attention / Linear 粒度或 mixed precision 策略。

#### E006e：策略收束

目的：把 E006 的粒度实验转化为后续可用策略，而不是无限扩展实验。

需要输出：

- per-channel 上限效果。
- group-wise 成本 / 效果折中。
- 结构化粒度策略表。
- 与 E004 sensitivity 表、E005 range/clipping 负结果的联合结论。

最终判断：

- 若 per-channel / group-wise 明显有效：E006 成为下一阶段主创新方向。
- 若 per-channel 有效但 group-wise 不够：考虑 selective per-channel 或 selective FP32。
- 若 per-channel 仍无效：停止细粒度方向，进入 E007 mixed precision / selective FP32 / reconstruction objective。

#### E006 的风险和注意事项

- 不要直接复用 weight `channel_wise=True`。
- per-channel activation 会改变 `delta/zero_point` shape，必须先确认 checkpoint restore 支持。
- 先不跑 activation reconstruction，避免把“粒度初始化效果”和“重建优化稳定性”混在一起。
- 所有正式结论必须基于 128-sample eval；single-sample 只用于 smoke。
- 每次涉及 activation quantization 的代码或实验，必须同步记录到本日志和 `DEVELOPMENT_LOG.md`。

### 2026-05-06 E006a：Conv2d activation per-channel feasibility

#### 目标

E006a 用来验证 Conv2d activation tensor-wise A8 粒度是否过粗。第一版只做 all Conv2d activation per-channel feasibility：

- 不重跑 W20000。
- 从 E002b W4 weight-recon checkpoint 出发。
- 不跑 activation reconstruction。
- 使用 per-channel MSE grid，而不是 max，尽量只改变 activation 粒度这一变量。
- Conv2d activation output `[N, C, H, W]` 按 C 维统计，写入 `delta/zero_point` shape `[1, C, 1, 1]`。

#### 工具改动

- `quant/activation_range.py`
  - 新增 `activation_granularity`，支持 `tensor` 和 `per_channel`。
  - `per_channel` 当前只支持 `mse_grid`。
  - 对 4D activation 逐通道执行 MSE grid range search。
  - per-channel 写入时如果原 `delta` 是 scalar `nn.Parameter`，会替换成新 shape 的 `nn.Parameter`。
  - summary 中记录 `activation_granularity`、`channel_count`、`delta_shape` 和 `zero_point_shape`。
- `cli/activation_only_quantize_scrn.py`
  - 新增 `--activation-granularity {tensor,per_channel}`。
  - 默认 `tensor`，保持 E005 配置兼容。
- `quant/activation_diagnostics.py`
  - per-channel activation fake-quant diagnostics 对大 4D tensor 做保持通道维的确定性采样。
- 新增配置：
  - `configs/activation_quantization/e006a_conv2d_per_channel.json`

#### TDD 与最小验证

新增/更新测试：

- `tests/test_activation_range.py`
  - per-channel MSE 对 4D Conv2d activation 写出 `[1, C, 1, 1]`。
  - per-channel MSE 拒绝非 4D activation output。
  - selected Conv2d 改成 per-channel 时，未选 Linear 保持原 shape。
- `tests/test_activation_only_quantize_scrn.py`
  - parser 接受 `--activation-granularity per_channel`。
  - normalize config 默认 `tensor`，可解析 `per_channel`。
- `tests/test_activation_diagnostics.py`
  - per-channel activation quantizer fake-quant stats 可运行。
  - 大 4D per-channel fake-quant stats 会按通道采样。
- `tests/test_evaluate_quantized_scrn.py`
  - checkpoint restore 支持 `[1, C, 1, 1]` activation `delta/zero_point` 严格加载并 forward。

验证命令：

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_range`
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn`
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_quantized_scrn`
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/activation_range.py SCRN_BRECQ_app/scrn_brecq/quant/activation_diagnostics.py SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.activation_only_quantize_scrn --help`
- `conda run -n quant python -m json.tool SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/e006a_conv2d_per_channel.json`

#### E006a-0 smoke

- run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/quant/20260506_210819_e006a_smoke_conv2d_per_channel_mse/`
- device：`cuda:1`
- config：
  - `num_samples=2`
  - `init_batch_size=2`
  - `activation_range_method=mse_grid`
  - `activation_granularity=per_channel`
  - `range_module_type=Conv2d`
  - `skip_act_recon=true`
- result：
  - selected Conv2d quantizers：31
  - first selected layer：`model.head`
  - first selected layer `delta/zero_point` shape：`[1, 64, 1, 1]`
  - `activation_quantizers=52`
  - `activation_delta_count=52`
  - `activation_zero_point_count=52`
  - `non_positive_delta_count=0`
  - single-sample `quant_pre_act_recon_snr_db=5.1128 dB`

Reload smoke：

- run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/eval/20260506_210901_e006a_smoke_reload_eval2/`
- 2-sample reload eval 成功，说明 `[1, C, 1, 1]` activation checkpoint 可以恢复并 forward。

#### E006a-1 formal all Conv2d per-channel MSE

统一 baseline：

- all_on A8 init：128-sample SNR mean `-7.1021 dB`。
- all_off / W4A32 近似：128-sample SNR mean `4.7973 dB`。
- all_on 到 all_off gap：`11.8993 dB`。

run 目录：

- quant：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/quant/20260506_210914_e006a_conv2d_per_channel_mse/`
- 128-sample eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/eval/20260506_211010_e006a_conv2d_per_channel_eval128/`
- 128-sample diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/diagnostics/20260506_211053_e006a_conv2d_per_channel_diagnostics128/`

| run | selected | single-sample SNR | 128-sample SNR mean | median | min | max | SSIM mean | delta vs all_on | recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all Conv2d per-channel MSE | 31 | 3.2283 | -5.3817 | -5.9146 | -12.3092 | 5.9752 | 0.5549 | +1.7203 | 0.145 |

Diagnostics：

- `activation_quantizers=52`
- `activation_delta_count=52`
- `activation_zero_point_count=52`
- `non_positive_delta_count=0`
- selected Conv2d quantizers：31
- selected Conv2d channel count：min `32`，max `64`
- selected Conv2d `delta/zero_point` shape：`[1, C, 1, 1]`
- tail output quantizer 保持 scalar 且 disabled。
- `effective_int_levels_min=139`
- Conv2d effective int level mean：`250.5625`
- Linear effective int level min：`139`
- Linear fake-quant relative MSE max：`0.0002060855`

#### E006a 结论

E006a 给出强正信号：

- all Conv2d per-channel MSE 让 128-sample SNR mean 从 `-7.1021 dB` 恢复到 `-5.3817 dB`。
- 提升 `+1.7203 dB`，超过预设的 `+1 dB` 强信号阈值。
- 说明 tensor-wise Conv2d activation 粒度过粗确实是当前 W4A8 A8 init 崩坏的重要因素。

但这不是完整修复：

- per-channel 只恢复了 all_on 到 all_off gap 的约 `14.5%`。
- 结果仍远低于 W4A32 / all_off 的 `4.7973 dB`。
- Conv2d diagnostics 中局部 fake-quant MSE / relative MSE 仍有异常层，说明更细粒度改善了最终输出，但没有消除所有 activation 误差来源。

下一步：

1. 进入 E006b：Conv2d activation group-wise feasibility，优先 group size 4/8/16。
2. 进入 E006c：结构化 per-channel 对照，判断 all Conv2d 是否必要，还是只处理 fusion / merge_proj / stage_output_conv / stage5。
3. 保留 Linear / transformer sanity check；当前 Linear 指标没有变成 A8 init 主瓶颈。

### 2026-05-06 E006a default single-sample sanity check

#### 目的

使用默认单图测试对复核 E006a：

- clean：`SCRN-main/test_data/clear.npy`
- degraded：`SCRN-main/test_data/noise_and_miss.npy`

这次不是替代 128-sample eval，而是做可视化 sanity check，并生成七面板图，把旧 tensor-wise W4A8 init 和 E006a Conv2d per-channel W4A8 init 放在同一张图里。

#### 运行

- E006a checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/quant/20260506_210914_e006a_conv2d_per_channel_mse/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- eval run：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/single_eval/20260506_213432_e006a_clear_noise_single_cpu/`
- 命令：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn --checkpoint ... --eval-clean-path SCRN-main/test_data/clear.npy --eval-input-path SCRN-main/test_data/noise_and_miss.npy --device cpu --save-figure`
- 设备说明：
  - `evaluate_quantized_scrn.py` 目前没有 `--cuda-device-index`。
  - 尝试用 `CUDA_VISIBLE_DEVICES=1` 跑该 CLI 时当前环境报告 `torch.cuda.is_available() is False`。
  - 因单图计算量很小，本次改用 CPU，避免默认占用 GPU 0。

#### 七面板内容

七面板不是重新跑 W20000，也不是跑 activation reconstruction。面板来源如下：

1. Ground Truth：`clear.npy`
2. Input：`noise_and_miss.npy`
3. FP32：E002b source run 的 `fp32_prediction.npy`
4. W4A32 pre weight recon：E002b source run 的 `quant_pre_recon_prediction.npy`
5. W4A32 post weight recon：E002b source run 的 `quant_post_weight_recon_prediction.npy`
6. W4A8 tensor-wise init：E002b source run 的 `quant_pre_act_recon_prediction.npy`
7. W4A8 E006a per-channel init：本次 E006a checkpoint reload eval 的 `prediction.npy`

输出：

- `comparison.png`：`evaluate_quantized_scrn.py` 自带三面板图。
- `seven_panel_tensor_vs_e006a.png`：自定义七面板图。
- `seven_panel_metrics.json`：七面板指标。

#### 结果

| panel | SNR dB | SSIM |
|---|---:|---:|
| Input | 3.9693 | 0.6053 |
| FP32 | 11.7869 | 0.8697 |
| W4A32 pre weight recon | 11.4071 | 0.8255 |
| W4A32 post weight recon | 11.6961 | 0.8660 |
| W4A8 tensor-wise init | 4.9875 | 0.6576 |
| W4A8 E006a per-channel init | 3.2318 | 0.5931 |

差值：

- E006a per-channel vs tensor-wise init：`-1.7556 dB`。
- E006a per-channel vs W4A32 post weight recon：`-8.4643 dB`。

#### 结论

默认单图上，E006a per-channel A8 init 反而低于旧 tensor-wise A8 init。这和 128-sample eval 的方向相反：

- 128-sample：E006a per-channel 相对 all_on 提升 `+1.7203 dB`。
- 默认单图：E006a per-channel 相对 tensor-wise init 下降 `-1.7556 dB`。

因此这次单样本 sanity check 进一步证明：单样本 SNR 只能用于 smoke / 可视化，不适合作为 E006 粒度策略的正式判断依据。正式结论仍以固定 128-sample eval 为准。

### 2026-05-06 E006b：Conv2d activation group-wise feasibility implementation and smoke

E006b 目标：

> 在 E006a per-channel 给出 `+1.7203 dB` 强信号后，验证 group-wise 是否能以更低复杂度接近 per-channel 收益。

实现口径：

- 新增 `activation_granularity=group_wise`。
- 新增 `activation_group_size`。
- E006b 只支持 `group_wise + mse_grid`。
- 仅支持 Conv2d 4D activation `[N, C, H, W]`。
- 按 C 维连续分组，最后一组允许不足 group size。
- `delta/zero_point` 仍写为 `[1, C, 1, 1]`，同组 channel 重复相同 scale / zero point。
- 这样可以复用 E006a 已验证的 forward 广播、checkpoint restore 和 diagnostics 兼容路径。

新增配置：

- `configs/activation_quantization/e006b_conv2d_group_wise_g4.json`
- `configs/activation_quantization/e006b_conv2d_group_wise_g8.json`
- `configs/activation_quantization/e006b_conv2d_group_wise_g16.json`

新增 / 更新测试：

- `test_activation_range.py`
  - group-wise MSE 对 4D Conv2d 写出 `[1, C, 1, 1]`。
  - 同一 group 内 `delta/zero_point` 重复。
  - 非 4D output 报错。
  - `activation_group_size` 缺失或非正时报错。
  - selected Conv2d group-wise 不改 Linear activation delta shape。
- `test_activation_only_quantize_scrn.py`
  - parser 接受 `--activation-granularity group_wise --activation-group-size 8`。
  - 默认仍是 `tensor`，`activation_group_size=None`。
  - E006b 三个配置可解析。
- `test_activation_diagnostics.py`
  - repeated `[1, C, 1, 1]` group-wise scale fake-quant stats 可运行，`non_positive_delta_count=0`。

最小检查：

- `test_activation_range`：19 tests OK。
- `test_activation_only_quantize_scrn`：15 tests OK。
- `test_activation_diagnostics + test_evaluate_quantized_scrn`：11 tests OK。
- `py_compile` passed。
- CLI `--help` 已确认新参数。

#### E006b-0 smoke：group size 8

run：

- quant：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/quant/20260506_220432_e006b_smoke_group_wise_g8_mse/`
- checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/quant/20260506_220432_e006b_smoke_group_wise_g8_mse/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- reload eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/eval/20260506_220506_e006b_smoke_g8_reload_eval2/`
- diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/diagnostics/20260506_220522_e006b_smoke_g8_diagnostics2/`

smoke 结果：

| item | value |
|---|---:|
| group size | 8 |
| selected Conv2d quantizers | 31 |
| activation quantizers | 52 |
| activation delta count | 52 |
| activation zero point count | 52 |
| non-positive delta count | 0 |
| smoke single pre-act SNR | 7.3547 dB |
| diagnostics activation stat count | 52 |
| diagnostics fake quant MSE max | 7.977941277204081e-05 |

复核：

- selected Conv2d activation `delta/zero_point` shape 为 `[1, C, 1, 1]`。
- group size 8 时，64-channel Conv2d 通常有 8 组，32-channel Conv2d 通常有 4 组。
- checkpoint reload eval completed on `cuda:1`，说明保存/恢复和 forward 兼容。

当前结论：

- E006b group-wise 代码路径已通过 smoke。
- smoke 单图 SNR 不作为正式效果证据。
- 下一步进入 E006b-1：g4 / g8 / g16 固定 128-sample eval 和 128-sample diagnostics。

#### E006b-1 formal：group size 4 / 8 / 16

统一口径：

- 起点 checkpoint：E002b W4 weight-recon checkpoint。
- activation range：`mse_grid`。
- activation granularity：`group_wise`。
- `skip_act_recon=true`。
- selected quantizers：all Conv2d activation quantizers，`selected_count=31`。
- eval：128 samples，`seed=20260427`，`batch_size=16`。
- baseline：
  - all_on tensor-wise A8 init：`-7.1021 dB`。
  - E006a all Conv2d per-channel MSE：`-5.3817 dB`。
  - E006a per-channel recovery：`+1.7203 dB`。
  - E006b 70% threshold：`+1.204 dB`。

run paths：

- g4 quant：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/quant/20260506_220727_e006b_conv2d_group_wise_g4_mse/`
- g8 quant：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/quant/20260506_220727_e006b_conv2d_group_wise_g8_mse/`
- g16 quant：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/quant/20260506_220727_e006b_conv2d_group_wise_g16_mse/`
- g4 eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/eval/20260506_220836_e006b_group_wise_g4_eval128/`
- g8 eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/eval/20260506_220837_e006b_group_wise_g8_eval128/`
- g16 eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/eval/20260506_220837_e006b_group_wise_g16_eval128/`
- diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/diagnostics/20260506_220911_e006b_group_wise_g4_diagnostics128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/diagnostics/20260506_220912_e006b_group_wise_g8_diagnostics128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/diagnostics/20260506_220912_e006b_group_wise_g16_diagnostics128/`

结果：

| group size | SNR mean | median | min | max | SSIM mean | delta vs all_on | recovery vs E006a gain | improved vs all_on |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | -6.1885 | -6.8713 | -13.2416 | 6.2019 | 0.4033 | +0.9136 | 53.1% | 120 / 128 |
| 8 | -8.5523 | -9.4064 | -15.9452 | 6.6961 | 0.3927 | -1.4502 | -84.3% | 3 / 128 |
| 16 | -7.4376 | -8.2121 | -14.7473 | 6.7325 | 0.2547 | -0.3355 | -19.5% | 1 / 128 |

diagnostics：

| group size | activation quantizers | activation stat count | non-positive delta count | fake quant MSE max |
|---:|---:|---:|---:|---:|
| 4 | 52 | 52 | 0 | 0.023934995755553246 |
| 8 | 52 | 52 | 0 | 0.023966144770383835 |
| 16 | 52 | 52 | 0 | 0.009504307061433792 |

结论：

- g4 是唯一有正收益的 group-wise 设置：
  - all_on `-7.1021 dB` -> g4 `-6.1885 dB`
  - mean gain `+0.9136 dB`
  - 120 / 128 samples 优于 all_on
- 但 g4 只达到 E006a per-channel 恢复量的约 `53.1%`，没有达到预设 `70%` / `+1.204 dB` 阈值。
- g8 / g16 不但没有接近 per-channel，均值还低于 all_on，说明简单连续 group-wise 在较大 group 下不稳定。
- 所有正式 run 的 `non_positive_delta_count=0`，说明这次失败不是非法 scale，而是量化粒度 / 分组方式本身不足。

E006b 总结：

> Conv2d activation group-wise 有方向信号，但当前简单连续分组不能作为 per-channel 的高质量替代。E006 后续应进入 E006c 结构化粒度对照：保留 per-channel 作为上限，重点测试 selective per-channel / selective group-wise 是否能用少数结构恢复主要收益。

### 2026-05-07 E006c：structured activation granularity configs and smoke

E006c 目标：

> 验证 E006a 的 per-channel 收益是否必须来自 all Conv2d 多层累积，还是少数结构使用细粒度就能恢复主要收益。

背景：

- all_on tensor-wise A8 init：`-7.1021 dB`。
- E006a all Conv2d per-channel：`-5.3817 dB`，提升 `+1.7203 dB`。
- E006b all Conv2d g4：`-6.1885 dB`，提升 `+0.9136 dB`，只恢复 E006a 收益的 `53.1%`。
- E006c acceptance threshold：selective per-channel 相对 all_on 提升 `>= +1.204 dB`，即达到 E006a 收益的 70%。

实现口径：

- 不改核心量化代码。
- 复用 E006a / E006b 已验证能力：
  - `activation_granularity=per_channel`
  - `activation_granularity=group_wise`
  - `activation_group_size=4`
  - `range_selector_groups`
- 不跑 activation reconstruction。
- 不重跑 W20000。

新增配置：

| config | granularity | selector | expected selected count |
|---|---|---|---:|
| `e006c_pc_fusion.json` | per_channel | `branch=fusion,module_type=Conv2d` | 10 |
| `e006c_pc_split_proj.json` | per_channel | `role=split_proj,module_type=Conv2d` | 5 |
| `e006c_pc_merge_proj.json` | per_channel | `role=merge_proj,module_type=Conv2d` | 5 |
| `e006c_pc_stage_output_conv.json` | per_channel | `role=stage_output_conv,module_type=Conv2d` | 5 |
| `e006c_pc_stage5.json` | per_channel | `stage=stage5,module_type=Conv2d` | 6 |
| `e006c_pc_split_merge_stage_output.json` | per_channel | `split_proj + merge_proj + stage_output_conv` | 15 |
| `e006c_g4_split_merge_stage_output.json` | group_wise g4 | `split_proj + merge_proj + stage_output_conv` | 15 |

配置统一口径：

- 起点 checkpoint：E002b W4 weight-recon checkpoint。
- calibration：`num_samples=64`、`init_batch_size=64`。
- activation range：`mse_grid`。
- `skip_act_recon=true`。
- formal eval：128 samples，`seed=20260427`，`batch_size=16`。

测试：

- TDD red：
  - 新增 E006c config parse 测试后，`test_activation_only_quantize_scrn` 因 7 个配置缺失失败。
- Green：
  - `test_activation_only_quantize_scrn`：17 tests OK。
  - `py_compile activation_only_quantize_scrn.py activation_range.py`：passed。
  - CLI `--help` 已确认 selector / granularity 参数仍可用。

#### E006c-0 smoke：split + merge + stage_output selective per-channel

run：

- quant：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_140733_e006c_smoke_pc_split_merge_stage_output_mse/`
- checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_140733_e006c_smoke_pc_split_merge_stage_output_mse/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- reload eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/eval/20260507_140757_e006c_smoke_pc_split_merge_stage_output_reload_eval2/`
- diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/diagnostics/20260507_140828_e006c_smoke_pc_split_merge_stage_output_diagnostics2/`

smoke 结果：

| item | value |
|---|---:|
| selected Conv2d quantizers | 15 |
| activation quantizers | 52 |
| activation delta count | 52 |
| activation zero point count | 52 |
| non-positive delta count | 0 |
| smoke single pre-act SNR | 6.9413 dB |
| diagnostics activation stat count | 52 |
| diagnostics fake quant MSE max | 9.835336823016405e-05 |

复核：

- 15 个 selected Conv2d 正确对应 `split_proj + merge_proj + stage_output_conv`。
- selected Conv2d activation `delta/zero_point` shape 为 `[1, C, 1, 1]`。
- checkpoint reload eval completed on `cuda:1`，说明 selective per-channel checkpoint 可保存、恢复并 forward。

当前结论：

- E006c 配置、测试、smoke、checkpoint restore 和 diagnostics 已通过。
- smoke 单图仍只用于 sanity check，不作为策略判断。
- 下一步执行 6 个 selective per-channel 的固定 128-sample eval 和 diagnostics，再决定是否补跑单结构 g4。

#### E006c-1 formal：selective per-channel

统一口径：

- 起点 checkpoint：E002b W4 weight-recon checkpoint。
- activation range：`mse_grid`。
- activation granularity：`per_channel`。
- `skip_act_recon=true`。
- eval：128 samples，`seed=20260427`，`batch_size=16`。
- improved count：按 `path` 对齐 E004b all_on baseline per-sample metrics。
- baseline：
  - all_on tensor-wise A8 init：`-7.1021 dB`。
  - E006a all Conv2d per-channel：`-5.3817 dB`。
  - E006b all Conv2d g4：`-6.1885 dB`。
  - E006c 70% threshold：`+1.204 dB`。

run paths：

- quant：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141108_e006c_pc_fusion_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141108_e006c_pc_split_proj_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141108_e006c_pc_merge_proj_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141227_e006c_pc_stage_output_conv_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141227_e006c_pc_stage5_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141227_e006c_pc_split_merge_stage_output_mse/`
- eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/eval/20260507_141346_e006c_pc_fusion_eval128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/eval/20260507_141346_e006c_pc_split_proj_eval128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/eval/20260507_141346_e006c_pc_merge_proj_eval128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/eval/20260507_141412_e006c_pc_stage_output_conv_eval128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/eval/20260507_141413_e006c_pc_stage5_eval128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/eval/20260507_141413_e006c_pc_split_merge_stage_output_eval128/`

结果：

| run | selected | SNR mean | median | min | max | SSIM mean | delta vs all_on | recovery vs E006a gain | improved vs all_on |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fusion | 10 | -5.7299 | -6.3070 | -12.8216 | 6.4301 | 0.1551 | +1.3722 | 79.8% | 124 / 128 |
| split_proj | 5 | -6.8865 | -7.6169 | -14.0823 | 6.4158 | 0.1817 | +0.2156 | 12.5% | 122 / 128 |
| merge_proj | 5 | -5.9768 | -6.6454 | -13.1202 | 6.7905 | 0.1649 | +1.1253 | 65.4% | 128 / 128 |
| stage_output_conv | 5 | -4.9720 | -5.5819 | -11.9419 | 6.8071 | 0.2423 | +2.1301 | 123.8% | 128 / 128 |
| stage5 | 6 | -10.7413 | -11.7262 | -18.3867 | 6.7568 | 0.4661 | -3.6392 | -211.5% | 1 / 128 |
| split + merge + stage_output | 15 | -2.1925 | -2.4806 | -8.5691 | 6.8545 | 0.2729 | +4.9096 | 285.4% | 124 / 128 |

diagnostics：

| run | activation quantizers | activation stat count | non-positive delta count | fake quant MSE max |
|---|---:|---:|---:|---:|
| fusion | 52 | 52 | 0 | 9.82716228463687e-05 |
| split_proj | 52 | 52 | 0 | 9.82716228463687e-05 |
| merge_proj | 52 | 52 | 0 | 9.82716228463687e-05 |
| stage_output_conv | 52 | 52 | 0 | 9.82716228463687e-05 |
| stage5 | 52 | 52 | 0 | 0.05554213374853134 |
| split + merge + stage_output | 52 | 52 | 0 | 9.82716228463687e-05 |

E006c-1 结论：

- `fusion` selective per-channel 已超过部署价值阈值：
  - gain `+1.3722 dB`
  - E006a gain recovery `79.8%`
- `stage_output_conv` selective per-channel 是单结构中最强：
  - gain `+2.1301 dB`
  - 超过 E006a all Conv2d per-channel 的收益。
- `split + merge + stage_output` 是当前最强 selective per-channel 策略：
  - SNR mean `-2.1925 dB`
  - 相对 all_on 提升 `+4.9096 dB`
  - 明显强于 E006a all Conv2d per-channel `-5.3817 dB`
- `stage5` 独立 per-channel 明显有害：
  - SNR mean `-10.7413 dB`
  - fake quant MSE max `0.05554213374853134`
  - 后续不应把 stage5 作为独立细粒度策略。
- 所有 per-channel formal runs 的 `non_positive_delta_count=0`，这次效果差异不是非法 scale 导致。

#### E006c-2 formal：selective g4 supplement

触发条件：

- E006c-1 中以下单结构 per-channel gain 超过 `+0.8 dB`：
  - fusion：`+1.3722 dB`
  - merge_proj：`+1.1253 dB`
  - stage_output_conv：`+2.1301 dB`
- 因此补跑对应 g4，并保留计划中的 `split + merge + stage_output` g4。

新增补充配置：

- `configs/activation_quantization/e006c_g4_fusion.json`
- `configs/activation_quantization/e006c_g4_merge_proj.json`
- `configs/activation_quantization/e006c_g4_stage_output_conv.json`

追加检查：

- `test_activation_only_quantize_scrn`：17 tests OK。
- `py_compile activation_only_quantize_scrn.py activation_range.py`：passed。

run paths：

- quant：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141721_e006c_g4_fusion_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141721_e006c_g4_merge_proj_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141721_e006c_g4_stage_output_conv_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141820_e006c_g4_split_merge_stage_output_mse/`
- eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/eval/20260507_141913_e006c_g4_fusion_eval128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/eval/20260507_141914_e006c_g4_merge_proj_eval128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/eval/20260507_141914_e006c_g4_stage_output_conv_eval128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/eval/20260507_141931_e006c_g4_split_merge_stage_output_eval128/`

结果：

| run | selected | SNR mean | median | min | max | SSIM mean | delta vs all_on | recovery vs E006a gain | improved vs all_on |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fusion g4 | 10 | -6.1183 | -6.7607 | -13.2373 | 6.7701 | 0.1693 | +0.9838 | 57.2% | 125 / 128 |
| merge_proj g4 | 5 | -6.3012 | -7.0148 | -13.4791 | 6.7940 | 0.1712 | +0.8009 | 46.6% | 128 / 128 |
| stage_output_conv g4 | 5 | -5.8190 | -6.4788 | -12.8934 | 6.7978 | 0.2231 | +1.2831 | 74.6% | 127 / 128 |
| split + merge + stage_output g4 | 15 | -4.3776 | -4.8955 | -11.2607 | 6.7751 | 0.2353 | +2.7245 | 158.4% | 127 / 128 |

diagnostics：

| run | activation quantizers | activation stat count | non-positive delta count | fake quant MSE max |
|---|---:|---:|---:|---:|
| fusion g4 | 52 | 52 | 0 | 9.82716228463687e-05 |
| merge_proj g4 | 52 | 52 | 0 | 9.82716228463687e-05 |
| stage_output_conv g4 | 52 | 52 | 0 | 9.82716228463687e-05 |
| split + merge + stage_output g4 | 52 | 52 | 0 | 9.82716228463687e-05 |

E006c-2 结论：

- g4 仍有部署价值信号：
  - `stage_output_conv g4` gain `+1.2831 dB`，超过 70% threshold。
  - `split + merge + stage_output g4` gain `+2.7245 dB`，明显强于 E006a all Conv2d per-channel。
- g4 组合虽然低于对应 per-channel 组合：
  - per-channel combo `-2.1925 dB`
  - g4 combo `-4.3776 dB`
  - 但 g4 combo 仍强于 all Conv2d per-channel `-5.3817 dB`。

E006c 总结：

> E006c 反驳了“必须 all Conv2d per-channel”的假设。当前最强信号来自结构化 selective 策略，尤其是 `split + merge + stage_output`；stage5 独立细粒度明显有害。后续应从 selective per-channel / selective g4 收束，而不是继续把 all Conv2d per-channel 作为默认策略。

## 2026-05-07 数据集口径修订：stratified calibration 与 legacy-logic test

目的：

- 当前 W4A8 评估依赖 `scrn_quant_10750_0_patches` 作为 calibration 和 multi-eval clean patch pool。
- 为减少 calibration 随机抽样对来源分布的偏差，新增按 `10750_0` 来源区间分层抽样的 1024 calibration clean patch 准备流程。
- 为后续更接近 SCRN 原文的多样本测试口径，新增去掉缺失 Marmousi 后的 478 clean test patch 准备流程。

实现：

- 新增 `data/stratified_scrn_datasets.py` 和 `cli/prepare_scrn_stratified_datasets.py`。
- calibration 来源区间按 `train_data_N.npy` 编号恢复：
  - `1997_2.5D_shots`: `1-300`
  - `7m_shots_0201`: `301-3655`
  - `Anisotropic_FD_Model`: `3656-4405`
  - `Kerry3D`: `4406-4885`
  - `Shots0001_0200`: `4886-10750`
- 1024 calibration quota 采用最大余数法：
  - `28 / 320 / 71 / 46 / 559`
  - 注意：早期讨论中的 `29 / 320 / 71 / 46 / 559` 合计为 `1025`，因此不能作为 1024 calibration 配额。
- test quota：
  - `Anisotropic`: `75`
  - `Kerry3D`: `16`
  - `Shots0001`: `387`
  - total `478`

对激活量化实验的影响：

- 后续 W4A8 calibration 可以显式指定新的 stratified calibration clean patch 目录，避免默认前 1024 或全局随机抽样改变来源比例。
- 后续 multi-eval 可以使用 478 legacy-logic test clean patch 目录，和当前 128-sample 从训练 patch pool 随机抽评估区分开。
- 由于 `10750_0` 没有保存 patch 坐标，当前测试集只能通过 exact hash 排除直接重复，不能声明与训练 patch 空间区域完全隔离。
- 实际生成的 test manifest 记录 `training_hash_excluded_count=10676`，表示有 10676 个候选 patch 因与训练 patch float32 内容 hash 完全相同而被排除。

验证：

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_stratified_scrn_datasets -v`
  - 7 tests OK。
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/data/stratified_scrn_datasets.py SCRN_BRECQ_app/scrn_brecq/cli/prepare_scrn_stratified_datasets.py`
  - passed。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.prepare_scrn_stratified_datasets --mode calibration`
  - generated `1024` clean calibration patches。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.prepare_scrn_stratified_datasets --mode test`
  - generated `478` clean test patches。

## 2026-05-07 激活量化评估数据口径补充：paper-style 5-source dataset rebuild

目的：

- 旧 `10750_0` 训练集来自整文件 sliding-window 后按文件 quota 随机保留，未严格执行 SCRN 原文 Table 2 的 `samples / traces / shots / augmentation` 约束。
- 为后续 W4A8 calibration 和 multi-eval 提供更接近 SCRN 原文的 clean patch pool，新建 paper-style 5-source train / calibration / test 数据集。

实现：

- 新增 `data/paper_scrn_datasets.py` 和 `cli/prepare_scrn_paper_datasets.py`。
- 新 train set：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_train_10750`
  - counts: `300 / 3355 / 750 / 480 / 5865`
  - final total: `10750`
- 新 calibration set：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_cali_1024_stratified`
  - stratified quotas: `28 / 320 / 71 / 46 / 559`
  - final total: `1024`
- 新 test set：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_test_478`
  - counts: `Anisotropic=75, Kerry3D=16, Shots0001=387`
  - final total: `478`

关键口径修订：

- paper-style train/test 默认按 Table 2/3 几何窗口计数保留 patch，不套用旧 `10750_0` 的 `std > 1e-3` 过滤；否则 1997 和 Anisotropic 等来源无法达到原文 patch 数。
- test 默认对新 train set 做 float32 patch SHA-256 exact hash 排除。
- 因部分 deterministic next-region patch 与 train hash 完全重复，test 生成会从训练区域后继续读取连续 SourceX shot 或 Kerry3D trace window，直到满足固定 quota。
- 本次 test manifest 记录：
  - `training_hash_excluded_count=1079`
  - `per_source_region_counts`: `Anisotropic=3, Kerry3D=7, Shots0001=3`
  - `per_source_candidate_counts`: `Anisotropic=225, Kerry3D=168, Shots0001=1173`
  - `per_source_training_hash_excluded_counts`: `Anisotropic=150, Kerry3D=144, Shots0001=785`

对后续 W4A8 实验的影响：

- 后续 calibration 可切换到 `scrn_paper5_cali_1024_stratified`，避免继续绑定旧 `10750_0` 的非 paper-style 构建口径。
- 后续 multi-eval 可切换到 `scrn_paper5_test_478`，和旧 128-sample train-pool eval 明确区分。
- 切换数据口径会影响 W4A8 all_on / all_off baseline，后续 E006/E007 结论需要注明使用的是 legacy calibration/test 还是 paper-style calibration/test。

验证：

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_paper_scrn_datasets -v`
  - 11 tests OK。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.prepare_scrn_paper_datasets --mode all --overwrite`
  - generated train `10750`、calibration `1024`、test `478`。

## 2026-05-07 激活量化评估数据口径补充：paper-style FP32 training candidate

目的：

- 基于 `scrn_paper5_train_10750` 重新训练一个 FP32 SCRN candidate，作为后续 paper-style calibration/test 口径下 BRECQ 的潜在起点。
- 本次不重跑 W4A32 / W4A8 / activation reconstruction，因此不更新任何 activation quantization baseline 结论。

训练记录：

- Train run:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_170001_paper5_10750_ddp3_seed20260425`
- Dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_train_10750`
- GPU / batch:
  - `CUDA_VISIBLE_DEVICES=1,2,3`
  - `world_size=3`
  - per-GPU batch size `32`
  - global batch size `96`
- 与旧 app DDP baseline 的主要差异：
  - dataset 从旧 `10750_0` 变为 paper-style 5-source dataset
  - world size/global batch 从 `4/128` 变为 `3/96`
- 关键训练参数保持：
  - `epochs=80`
  - `lr=0.001`
  - `milestones=20,40,60`
  - `gamma=0.2`
  - `seed=20260425`
  - SCRN config: `dim=64, stage_depths=1,1,1,1,1, head_dim=32, window_size=8, input_resolution=128`
- Train metrics:
  - `best_epoch=74`
  - `best_loss=0.028283805948116685`
  - `last_loss=0.03233760286976966`

旧单样本 eval 口径：

- Eval run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test/20260507_180355_paper5_10750_ddp3_seed20260425_best_eval_gt_colorbar`
- Metrics:
  - `before_snr_db=3.969324203252889`
  - `before_ssim=0.6052755957782698`
  - `after_snr_db=8.286237604245681`
  - `after_ssim=0.7869134500690693`

对后续 W4A8 实验的影响：

- 新 checkpoint 暂记为 paper-style FP32 training candidate，不替代旧 BRECQ 默认 FP32 checkpoint。
- 后续如果以它为起点重跑 BRECQ，需要单独建立新的 W4A32 / W4A8 baseline 表，并明确标注：
  - FP32 checkpoint: `paper5_10750_ddp3_seed20260425`
  - calibration set: legacy stratified 或 paper-style stratified
  - eval set: old 128-sample train-pool eval、legacy 478 test、或 paper-style 478 test
- 旧单样本 eval 显示该 candidate 在 `SCRN-main/test_data/clear.npy` / `noise_and_miss.npy` 上低于旧 `10750_0` baseline，因此不能直接用作量化实验主起点，除非后续多样本 paper-style eval 支持切换。

## 2026-05-07 激活量化评估数据口径补充：FP32 two-model two-testset 478 benchmark

目的：

- 建立两个 FP32 SCRN checkpoint 在两个 478 clean patch test sets 上的固定退化网格 benchmark。
- 本次仍不重跑 BRECQ；结果用于判断后续量化实验应继续使用旧 FP32 起点，还是建立 paper-style FP32 起点的独立量化线。

实现：

- 新增 FP32 multi-eval CLI：
  - `SCRN_BRECQ_app/scrn_repro/cli/evaluate_scrn_multi.py`
- Preset:
  - `fp32-two-model-two-testset-478`
- Models:
  - `old10750_main`: `20260425_192916_four_gpu_train_quant_10750_0/checkpoints/best.pth`
  - `paper5`: `20260507_170001_paper5_10750_ddp3_seed20260425/checkpoints/best.pth`
- Test sets:
  - `legacy478`: `scrn_quant_test_478_legacy_logic`
  - `paper5_478`: `scrn_paper5_test_478`
- Degradation grid:
  - SNR settings: `-2,-1,1,5,10`
  - missing rates: `0.02,0.08,0.18,0.28,0.38`
  - seed: `20260507`

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/test_multi/20260507_184439_fp32_two_model_two_testset_grid478_seed20260507`
- `per_sample_metrics.jsonl`: `47800` rows。
- 每行包含 input/output SNR/SSIM 和 gain，后续可作为 W4A32/W4A8 评估对照口径。

Overall:

| model | testset | output SNR mean / median | output SSIM mean / median |
|---|---|---:|---:|
| old10750_main | legacy478 | 5.6730 / 5.4099 | 0.7527 / 0.7519 |
| old10750_main | paper5_478 | -6.5491 / 5.1644 | 0.8096 / 0.7965 |
| paper5 | legacy478 | 4.7196 / 3.8869 | 0.6787 / 0.6592 |
| paper5 | paper5_478 | -3.0017 / 6.5355 | 0.8821 / 0.8976 |

Paired comparison:

- On `legacy478`, `paper5 - old10750_main`:
  - SNR mean / median: `-0.9534 / -1.2126`
  - SSIM mean / median: `-0.0740 / -0.0802`
- On `paper5_478`, `paper5 - old10750_main`:
  - SNR mean / median: `+3.5473 / +2.9373`
  - SSIM mean / median: `+0.0725 / +0.0703`

Interpretation for quantization:

- `paper5` is better on its corresponding `paper5_478` test set, but worse on `legacy478` overall。
- `paper5_478` has `33` near-zero-energy clean patches; these produce extreme negative SNR values and make SNR mean less stable than SNR median.
- For subsequent W4A8 comparisons, report both mean and median SNR/SSIM, and keep testset identity explicit:
  - `legacy478` for continuity with old data construction。
  - `paper5_478` for paper-style deterministic split evaluation。

验证：

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_repro.tests.test_evaluate_scrn_multi -v`
  - 6 tests OK。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.evaluate_scrn_multi --preset fp32-two-model-two-testset-478 --device cuda --cuda-device-index 1 --batch-size 64 --seed 20260507`
  - completed four groups and wrote `47800` rows。

## 手动记录一下第一次的按原论文设定的数据集效果差主要原因

“我刚看了 clean patch 的能量分布：旧 10750_0 训练集里没有 std <= 1e-3 的 patch；但新的 paper5_train_10750 里有 8400/10750 个 std <= 1e-3，其中 5715 个甚至 std <= 1e-6。这不是“少量低能量背景”，而是训练集主体被近空白 patch 占了。

直接原因也找到了：paper_scrn_datasets.py 里虽然定义了 DEFAULT_MIN_STD = 1e-3，但生成训练集和测试集时默认传的是 min_std=None，也就是为了凑齐论文表格 count，实际关闭了低方差过滤。再加上每个 raw patch 会生成 original + 4 个增强，空白 patch 也被放大成 5 份”

因此决定加上过滤再试一次

## 2026-05-07 W4A8 calibration/eval data protocol update: paper5 energy-filtered datasets

背景：

- 首版 `scrn_paper5_*` 数据集保留了大量近零 clean patch：
  - `scrn_paper5_train_10750`: `8400/10750` patches have `std <= 1e-3`
  - `scrn_paper5_test_478`: previously observed near-zero patches can distort SNR mean
- 这些样本对 SCRN/BRECQ calibration 和 W4A8 evaluation 都是不稳定因素：
  - calibration 可能被无效 activation 分布污染。
  - eval SNR mean 会被极小 clean energy patch 放大成极端负值。

本次更新：

- 新增 energy-filtered clean patch 数据集：
  - train: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_train_10750`
  - calibration: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_cali_1024_stratified`
  - test: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_test_478`
- hard reject 规则：
  - `std <= 1e-3`
  - `absmax <= 1e-3`
  - non-finite patch
  - all-zero / near-zero patch
- calibration 从 filtered train manifest 中分层抽样，且只使用 `augmentation_index=0` 的原始 patch：
  - `28 / 320 / 71 / 46 / 559 = 1024`
- test 从 filtered train 实际使用区域之后开始扫描：
  - `Anisotropic`: shot index `7`
  - `Kerry3D`: trace start `1435`
  - `Shots0001`: shot index `15`

验证结果：

- train:
  - count `10750`
  - `std <= 1e-3`: `0`
  - min std `0.0010023288`
- calibration:
  - count `1024`
  - `std <= 1e-3`: `0`
  - min std `0.0010070483`
- test:
  - count `478`
  - `std <= 1e-3`: `0`
  - min std `0.0010794682`

对后续量化实验的影响：

- 后续 W4A8 activation calibration / eval 应优先使用 energy-filtered calibration/test 协议。
- 旧 `scrn_paper5_*` 结果应标注为 unfiltered diagnostic，不再作为正式 paper-style 量化评估主口径。
- 后续报告仍需同时保留 mean / median SNR 和 SSIM，避免少量异常样本再次掩盖整体趋势。

## 2026-05-07 FP32 baseline update: paper5 energy-filtered training candidate

本次只训练 FP32 SCRN，不进入 BRECQ / W4A8。

目的：

- 为 energy-filtered calibration/test 协议补一个对应的 FP32 training candidate。
- 先判断 `scrn_paper5_energy_filtered_train_10750` 是否修复首版 `paper5_unfiltered` 的近零 patch 污染问题，再决定后续是否用它作为 W4A8 重量重建起点。

Run:

- Train dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_train_10750`
- Train run:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_195614_paper5_energy_filtered_10750_ddp3_seed20260425`
- Checkpoint:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_195614_paper5_energy_filtered_10750_ddp3_seed20260425/checkpoints/best.pth`
- Config note:
  - `world_size=3`
  - per-GPU batch size `32`
  - global batch size `96`
  - git commit `d2ad387cbe7f9cd200ef074bc2de08d02534bfb7`

Training metrics:

- `best_loss=3.8867502100987448`
- `best_epoch=63`
- `last_loss=3.9057623196969784`

Old single-sample eval:

- Eval run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test/20260507_214045_paper5_energy_filtered_10750_ddp3_seed20260425_best_eval_gt_colorbar`
- Metrics:
  - `before_snr_db=3.969324203252889`
  - `before_ssim=0.6052755957782698`
  - `after_snr_db=10.842029016359723`
  - `after_ssim=0.8614298903567633`

Interpretation for W4A8:

- Compared with `paper5_unfiltered`, energy filtering improves the historical single-sample FP32 eval:
  - SNR: `8.2862 -> 10.8420` dB
  - SSIM: `0.7869 -> 0.8614`
- Compared with old `10750_0` main baseline, it is still slightly lower:
  - old main single eval: `after_snr_db=11.78722661219287`, `after_ssim=0.8699862043155245`
- This checkpoint is a plausible FP32 candidate for future BRECQ runs, but it should not replace the old main baseline until 478-patch multi-eval confirms the trend.
- Next quantization comparison should keep dataset identities explicit:
  - old `10750_0` training / legacy test
  - unfiltered paper5 diagnostic
  - energy-filtered paper5 calibration/test protocol

## 2026-05-07 FP32 478-eval check before BRECQ: energy-filtered test set

本次仍只评估 FP32 SCRN，不进入 BRECQ / W4A8。

目的：

- 单样本 eval 中 `paper5_energy_filtered` 优于 `paper5_unfiltered`，但后续量化不能依赖单样本口径。
- 用 `scrn_paper5_energy_filtered_test_478` 做固定退化网格，先确认 FP32 baseline 排序。

Eval run:

- `SCRN_BRECQ_app/scrn_repro/runs/test_multi/20260507_215143_fp32_three_model_energy_filtered_test478_seed20260507`
- Rows:
  - `3` models x `478` clean patches x `25` degradation conditions = `35850`
- Test set:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_test_478`

Overall on `paper5_energy_filtered_478`:

| model | output SNR mean | output SNR median | output SSIM mean | output SSIM median |
|---|---:|---:|---:|---:|
| `old10750_main` | 6.4933 | 6.2893 | 0.8183 | 0.8333 |
| `paper5_unfiltered` | 7.9441 | 7.8208 | 0.8675 | 0.8924 |
| `paper5_energy_filtered` | 6.7422 | 6.6473 | 0.8197 | 0.8441 |

Pairwise:

- `paper5_unfiltered - old10750_main`:
  - SNR mean / median: `+1.4508 / +1.7236`
  - SSIM mean / median: `+0.0492 / +0.0450`
- `paper5_energy_filtered - old10750_main`:
  - SNR mean / median: `+0.2489 / +0.5539`
  - SSIM mean / median: `+0.0014 / -0.0038`
- `paper5_energy_filtered - paper5_unfiltered`:
  - SNR mean / median: `-1.2019 / -1.1296`
  - SSIM mean / median: `-0.0478 / -0.0360`

Implication for quantization:

- `paper5_energy_filtered` is not yet justified as the new main FP32 checkpoint for W4A8 experiments.
- It only slightly improves SNR over `old10750_main` on the corresponding filtered test set, while `paper5_unfiltered` is still best overall on this test set.
- The discrepancy between single-sample eval and 478-eval reinforces the current rule:
  - single sample only for smoke / continuity
  - 478-patch fixed-grid SNR/SSIM is the meaningful FP32 and W4A8 comparison口径
- Before rerunning BRECQ, run the full 3-model x 3-testset FP32 comparison and inspect `Shots0001`, since it dominates `paper5_energy_filtered_478` and drives the current ranking.

## 2026-05-07 Planned FP32 3-model x 3-testset eval before BRECQ

本次仍只评估 FP32 SCRN，不进入 BRECQ / W4A8。

目的：

- 在启动任何新的 BRECQ 之前，先确认三个 FP32 checkpoint 在三个 478 test set 上的稳定排序。
- 避免用单样本 eval 或单一 test set 过早选择 W4A8 起点。

Eval matrix:

- Models:
  - `old10750_main`
  - `paper5_unfiltered`
  - `paper5_energy_filtered`
- Test sets:
  - `legacy478`
  - `paper5_478`
  - `paper5_energy_filtered_478`
- Conditions:
  - SNR settings: `-2,-1,1,5,10`
  - missing rates: `0.02,0.08,0.18,0.28,0.38`
  - seed: `20260507`
- Expected rows:
  - `107550`

判断标准：

- 以 478-patch fixed-grid mean / median SNR 和 SSIM 为主。
- 同时看 by-source 结果，尤其是 `Shots0001`。
- 如果 `paper5_energy_filtered` 不能在多个 test set 上稳定优于旧主 baseline 或 `paper5_unfiltered`，则暂不把它作为 W4A8 主起点。
- 如果 `paper5_unfiltered` 仍然最强，下一步应优先做 dataset energy diagnostics / energy-balanced data protocol，而不是直接进入量化。

## 2026-05-07 FP32 3-model x 3-testset eval result before BRECQ

本次仍只评估 FP32 SCRN，不进入 BRECQ / W4A8。

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/test_multi/20260507_224936_fp32_three_model_three_testset_grid478_seed20260507`
- Rows:
  - `107550`

Overall ranking:

- `legacy478`:
  - best: `old10750_main`
  - SNR mean / median: `5.6730 / 5.4099`
  - SSIM mean / median: `0.7527 / 0.7519`
- `paper5_478`:
  - best: `paper5_unfiltered`
  - SNR mean / median: `-3.0017 / 6.5355`
  - SSIM mean / median: `0.8821 / 0.8976`
  - note: SNR mean is distorted by near-zero clean patches; median SNR and SSIM are more informative.
- `paper5_energy_filtered_478`:
  - best: `paper5_unfiltered`
  - SNR mean / median: `7.9441 / 7.8208`
  - SSIM mean / median: `0.8675 / 0.8924`

`paper5_energy_filtered` summary:

- vs `old10750_main`:
  - `legacy478`: worse by SNR median `-1.7892` and SSIM median `-0.1347`
  - `paper5_478`: similar SNR median `+0.0548`, but worse SSIM median `-0.0183`
  - `paper5_energy_filtered_478`: slightly better SNR median `+0.5539`, but slightly worse SSIM median `-0.0038`
- vs `paper5_unfiltered`:
  - worse on all three test sets by SNR median and SSIM median.

Implication for W4A8:

- Do not use `paper5_energy_filtered` as the main BRECQ starting checkpoint yet.
- The current best FP32 checkpoint depends on test protocol:
  - `old10750_main` for legacy continuity.
  - `paper5_unfiltered` for both paper-style test sets.
- Since `paper5_energy_filtered` fixes low-energy pollution but loses overall FP32 quality, the next step should be energy diagnostics / energy-balanced data construction before any new W4A8 run.
- Keep reporting mean and median SNR/SSIM together, especially for any test set that still contains near-zero clean patches.

## 2026-05-07 Train energy diagnostics before new W4A8 runs

本次仍不进入 BRECQ / W4A8，只分析 FP32 training data。

Report:

- `SCRN_BRECQ_app/scrn_repro/runs/dataset_diagnostics/20260507_232819_train_energy_diagnostics_10750`
- Artifacts:
  - `summary.md`
  - `energy_diagnostics.json`

Purpose:

- Check whether source-count quotas preserve training influence under the current SCRN FP32 loss:
  - training uses `MSELoss(sum)`
  - higher-energy patches can dominate loss even when source count is small
- Use `std² sum` as a rough MSE-scale influence proxy.

Key findings:

| dataset | std <= 1e-3 | dominant std² source | dominant std² share |
|---|---:|---|---:|
| `legacy10750_0` | 0 | `Kerry3D` | 0.9493 |
| `paper5_unfiltered` | 8400 | `Anisotropic_FD_Model` | 0.4976 |
| `paper5_energy_filtered` | 0 | `Kerry3D` | 0.9510 |

Important nuance:

- `paper5_energy_filtered` is not weak merely because `Kerry3D` has high std² share.
- `legacy10750_0` also has high `Kerry3D` std² share and is still best on `legacy478`.
- Therefore the issue is broader than one source:
  - train/test protocol matching matters,
  - spatial region selection matters,
  - near-zero patch distribution matters,
  - energy-scale imbalance still must be controlled before retraining.

Implication for quantization:

- Do not start a new W4A8 run from `paper5_energy_filtered` yet.
- Current BRECQ candidate choice remains unresolved:
  - `old10750_main` is strongest for legacy continuity.
  - `paper5_unfiltered` is strongest on paper-style test sets, but contains many near-zero train patches.
  - `paper5_energy_filtered` has cleaner patches but weaker FP32 evaluation.
- Next dataset protocol should be energy-balanced:
  - keep low-energy rejection,
  - preserve source count quotas,
  - constrain per-source energy distribution or std² share,
  - validate diagnostics before any new FP32 training or W4A8 quantization.

## 2026-05-08 Per-patch absmax normalized dataset candidates

本次不运行 BRECQ / W4A8，只新增后续可能用于 FP32 和量化对照的 clean 数据协议。

Motivation:

- 之前的 train energy diagnostics 显示，source patch 数量比例一致并不等于训练影响一致。
- 每个 patch 的幅值尺度会影响 FP32 `MSELoss(sum)`，也会影响后续 calibration / activation range 统计。
- 新协议先把每个 clean patch 单独做 absmax normalization，用于隔离“振幅尺度差异”这个变量。

Protocol:

- `scale = max(abs(patch))`
- `patch_norm = patch / scale if scale > 1e-12 else patch`
- manifest 中记录 `normalization_scale`，后续可用：
  - `restored = normalized * normalization_scale`
- 反归一化恢复的是原 clean patch 空间，不是 raw SEG-Y 原始振幅空间。

New datasets:

- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_perpatch_absmax_train_10750`
- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_perpatch_absmax_cali_1024_stratified`
- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_perpatch_absmax_test_478`
- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`

Counts and tiny-scale notes:

| dataset | count | tiny scale count |
|---|---:|---:|
| `paper5_perpatch_absmax_train_10750` | 10750 | 5400 |
| `paper5_perpatch_absmax_cali_1024` | 1024 | 504 |
| `paper5_perpatch_absmax_test_478` | 478 | 18 |
| `paper5_energy_filtered_perpatch_absmax_train_10750` | 10750 | 0 |
| `paper5_energy_filtered_perpatch_absmax_cali_1024` | 1024 | 0 |
| `paper5_energy_filtered_perpatch_absmax_test_478` | 478 | 0 |

Implication for W4A8:

- Do not treat these as the new default W4A8 calibration/test protocol yet.
- `paper5_unfiltered_perpatch_absmax` is useful but risky because many near-zero patches remain tiny-scale or can be amplified if just above threshold.
- `paper5_energy_filtered_perpatch_absmax` is the cleaner candidate for testing whether per-patch normalization helps FP32 and activation quantization.
- Next safe order:
  1. Train FP32 on the normalized dataset candidates.
  2. Run fixed-grid 478 eval with mean and median SNR/SSIM.
  3. Only if FP32 is competitive, use the matching normalized calibration/test sets for a W4A8 BRECQ experiment.

## 2026-05-08 Paper5 per-patch absmax FP32 training result

本次仍不运行 BRECQ / W4A8，只记录第一个 normalized FP32 checkpoint。

Train dataset:

- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_perpatch_absmax_train_10750`
- This is the unfiltered paper5 derivative with per-patch absmax normalization.
- tiny scale patch count remains high:
  - `5400 / 10750`

Train run:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_150851_paper5_perpatch_absmax_10750_ddp3_seed20260425`
- 3-GPU DDP on physical GPUs `1,2,3`
- global batch `96`
- same SCRN model and optimizer schedule as recent paper5 runs
- run config git commit:
  - `247c1ee9522f515ef94335d74012fa5f3236a1a0`

Training metrics:

- best epoch: `75`
- best loss: `11.12054773739406`
- last loss: `11.920819751563526`

Historical single-sample eval:

- run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test/20260508_164151_paper5_perpatch_absmax_10750_ddp3_seed20260425_best_eval_gt_colorbar`
- metrics:
  - before SNR / SSIM: `3.9693 dB / 0.6053`
  - after SNR / SSIM: `13.3167 dB / 0.9130`

Quantization implication:

- This is a promising FP32 single-sample result, but it is not enough to choose a W4A8 starting point.
- The dataset still contains many tiny-scale patches, so activation calibration behavior may be risky.
- Do not start BRECQ from this checkpoint yet.
- Next step should train `paper5_energy_filtered_perpatch_absmax`, then run a joint fixed-grid 478 eval before selecting any normalized checkpoint for W4A8.

## 2026-05-08 Paper5 energy-filtered per-patch absmax FP32 training result

本次仍不运行 BRECQ / W4A8，只记录第二个 normalized FP32 checkpoint。

Train dataset:

- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- This is the energy-filtered paper5 derivative with per-patch absmax normalization.
- tiny scale patch count:
  - `0 / 10750`

Train run:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_164907_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425`
- 3-GPU DDP on physical GPUs `1,2,3`
- global batch `96`
- same SCRN model and optimizer schedule as recent paper5 runs
- run config git commit:
  - `c96ebc81444c9066f4035fd35e258c1fa940513d`

Training metrics:

- best epoch: `63`
- best loss: `22.153653881379537`
- last loss: `22.433070646865026`

Historical single-sample eval:

- run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test/20260508_183043_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_best_eval_gt_colorbar`
- metrics:
  - before SNR / SSIM: `3.9693 dB / 0.6053`
  - after SNR / SSIM: `13.5520 dB / 0.9273`

Quantization implication:

- This checkpoint has a cleaner normalized training set than `paper5_perpatch_absmax` because tiny-scale patches are removed.
- Single-sample SNR/SSIM are slightly better than the unfiltered per-patch normalized checkpoint.
- Do not start W4A8 yet; single-sample eval is insufficient.
- Next step should run a fixed-grid 478 multi-eval including:
  - `paper5_perpatch_absmax`
  - `paper5_energy_filtered_perpatch_absmax`
  - existing FP32 baselines as needed for continuity
  - normalized and non-normalized test sets, with mean and median SNR/SSIM.

## 2026-05-08 FP32 5-model x 5-testset 478 eval before normalized W4A8 selection

本次仍未运行 BRECQ / W4A8，只用 FP32 fixed-grid eval 选择后续量化候选。

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/test_multi/20260508_184559_fp32_five_model_five_testset_grid478_seed20260507`
- rows: `298750`
- condition count: `25`
- each model/testset bucket: `11950` rows
- manifest warnings: none

Native-pair FP32 results:

| checkpoint | matching testset | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---|---:|---:|---:|---:|
| `old10750_main` | `legacy478` | 5.6730 | 5.4099 | 0.7527 | 0.7519 |
| `paper5_unfiltered` | `paper5_478` | -3.0017 | 6.5355 | 0.8821 | 0.8976 |
| `paper5_energy_filtered` | `paper5_energy_filtered_478` | 6.7422 | 6.6473 | 0.8197 | 0.8441 |
| `paper5_perpatch_absmax` | `paper5_perpatch_absmax_478` | 11.1449 | 16.1541 | 0.9345 | 0.9833 |
| `paper5_energy_filtered_perpatch_absmax` | `paper5_energy_filtered_perpatch_absmax_478` | 16.7960 | 17.1248 | 0.9615 | 0.9792 |

Activation quantization implication:

- Per-patch absmax normalization is not just a harmless preprocessing detail; it changes the amplitude space seen by both the model and activation quantizers.
- The normalized FP32 checkpoints perform poorly on raw-amplitude testsets, so they should not be used with raw-amplitude calibration/test data.
- The clean normalized candidate is `paper5_energy_filtered_perpatch_absmax`:
  - matching normalized test SNR / SSIM: `16.7960 dB / 0.9615`
  - Shots0001 matching normalized SNR / SSIM: `16.3505 dB / 0.9551`
  - tiny-scale training patches: `0`
- If a normalized W4A8 path is pursued, use the matching normalized calibration/test sets:
  - calibration: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
  - test: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- Raw-amplitude W4A8 and normalized-amplitude W4A8 should be reported as separate protocols.

## 2026-05-08 No-decay normalized FP32 candidate

本次仍未运行 BRECQ / W4A8，只更新 normalized FP32 candidate 的选择依据。

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3`
- dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- only changed variable:
  - `--milestones ""`
  - LR stayed at `0.001` for all 80 epochs

Training comparison:

| checkpoint | best epoch | best loss | last loss |
|---|---:|---:|---:|
| decayed LR | 63 | 22.153653881379537 | 22.433070646865026 |
| no decay LR | 79 | 18.80715600649516 | 19.151402472030547 |

FP32 eval comparison:

| checkpoint | single-sample SNR / SSIM | matching 478 SNR mean / SSIM mean | Shots0001 SNR mean / SSIM mean |
|---|---:|---:|---:|
| decayed LR | 13.5520 / 0.9273 | 16.7960 / 0.9615 | 16.3505 / 0.9551 |
| no decay LR | 13.8807 / 0.9324 | 17.8346 / 0.9644 | 17.3551 / 0.9594 |

Quantization implication:

- The no-decay normalized FP32 checkpoint is now the preferred starting point for a normalized W4A8 experiment.
- Use it only with the normalized amplitude-space protocol:
  - calibration: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
  - test: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- Do not mix this checkpoint with raw-amplitude calibration/test data when judging W4A8 quality.

## 2026-05-08 LR 0.005 normalized FP32 ablation

本次仍未运行 BRECQ / W4A8，只测试 normalized FP32 candidate 的 aggressive LR 训练是否可行。

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_213810_paper5_energy_filtered_perpatch_absmax_10750_ddp4_seed20260425_nodecay_lr5e-3`
- dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- changed variables versus preferred `lr=0.001` run:
  - LR: `0.001 -> 0.005`
  - world size: `3 -> 4`
  - global batch: `96 -> 128`

Training comparison:

| checkpoint | best epoch | best loss | last loss |
|---|---:|---:|---:|
| `lr=0.001`, DDP3, no decay | 79 | 18.80715600649516 | 19.151402472030547 |
| `lr=0.005`, DDP4, no decay | 72 | 1674639.4090401786 | 780769248405156864 |

FP32 eval comparison:

| checkpoint | single-sample SNR / SSIM | matching 478 SNR mean / SSIM mean | Shots0001 SNR mean / SSIM mean |
|---|---:|---:|---:|
| `lr=0.001`, DDP3, no decay | 13.8807 / 0.9324 | 17.8346 / 0.9644 | 17.3551 / 0.9594 |
| `lr=0.005`, DDP4, no decay | -35.8768 / 0.0791 | -38.8719 / 0.1705 | -38.4443 / 0.1842 |

Quantization implication:

- `lr=0.005` is not a viable normalized FP32 starting point for W4A8.
- The preferred normalized W4A8 FP32 starting checkpoint remains:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- Do not use the LR `0.005` checkpoint for calibration, activation reconstruction, or packed export experiments.

## 2026-05-09 LR 0.002 normalized FP32 ablation

本次仍未运行 BRECQ / W4A8，只测试 normalized FP32 candidate 的 moderate LR 训练是否可行。

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_231616_paper5_energy_filtered_perpatch_absmax_10750_ddp4_seed20260425_nodecay_lr2e-3`
- dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- changed variables versus preferred `lr=0.001` run:
  - LR: `0.001 -> 0.002`
  - world size: `3 -> 4`
  - global batch: `96 -> 128`

Training comparison:

| checkpoint | best epoch | best loss | last loss |
|---|---:|---:|---:|
| `lr=0.001`, DDP3, no decay | 79 | 18.80715600649516 | 19.151402472030547 |
| `lr=0.002`, DDP4, no decay | 80 | 24.6274932878358 | 24.6274932878358 |
| `lr=0.005`, DDP4, no decay | 72 | 1674639.4090401786 | 780769248405156864 |

FP32 eval comparison:

| checkpoint | single-sample SNR / SSIM | matching 478 SNR mean / SSIM mean | Shots0001 SNR mean / SSIM mean |
|---|---:|---:|---:|
| `lr=0.001`, DDP3, no decay | 13.8807 / 0.9324 | 17.8346 / 0.9644 | 17.3551 / 0.9594 |
| `lr=0.002`, DDP4, no decay | 13.5487 / 0.9226 | 16.0660 / 0.9546 | 15.6622 / 0.9473 |
| `lr=0.005`, DDP4, no decay | -35.8768 / 0.0791 | -38.8719 / 0.1705 | -38.4443 / 0.1842 |

Quantization implication:

- `lr=0.002` is stable but weaker than the current preferred `lr=0.001` normalized FP32 checkpoint.
- The preferred normalized W4A8 FP32 starting checkpoint remains:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- Do not use the LR `0.002` checkpoint as the default for calibration, activation reconstruction, or packed export experiments.
- Further LR tuning should isolate global batch effects before changing the preferred W4A8 starting point.

## 2026-05-09 LR 0.0015 normalized FP32 ablation

本次仍未运行 BRECQ / W4A8，只测试 normalized FP32 candidate 的 intermediate LR 训练是否可行。

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260509_001355_paper5_energy_filtered_perpatch_absmax_10750_ddp4_seed20260425_nodecay_lr1p5e-3`
- dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- changed variables versus preferred `lr=0.001` run:
  - LR: `0.001 -> 0.0015`
  - world size: `3 -> 4`
  - global batch: `96 -> 128`

Training comparison:

| checkpoint | best epoch | best loss | last loss |
|---|---:|---:|---:|
| `lr=0.001`, DDP3, no decay | 79 | 18.80715600649516 | 19.151402472030547 |
| `lr=0.0015`, DDP4, no decay | 79 | 20.458939271313803 | 20.465225462402618 |
| `lr=0.002`, DDP4, no decay | 80 | 24.6274932878358 | 24.6274932878358 |
| `lr=0.005`, DDP4, no decay | 72 | 1674639.4090401786 | 780769248405156864 |

FP32 eval comparison:

| checkpoint | single-sample SNR / SSIM | matching 478 SNR mean / SSIM mean | Shots0001 SNR mean / SSIM mean |
|---|---:|---:|---:|
| `lr=0.001`, DDP3, no decay | 13.8807 / 0.9324 | 17.8346 / 0.9644 | 17.3551 / 0.9594 |
| `lr=0.0015`, DDP4, no decay | 13.7548 / 0.9374 | 16.9180 / 0.9566 | 16.4150 / 0.9504 |
| `lr=0.002`, DDP4, no decay | 13.5487 / 0.9226 | 16.0660 / 0.9546 | 15.6622 / 0.9473 |
| `lr=0.005`, DDP4, no decay | -35.8768 / 0.0791 | -38.8719 / 0.1705 | -38.4443 / 0.1842 |

Quantization implication:

- `lr=0.0015` is stable and better than `lr=0.002`, but it remains weaker than the current preferred `lr=0.001` normalized FP32 checkpoint on matching 478 eval.
- The preferred normalized W4A8 FP32 starting checkpoint remains:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- Do not use the LR `0.0015` checkpoint as the default for calibration, activation reconstruction, or packed export experiments.
- Current LR sweep evidence indicates that increasing LR under DDP4/global batch `128` does not improve the normalized W4A8 candidate.

## 2026-05-09 Default W4A8 dataset and FP32 starting checkpoint

This section records the default dataset/checkpoint choice for future activation quantization work.

Default dataset family:

- Protocol name:
  - `paper5_energy_filtered_perpatch_absmax`
- Calibration:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- Test:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- Matching train dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`

Default FP32 checkpoint for W4A8:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`

Default FP32 training setup:

- `epochs=80`
- `seed=20260425`
- `batch_size=32` per GPU
- `world_size=3`
- global batch `96`
- `lr=0.001`
- no LR decay: `--milestones ""`
- `weight_decay=0.0`
- SCRN model config:
  - `dim=64`
  - `stage_depths=1,1,1,1,1`
  - `head_dim=32`
  - `window_size=8`
  - `input_resolution=128`

Default FP32 baseline metrics:

| metric | value |
|---|---:|
| best epoch | 79 |
| best loss | 18.80715600649516 |
| single-sample after SNR / SSIM | 13.8807 / 0.9324 |
| matching normalized 478 SNR mean / SSIM mean | 17.8346 / 0.9644 |
| matching normalized 478 SNR median / SSIM median | 18.1752 / 0.9788 |
| Shots0001 SNR mean / SSIM mean | 17.3551 / 0.9594 |

Quantization rule going forward:

- Future W4A8 calibration, activation reconstruction, sensitivity diagnosis, and packed export should default to this checkpoint and the matching normalized calibration/test data above.
- Raw-amplitude datasets and non-normalized checkpoints should not be mixed with this protocol when judging W4A8 quality.
- Older FP32 results remain useful as diagnostics, but not as the default W4A8 starting point.
- Current LR sweep conclusion:
  - `lr=0.001`, DDP3/global batch `96`, no decay is the best current FP32 starting point.
  - `lr=0.0015` and `lr=0.002` under DDP4/global batch `128` are stable but worse.
  - `lr=0.005` under DDP4/global batch `128` is unusable.

## 2026-05-09 E007 fixed-grid quantized evaluator

Implemented the evaluator needed to rebuild W4A32 and later W4A8 baselines on the normalized dataset protocol.

CLI:

- `SCRN_BRECQ_app/scrn_brecq/cli/evaluate_quantized_scrn_grid.py`

Default protocol:

- Test dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- SNR settings: `-2,-1,1,5,10`
- missing rates: `0.02,0.08,0.18,0.28,0.38`
- seed: `20260507`
- full grid size: `478 * 25 = 11950` rows

Why this was needed:

- The earlier `evaluate_quantized_scrn_multi.py` used a sampled evaluation protocol and did not match the normalized FP32 478x25 grid baseline.
- Future W4A32/W4A8 comparisons now have a fixed quantized evaluator that reports FP32, pre-reconstruction, post-reconstruction, and delta metrics under the same degradation grid.

Validation:

- Unit tests:
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_quantized_scrn_grid -v`
  - result: 6 tests passed
- Syntax:
  - `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/evaluate_quantized_scrn_grid.py`
  - result: passed
- CLI help:
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_grid --help`
  - result: passed
- Smoke:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E007_normalized_w4a32_baseline/smoke/20260509_144714_smoke_quantized_grid2`
  - GPU: physical GPU `1`
  - checkpoint: historical W4A8 E002 positive-scale checkpoint, used only for CLI smoke
  - rows: `2`
  - outputs: `config.json`, `metrics.json`, `summary.md`, `per_sample_metrics.jsonl`

## 2026-05-09 E007 normalized W4A32 single-GPU baseline

Rebuilt the W4A32 weight-only baseline on the new normalized default protocol before restarting W4A8 activation quantization experiments.

Protocol:

- Dataset family:
  - `paper5_energy_filtered_perpatch_absmax`
- FP32 checkpoint:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- Calibration:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- Test:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`

W4A32 reconstruction:

- Run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1`
- Checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- Pre-recon checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq_pre_recon.pth`
- GPU: physical GPU `1`
- Settings:
  - `num_samples=1024`
  - `batch_size=16`
  - `iters_w=20000`
  - `n_bits_w=4`
  - `n_bits_a=8`
  - `act_quant=false`
- Single-sample sanity:
  - `fp32_snr=13.8808`
  - `pre_w_snr=13.4675`
  - `post_w_snr=13.8918`
  - `post_recon_ssim=0.9290`
  - `reconstruction_seconds=2597.64`

Verification:

- JSON:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/verification.json`
- `passed=true`
- final state:
  - `weight_quant=true`
  - `act_quant=false`
- weight quantization:
  - quant modules: `52`
  - weight bit counts: `4bit=50`, `8bit=2`
  - `level_offender_count=0`
- activation quantization:
  - activation quant modules: `52`
  - activation delta count: `0`

Grid eval:

- Run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E007_normalized_w4a32_baseline/eval/20260509_153415_normalized_w4a32_single_gpu1_grid478_seed20260507`
- Rows:
  - `11950 = 478 * 25`
- Grid:
  - SNR settings: `-2,-1,1,5,10`
  - missing rates: `0.02,0.08,0.18,0.28,0.38`
  - seed: `20260507`

Overall normalized 478x25 result:

| path | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| FP32 | 17.8329 | 18.1742 | 0.964330 | 0.978794 |
| W4A32 pre-recon | 16.6142 | 17.1775 | 0.935462 | 0.949638 |
| W4A32 post-recon | 17.7856 | 18.1128 | 0.964137 | 0.978461 |

Overall deltas:

| delta | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| pre - FP32 | -1.2186 | -0.8585 | -0.028868 | -0.023025 |
| post - FP32 | -0.0473 | -0.0370 | -0.000193 | 0.000015 |
| post - pre | 1.1713 | 0.8055 | 0.028675 | 0.022634 |

By-source W4A32 post-recon:

| source | rows | FP32 SNR mean | W4A32 SNR mean | delta mean | FP32 SSIM mean | W4A32 SSIM mean | delta SSIM mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anisotropic | 1875 | 22.0595 | 21.9718 | -0.0877 | 0.992762 | 0.992832 | 0.000070 |
| Kerry3D | 400 | 9.6085 | 9.6082 | -0.0003 | 0.950559 | 0.949887 | -0.000672 |
| Shots0001 | 9675 | 17.3538 | 17.3124 | -0.0414 | 0.959390 | 0.959165 | -0.000225 |

Conclusion for activation quantization:

- W4A32 remains close enough to FP32 on the matching normalized 478x25 grid to proceed to W4A8.
- The old raw-amplitude W4A8 absolute values should not be reused as normalized baseline numbers.
- The next activation quantization baseline should start from the new W4A32 checkpoint above and evaluate all W4A8 results on the same normalized 478x25 grid.

## 2026-05-09 E008 normalized W4A32 four-GPU baseline

Repeated the normalized W4A32 weight-only reconstruction with four-GPU distributed BRECQ. This was intended to change only the execution mode versus E007 single-GPU W4A32.

Execution notes:

- Initial sandboxed `torchrun --standalone` failed before model execution because local TCP rendezvous was blocked:
  - error class: `RendezvousConnectionError`
  - message included: `Operation not permitted`
- The same command was rerun outside the sandbox with identical experiment parameters.
- GPU `0` had a concurrent external `swinir` process using about `5.6 GiB`; the four-GPU W4A32 run completed successfully, but timing should be interpreted with that contention in mind.

Protocol:

- Dataset family:
  - `paper5_energy_filtered_perpatch_absmax`
- FP32 checkpoint:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- Calibration:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- Test:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`

W4A32 distributed reconstruction:

- Run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_160247_normalized_w4a32_1024cali_w20000_dist4_bsz16_global64`
- Checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_160247_normalized_w4a32_1024cali_w20000_dist4_bsz16_global64/checkpoints/quantized_scrn_brecq.pth`
- Pre-recon checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_160247_normalized_w4a32_1024cali_w20000_dist4_bsz16_global64/checkpoints/quantized_scrn_brecq_pre_recon.pth`
- Distributed config:
  - GPUs: `0,1,2,3`
  - `world_size=4`
  - local batch size: `16`
  - effective reconstruction global batch: `64`
- Settings:
  - `num_samples=1024`
  - `iters_w=20000`
  - `n_bits_w=4`
  - `n_bits_a=8`
  - `act_quant=false`
- Single-sample sanity:
  - `fp32_snr=13.8808`
  - `pre_w_snr=13.4675`
  - `post_w_snr=13.8607`
  - `post_recon_ssim=0.9276`
  - `reconstruction_seconds=4046.79`
  - `elapsed_seconds=4097.86`

Verification:

- JSON:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_160247_normalized_w4a32_1024cali_w20000_dist4_bsz16_global64/verification.json`
- `passed=true`
- final state:
  - `weight_quant=true`
  - `act_quant=false`
- weight quantization:
  - quant modules: `52`
  - weight bit counts: `4bit=50`, `8bit=2`
  - `level_offender_count=0`
- activation quantization:
  - activation quant modules: `52`
  - activation delta count: `0`

Grid eval:

- Run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E008_normalized_w4a32_dist4_baseline/eval/20260509_171210_normalized_w4a32_dist4_bsz16_global64_grid478_seed20260507`
- Rows:
  - `11950 = 478 * 25`
- Grid:
  - SNR settings: `-2,-1,1,5,10`
  - missing rates: `0.02,0.08,0.18,0.28,0.38`
  - seed: `20260507`

Overall normalized 478x25 result:

| path | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| FP32 | 17.8329 | 18.1742 | 0.964330 | 0.978794 |
| W4A32 pre-recon | 16.6142 | 17.1775 | 0.935462 | 0.949638 |
| W4A32 post-recon dist4 | 17.7120 | 18.0740 | 0.964234 | 0.978921 |

Overall deltas:

| delta | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| pre - FP32 | -1.2186 | -0.8585 | -0.028868 | -0.023025 |
| post - FP32 | -0.1209 | -0.0752 | -0.000096 | -0.000088 |
| post - pre | 1.0977 | 0.7782 | 0.028772 | 0.022477 |

By-source W4A32 post-recon:

| source | rows | FP32 SNR mean | W4A32 dist4 SNR mean | delta mean | FP32 SSIM mean | W4A32 dist4 SSIM mean | delta SSIM mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anisotropic | 1875 | 22.0595 | 21.8008 | -0.2586 | 0.992762 | 0.992702 | -0.000060 |
| Kerry3D | 400 | 9.6085 | 9.5971 | -0.0113 | 0.950559 | 0.950481 | -0.000078 |
| Shots0001 | 9675 | 17.3538 | 17.2551 | -0.0987 | 0.959390 | 0.959286 | -0.000104 |

Single-GPU comparison:

| checkpoint | SNR mean | SNR median | SSIM mean | SSIM median | post - FP32 SNR mean |
|---|---:|---:|---:|---:|---:|
| E007 single GPU | 17.7856 | 18.1128 | 0.964137 | 0.978461 | -0.0473 |
| E008 dist4 global64 | 17.7120 | 18.0740 | 0.964234 | 0.978921 | -0.1209 |

Conclusion for activation quantization:

- E008 dist4 is a valid W-only checkpoint and remains close to FP32, but it is slightly worse in SNR than E007 single GPU.
- E008 was slower than E007 in this run, likely due to GPU `0` contention and distributed overhead, so it should not replace E007 as the preferred W4A8 starting checkpoint.
- Keep E007 single-GPU W4A32 as the default W4A8 activation-init base unless a clean uncontended dist4 rerun shows both comparable quality and a real speed benefit.

## 2026-05-09 E009 normalized W4A32 dist4 global128 probe

Ran a second normalized W4A32 distributed reconstruction with local batch `32`, effective global batch `128`. This was a controlled probe against E008: the intended experiment change was only `batch_size=16 -> 32`, with the same FP32 checkpoint, calibration set, W4A32 settings, and normalized 478x25 eval.

Execution notes:

- The sandboxed `torchrun --standalone` attempt failed before model work with `RendezvousConnectionError` / `Operation not permitted`, so the same command was rerun outside the sandbox with unchanged parameters.
- GPU `0` had an external `swinir` process using about `5.6 GiB` during the first part of reconstruction; it ended before completion.
- The run completed without OOM.

Protocol:

- Dataset family:
  - `paper5_energy_filtered_perpatch_absmax`
- FP32 checkpoint:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- Calibration:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- Test:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`

W4A32 distributed reconstruction:

- Run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_182611_normalized_w4a32_1024cali_w20000_dist4_bsz32_global128`
- Checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_182611_normalized_w4a32_1024cali_w20000_dist4_bsz32_global128/checkpoints/quantized_scrn_brecq.pth`
- Pre-recon checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_182611_normalized_w4a32_1024cali_w20000_dist4_bsz32_global128/checkpoints/quantized_scrn_brecq_pre_recon.pth`
- Distributed config:
  - GPUs: `0,1,2,3`
  - `world_size=4`
  - local batch size: `32`
  - effective reconstruction global batch: `128`
- Settings:
  - `num_samples=1024`
  - `iters_w=20000`
  - `n_bits_w=4`
  - `n_bits_a=8`
  - `act_quant=false`
- Single-sample sanity:
  - `fp32_snr=13.8808`
  - `pre_w_snr=13.4675`
  - `post_w_snr=13.8346`
  - `post_recon_ssim=0.9280`
  - `reconstruction_seconds=6920.35`
  - `elapsed_seconds=7076.65`

Verification:

- JSON:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_182611_normalized_w4a32_1024cali_w20000_dist4_bsz32_global128/verification.json`
- `passed=true`
- final state:
  - `weight_quant=true`
  - `act_quant=false`
- weight quantization:
  - quant modules: `52`
  - weight bit counts: `4bit=50`, `8bit=2`
  - `level_offender_count=0`
- activation quantization:
  - activation quant modules: `52`
  - activation delta count: `0`

Grid eval:

- Run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E009_normalized_w4a32_dist4_global128_probe/eval/20260509_202444_normalized_w4a32_dist4_bsz32_global128_grid478_seed20260507`
- Rows:
  - `11950 = 478 * 25`
- Grid:
  - SNR settings: `-2,-1,1,5,10`
  - missing rates: `0.02,0.08,0.18,0.28,0.38`
  - seed: `20260507`
- Eval runtime:
  - elapsed seconds: `198.69`
  - FP32 inference seconds: `26.76`
  - pre-recon quant inference seconds: `82.52`
  - post-recon quant inference seconds: `27.77`

Overall normalized 478x25 result:

| path | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| FP32 | 17.8329 | 18.1742 | 0.964330 | 0.978794 |
| W4A32 pre-recon | 16.6142 | 17.1775 | 0.935462 | 0.949638 |
| W4A32 post-recon dist4 global128 | 17.7370 | 18.0836 | 0.963943 | 0.978616 |

Overall deltas:

| delta | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| pre - FP32 | -1.2186 | -0.8585 | -0.028868 | -0.024085 |
| post - FP32 | -0.0959 | -0.0589 | -0.000387 | -0.000446 |
| post - pre | 1.1228 | 0.7935 | 0.028481 | 0.023789 |

By-source W4A32 post-recon:

| source | rows | FP32 SNR mean | W4A32 global128 SNR mean | delta mean | FP32 SSIM mean | W4A32 global128 SSIM mean | delta SSIM mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anisotropic | 1875 | 22.0595 | 21.9149 | -0.1446 | 0.992762 | 0.992198 | -0.000564 |
| Kerry3D | 400 | 9.6085 | 9.6534 | 0.0449 | 0.950559 | 0.950640 | 0.000080 |
| Shots0001 | 9675 | 17.3538 | 17.2616 | -0.0922 | 0.959390 | 0.959018 | -0.000372 |

Baseline comparison:

| checkpoint | SNR mean | SNR median | SSIM mean | SSIM median | post - FP32 SNR mean | reconstruction seconds |
|---|---:|---:|---:|---:|---:|---:|
| E007 single GPU | 17.7856 | 18.1128 | 0.964137 | 0.978461 | -0.0473 | 2597.64 |
| E008 dist4 global64 | 17.7120 | 18.0740 | 0.964234 | 0.978921 | -0.1209 | 4046.79 |
| E009 dist4 global128 | 17.7370 | 18.0836 | 0.963943 | 0.978616 | -0.0959 | 6920.35 |

Conclusion for activation quantization:

- E009 confirms that increasing dist4 effective batch from `64` to `128` partially recovers SNR: `+0.0250 dB` mean and `+0.0096 dB` median over E008.
- E009 still trails E007 single GPU by `0.0486 dB` mean SNR and `0.0292 dB` median SNR, and reconstruction is much slower.
- E007 single-GPU W4A32 remains the default W4A8 activation-init base. E009 should be kept as evidence that distributed batch size matters, not as the preferred checkpoint.

## 2026-05-09 E010 normalized W4A32 representative visuals

Generated representative comparison figures for E007, the current default W4A32 baseline for future W4A8 activation-init comparisons. This run did not change any checkpoint or metric baseline; it only regenerated selected visual examples from the existing E007 normalized 478x25 grid records.

Run:

- Visualization run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E010_normalized_w4a32_single_gpu_visuals/20260509_204855_representative_figures_source_x_condition`
- Checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- Pre-recon checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq_pre_recon.pth`
- Source metric rows:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E007_normalized_w4a32_baseline/eval/20260509_153415_normalized_w4a32_single_gpu1_grid478_seed20260507/per_sample_metrics.jsonl`

Selection:

- Sources: `Anisotropic`, `Kerry3D`, `Shots0001`
- Conditions:
  - low: SNR `-2`, missing rate `0.38`
  - mid: SNR `1`, missing rate `0.18`
  - high: SNR `10`, missing rate `0.02`
- Per source/condition, selected the sample closest to the median `quant_post_minus_fp32_snr_db` for that group.
- Figure panel order:
  - Ground Truth
  - Input
  - FP32
  - W4A32 pre-recon
  - W4A32 post-recon

Generated figures:

| # | source | condition | patch | input SNR | FP32 SNR | pre SNR | post SNR | post-FP32 | figure |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Anisotropic | low_snr_high_missing | `test_000044.npy` | -2.95 | 18.29 | 17.38 | 18.19 | -0.100 | `01_Anisotropic_low_snr_high_missing_test_000044.png` |
| 2 | Anisotropic | mid_snr_mid_missing | `test_000013.npy` | 0.06 | 23.32 | 21.77 | 23.21 | -0.114 | `02_Anisotropic_mid_snr_mid_missing_test_000013.png` |
| 3 | Anisotropic | high_snr_low_missing | `test_000025.npy` | 9.49 | 28.75 | 24.63 | 28.65 | -0.103 | `03_Anisotropic_high_snr_low_missing_test_000025.png` |
| 4 | Kerry3D | low_snr_high_missing | `test_000081.npy` | -2.96 | 6.05 | 5.92 | 6.04 | -0.008 | `04_Kerry3D_low_snr_high_missing_test_000081.png` |
| 5 | Kerry3D | mid_snr_mid_missing | `test_000084.npy` | 0.08 | 11.49 | 10.83 | 11.47 | -0.020 | `05_Kerry3D_mid_snr_mid_missing_test_000084.png` |
| 6 | Kerry3D | high_snr_low_missing | `test_000077.npy` | 10.00 | 10.63 | 11.27 | 10.64 | 0.008 | `06_Kerry3D_high_snr_low_missing_test_000077.png` |
| 7 | Shots0001 | low_snr_high_missing | `test_000374.npy` | -3.01 | 12.81 | 12.11 | 12.78 | -0.037 | `07_Shots0001_low_snr_high_missing_test_000374.png` |
| 8 | Shots0001 | mid_snr_mid_missing | `test_000395.npy` | 0.22 | 17.08 | 16.50 | 17.04 | -0.033 | `08_Shots0001_mid_snr_mid_missing_test_000395.png` |
| 9 | Shots0001 | high_snr_low_missing | `test_000349.npy` | 9.84 | 20.85 | 18.69 | 20.80 | -0.046 | `09_Shots0001_high_snr_low_missing_test_000349.png` |

Verification:

- Generated `9` PNG files under the run `figures/` directory.
- Each figure is `3600 x 720`, containing the five comparison panels.
- E007 remains the default W4A32 baseline for subsequent W4A8 activation quantization experiments.

## 2026-05-09 E011 normalized W4A32 seismic visuals

Generated seismic-colormap comparison figures for E007. This set keeps the same representative normalized examples as E010 and adds the usual default single sample used by `quantize_scrn.py`.

Run:

- Visualization run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E011_normalized_w4a32_seismic_visuals/20260509_205659_seismic_representative_plus_default_single`
- Checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- Pre-recon checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq_pre_recon.pth`

Display settings:

- Colormap: `seismic`
- Color scale: symmetric per figure over all panels, centered at `0`
- Panels: Ground Truth, Input, FP32, W4A32 pre-recon, W4A32 post-recon

Generated figures:

| # | source | condition | amplitude | patch | input SNR | FP32 SNR | pre SNR | post SNR | post-FP32 | figure |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Anisotropic | low_snr_high_missing | normalized per-patch absmax | `test_000044.npy` | -2.95 | 18.29 | 17.38 | 18.19 | -0.100 | `01_Anisotropic_low_snr_high_missing_test_000044_seismic.png` |
| 2 | Anisotropic | mid_snr_mid_missing | normalized per-patch absmax | `test_000013.npy` | 0.06 | 23.32 | 21.77 | 23.21 | -0.114 | `02_Anisotropic_mid_snr_mid_missing_test_000013_seismic.png` |
| 3 | Anisotropic | high_snr_low_missing | normalized per-patch absmax | `test_000025.npy` | 9.49 | 28.75 | 24.63 | 28.65 | -0.103 | `03_Anisotropic_high_snr_low_missing_test_000025_seismic.png` |
| 4 | Kerry3D | low_snr_high_missing | normalized per-patch absmax | `test_000081.npy` | -2.96 | 6.05 | 5.92 | 6.04 | -0.008 | `04_Kerry3D_low_snr_high_missing_test_000081_seismic.png` |
| 5 | Kerry3D | mid_snr_mid_missing | normalized per-patch absmax | `test_000084.npy` | 0.08 | 11.49 | 10.83 | 11.47 | -0.020 | `05_Kerry3D_mid_snr_mid_missing_test_000084_seismic.png` |
| 6 | Kerry3D | high_snr_low_missing | normalized per-patch absmax | `test_000077.npy` | 10.00 | 10.63 | 11.27 | 10.64 | 0.008 | `06_Kerry3D_high_snr_low_missing_test_000077_seismic.png` |
| 7 | Shots0001 | low_snr_high_missing | normalized per-patch absmax | `test_000374.npy` | -3.01 | 12.81 | 12.11 | 12.78 | -0.037 | `07_Shots0001_low_snr_high_missing_test_000374_seismic.png` |
| 8 | Shots0001 | mid_snr_mid_missing | normalized per-patch absmax | `test_000395.npy` | 0.22 | 17.08 | 16.50 | 17.04 | -0.033 | `08_Shots0001_mid_snr_mid_missing_test_000395_seismic.png` |
| 9 | Shots0001 | high_snr_low_missing | normalized per-patch absmax | `test_000349.npy` | 9.84 | 20.85 | 18.69 | 20.80 | -0.046 | `09_Shots0001_high_snr_low_missing_test_000349_seismic.png` |
| 10 | `SCRN-main/test_data` | default_single_sample | raw default SCRN sample | `clear.npy` | 3.97 | 13.88 | 13.47 | 13.89 | 0.011 | `10_default_single_sample_seismic_raw_amplitude.png` |

Verification:

- Generated `10` PNG files under the run `figures/` directory.
- Each figure is `3600 x 720`.
- This does not change the W4A32 baseline choice: E007 single-GPU remains the default W4A8 activation-init base.

## 2026-05-09 E012 normalized W4A32 seismic visuals with denormalized display

Updated the E011 seismic visualization helper and generated a corrected E012 visual set that restores normalized representative patches before display.

Script:

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E011_normalized_w4a32_seismic_visuals/generate_seismic_visuals.py`

Restoration:

- Formula: `display = normalized * normalization_scale`
- Scale source:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478/manifest.json`
  - `samples[].normalization_scale`
- Metrics:
  - SNR/SSIM labels remain the normalized-protocol metrics.
  - The restoration changes only visualization amplitude, not model input, checkpoint, or evaluation baseline.

Run:

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E012_normalized_w4a32_seismic_denormalized_visuals/20260509_211000_seismic_denormalized_representative_plus_default_single`

Generated figures:

| # | source | condition | display amplitude | scale | patch | input SNR | FP32 SNR | pre SNR | post SNR | post-FP32 | figure |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Anisotropic | low_snr_high_missing | restored from normalized per-patch absmax | 0.260053 | `test_000044.npy` | -2.95 | 18.29 | 17.38 | 18.19 | -0.100 | `01_Anisotropic_low_snr_high_missing_test_000044_seismic_denormalized.png` |
| 2 | Anisotropic | mid_snr_mid_missing | restored from normalized per-patch absmax | 0.0957499 | `test_000013.npy` | 0.06 | 23.32 | 21.77 | 23.21 | -0.114 | `02_Anisotropic_mid_snr_mid_missing_test_000013_seismic_denormalized.png` |
| 3 | Anisotropic | high_snr_low_missing | restored from normalized per-patch absmax | 0.133937 | `test_000025.npy` | 9.49 | 28.75 | 24.63 | 28.65 | -0.103 | `03_Anisotropic_high_snr_low_missing_test_000025_seismic_denormalized.png` |
| 4 | Kerry3D | low_snr_high_missing | restored from normalized per-patch absmax | 0.623027 | `test_000081.npy` | -2.96 | 6.05 | 5.92 | 6.04 | -0.008 | `04_Kerry3D_low_snr_high_missing_test_000081_seismic_denormalized.png` |
| 5 | Kerry3D | mid_snr_mid_missing | restored from normalized per-patch absmax | 0.817539 | `test_000084.npy` | 0.08 | 11.49 | 10.83 | 11.47 | -0.020 | `05_Kerry3D_mid_snr_mid_missing_test_000084_seismic_denormalized.png` |
| 6 | Kerry3D | high_snr_low_missing | restored from normalized per-patch absmax | 0.571108 | `test_000077.npy` | 10.00 | 10.63 | 11.27 | 10.64 | 0.008 | `06_Kerry3D_high_snr_low_missing_test_000077_seismic_denormalized.png` |
| 7 | Shots0001 | low_snr_high_missing | restored from normalized per-patch absmax | 0.056497 | `test_000374.npy` | -3.01 | 12.81 | 12.11 | 12.78 | -0.037 | `07_Shots0001_low_snr_high_missing_test_000374_seismic_denormalized.png` |
| 8 | Shots0001 | mid_snr_mid_missing | restored from normalized per-patch absmax | 0.0439193 | `test_000395.npy` | 0.22 | 17.08 | 16.50 | 17.04 | -0.033 | `08_Shots0001_mid_snr_mid_missing_test_000395_seismic_denormalized.png` |
| 9 | Shots0001 | high_snr_low_missing | restored from normalized per-patch absmax | 0.0698473 | `test_000349.npy` | 9.84 | 20.85 | 18.69 | 20.80 | -0.046 | `09_Shots0001_high_snr_low_missing_test_000349_seismic_denormalized.png` |
| 10 | `SCRN-main/test_data` | default_single_sample | raw default SCRN sample | n/a | `clear.npy` | 3.97 | 13.88 | 13.47 | 13.89 | 0.011 | `10_default_single_sample_seismic_raw_amplitude.png` |

Verification:

- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E011_normalized_w4a32_seismic_visuals/generate_seismic_visuals.py` passed.
- Running the helper completed with `figure_count=10`.
- `find .../figures -name '*.png' | wc -l` returned `10`.
- `file .../figures/*.png` reported all figures as PNG images with size `3600 x 720`.
- E007 single-GPU W4A32 remains the default W4A8 activation-init base; this change is visualization-only.

## 2026-05-09 NE000-NE006 归一化协议激活量化路线图

开启新的激活量化实验序列，统一命名为 `NE00X`；其中 `NE` 表示新的归一化数据协议。旧 E001-E006 结果仍作为机制证据保留，但在当前 `paper5_energy_filtered_perpatch_absmax` 协议下，NE000-NE006 才是后续判断的正式实验。

共享协议：

- 数据协议：
  - `paper5_energy_filtered_perpatch_absmax`
- FP32 checkpoint：
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- 校准集：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- 测试集：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- W4A8 起点：
  - E007 单卡 W4A32 checkpoint：
    - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- 主评估口径：
  - normalized `478 x 25` grid
  - SNR settings：`-2,-1,1,5,10`
  - missing rates：`0.02,0.08,0.18,0.28,0.38`
  - seed：`20260507`

计划实验：

### NE000 W4A8 激活量化重建 baseline

目的：

- 在正式诊断前，重建新协议下的 W4A8 baseline。
- 从 E007 W4A32 checkpoint 出发，不再使用任何旧 raw-amplitude checkpoint。

默认动作：

- 运行 tensor-wise A8 activation initialization。
- 运行 activation reconstruction，设置 `iters_a=5000`、`activation_lr=0.0004`、`lp_norm=2.4`。
- 同时保存：
  - `quantized_scrn_brecq_pre_act_recon.pth`
  - `quantized_scrn_brecq.pth`
- 在 normalized `478 x 25` grid 上同时评估 pre-act-recon 和 final checkpoint。

必要输出：

- checkpoint 验证：
  - `weight_quant=true`
  - `act_quant=true`
  - weight bits 保持 `4bit=50`、`8bit=2`
  - activation quantizer 数量预期约为 `52`
- full-grid 指标：
  - FP32
  - E007 W4A32 reference
  - W4A8 pre-act-recon
  - W4A8 post-act-recon
  - by-source metrics

决策点：

- 建立新的 W4A8 baseline，并量化 W4A8 相对 FP32 和 E007 W4A32 的 gap。

### NE001 激活诊断

目的：

- 在新数据和新模型协议下复现旧 E001 的诊断角色。

默认动作：

- 对 NE000 pre-act-recon 和 final checkpoint 分别运行 diagnostics。
- 使用 normalized calibration set。
- 比较 activation reconstruction 前后的 activation quantizer 合法性和 fake-quant 行为。

必要输出：

- quantizer 数量。
- `non_positive_delta_count`。
- activation delta min/max/shape 统计。
- fake-quant MSE / range 摘要。
- Conv2d vs Linear 分组摘要。
- 可用时记录 stage/role 摘要。

决策点：

- 确认 scale 合法性是否干净，并定位 W4A8 误差集中位置。

### NE002 合法状态和 checkpoint sanity sweep

目的：

- 将 checkpoint / quantizer 非法状态与真实 activation quantization 精度损失分离。

默认动作：

- 对 NE000 pre 和 final checkpoint 做 reload 后验证。
- 确认 final quant state、output quantizer 处理、activation delta 正值约束和 checkpoint shape restore。
- 检查直接运行指标与 reload/evaluator 指标是否在预期容差内一致。

必要输出：

- pre 和 final W4A8 checkpoint 的 verification JSON。
- reload consistency 记录。
- output quantizer state 记录。
- legal-state offender list，如果存在。

决策点：

- 如果 legality/reload 失败，先修复再进入敏感性实验。
- 如果状态干净，则把 W4A8 掉点视为真实量化问题。

### NE003 固定评估口径和单样本敏感性检查

目的：

- 重新确认 full normalized grid metrics 是正式判断依据。

默认动作：

- 使用同一组 NE000 checkpoint，对比默认单样本、代表样本、小 subset 和完整 `478 x 25` grid。
- 只有在解释需要时才重新生成代表图。

必要输出：

- 单样本、subset 和 full-grid SNR/SSIM 对比表。
- 记录任何与 full-grid 行为明显不一致的 sample/source。

决策点：

- 后续所有 NE 结论都以 full-grid mean/median 和 by-source metrics 为准。

### NE004 activation quantizer 敏感性

目的：

- 重新测试在 normalized 数据下，W4A8 gap 是否仍主要由 Conv2d activation quantization 导致。

默认动作：

- 在不改变权重的前提下，评估关闭不同 activation quantizer 分组的效果：
  - all Conv2d
  - all Linear/transformer
  - stage groups
  - 支持时测试 split/merge/stage-output 等 role groups
- 使用 NE000 W4A8 作为 baseline checkpoint。

必要输出：

- sensitivity table：
  - full-grid SNR/SSIM
  - 相对 all-on W4A8 的恢复量
  - 距离 E007 W4A32 的剩余 gap
  - by-source recovery

决策点：

- 确认旧 E004 结论是否仍成立：Conv2d activation quantization 是否仍是主瓶颈。

### NE005 range、clipping 和 outlier 控制

目的：

- 测试 normalized 数据是否改变旧结论：tensor-wise range/clipping 不足以解决主问题。

默认动作：

- 只有在 NE004 确定目标分组后，再运行 range variants：
  - tensor-wise max
  - percentile clipping
  - tensor-wise MSE grid
  - 如有必要，测试 selector-based structured range variants
- 评估口径保持 normalized `478 x 25` grid。

必要输出：

- range-method 对比表。
- fake-quant range diagnostics。
- by-source metrics。
- 明确说明 range tuning 是否实质改善 W4A8。

决策点：

- 如果 range 能修复 gap，优先推进 range/calibration policy。
- 如果不能，转入结构化 granularity。

### NE006 结构化 activation granularity 搜索

目的：

- 在新 normalized 协议下复测旧 E006 的强候选策略。

默认动作：

- 评估：
  - all Conv2d per-channel
  - split_proj + merge_proj + stage_output_conv per-channel
  - split_proj + merge_proj + stage_output_conv group-wise `g4`
  - stage_output_conv group-wise `g4`
  - stage5-focused sanity checks
- 使用 E007 W4A32 作为起点，并与 NE000 tensor-wise W4A8 对比。

必要输出：

- strategy table，包含：
  - 适用时记录 pre-act 和 post-act 指标
  - full-grid mean/median SNR/SSIM
  - by-source metrics
  - checkpoint paths
  - per-channel vs group-wise 候选的部署备注

决策点：

- 选择新的 W4A8 候选策略，用于后续更深 activation reconstruction、mixed precision 或 selective FP32 实验。

执行规则：

- 模型推理和重建优先使用 GPU。单卡默认优先级为 `1 -> 2 -> 3 -> 0`。
- 记录 GPU 选择和任何偏离默认策略的原因。
- 所有生成产物必须放在 `SCRN_BRECQ_app/` 下。
- 不修改 `SCRN-main/` 或 `BRECQ-main/`。
- 每个 NE 实验都必须同步更新两份日志。
- 每次完成实验或代码/日志变更后都要 commit。
- 不 push，除非明确要求。

## 2026-05-09 NE000 归一化 W4A8 激活量化重建 baseline

目标：

- 在新协议 `paper5_energy_filtered_perpatch_absmax` 下，正式重建 W4A8 activation quantization baseline。
- 从当前默认 E007 W4A32 checkpoint 出发，运行 tensor-wise A8 activation initialization 和 `iters_a=5000` activation reconstruction。
- 在 normalized `478 x 25` grid 上评估 pre-act-recon 和 final W4A8。

预检和 GPU：

- Branch：`main`
- Worktree：开始时 clean。
- GPU 状态：GPU 1/2/3 空闲；GPU 0 有外部 `swinir` 进程占用约 `5.6 GiB`。
- 本次按默认单卡优先级使用物理 GPU `1`。

输入：

- W4A32 起点：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- 校准集：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- 测试集：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`

运行中发现并修复的问题：

- 首次带 activation reconstruction 的运行失败于：
  - `RuntimeError: Activation quantizer delta is not learnable. Construct QuantModel with act_quant_params['leaf_param']=True.`
- 诊断结论：
  - E007 W4A32 checkpoint 的 `quant_config.act_quant=false` 是正确的 W-only 状态。
  - activation-only CLI 若直接用该 checkpoint 配置构造 `QuantModel`，activation delta 不是 learnable parameter，导致 activation reconstruction 无法优化 delta。
- 修复：
  - 在 `activation_only_quantize_scrn.py` 中新增 `build_activation_only_checkpoint_config()`。
  - 该 helper 只构造运行时 checkpoint view，将 `act_quant`、`n_bits_a`、`scale_method` 对齐到 activation-only 配置。
  - 源 E007 checkpoint 不被改写。
- 单测：
  - 增加测试确认 activation-only checkpoint view 会启用 activation quantization，并保留源 checkpoint 的 `act_quant=false`。
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn -v` 通过，18 tests OK。
  - `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py` 通过。

NE000 quant run：

- Run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction/quant/20260509_213701_normalized_w4a8_tensor_a5000_1024cali_single_gpu1`
- Pre-act-recon checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction/quant/20260509_213701_normalized_w4a8_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- Final checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction/quant/20260509_213701_normalized_w4a8_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- 配置：
  - `n_bits_w=4`
  - `n_bits_a=8`
  - `activation_granularity=tensor`
  - `activation_range_method=none`
  - `num_samples=1024`
  - `batch_size=16`
  - `init_batch_size=64`
  - `iters_a=5000`
  - `activation_lr=0.0004`
  - `lp_norm=2.4`
- 时间：
  - activation initialization：`24.8221 s`
  - activation reconstruction：`1335.2904 s`，约 `22.2548 min`
  - elapsed：`1363.2839 s`，约 `22.7214 min`

单样本 sanity：

| metric | value |
|---|---:|
| W4A32 post-weight SNR / SSIM | `13.8918 / 0.929029` |
| W4A8 pre-act SNR / SSIM | `13.8772 / 0.927889` |
| W4A8 final SNR / SSIM | `13.8640 / 0.928557` |
| activation init SNR delta | `-0.0146 dB` |
| activation reconstruction SNR gain | `-0.0131 dB` |

Activation quantizer 状态：

- `quant_modules=52`
- `activation_quantizers=52`
- `activation_delta_count=52`
- `activation_zero_point_count=52`
- `initialized_activation_quantizers=52`
- `learnable_activation_delta_count=52`
- `disabled_activation_quantizers=1`
- `delta_min=0.0040127067`
- `delta_max=0.0526451916`
- `zero_point_min=-0.0`
- `zero_point_max=158.0`
- `non_positive_delta_count=0`
- `non_positive_delta_elements=0`
- `offender_layers=[]`

Checkpoint verification：

| checkpoint | passed | final quant state | weight bits | level offenders | activation delta count | learnable activation deltas |
|---|---|---|---|---:|---:|---:|
| pre-act-recon | `true` | `weight_quant=true, act_quant=true` | `4bit=50, 8bit=2` | `0` | `52` | `52` |
| final | `true` | `weight_quant=true, act_quant=true` | `4bit=50, 8bit=2` | `0` | `52` | `52` |

Normalized `478 x 25` grid eval：

- Eval run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction/eval/20260509_221644_normalized_w4a8_tensor_a5000_grid478_seed20260507`
- 产物：
  - `config.json`
  - `metrics.json`
  - `summary.md`
  - `per_sample_metrics.jsonl`
- Row count：
  - `11950`

Overall：

| model | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| FP32 | `17.8329` | `18.1742` | `0.964330` | `0.978794` |
| E007 W4A32 final | `17.7856` | `18.1128` | `0.964137` | `0.978461` |
| NE000 W4A8 pre-act | `17.3727` | `17.7855` | `0.962674` | `0.977030` |
| NE000 W4A8 final | `17.4495` | `17.8777` | `0.962868` | `0.977292` |

Gap：

- W4A8 final 相对 FP32：
  - mean SNR：`-0.3834 dB`
  - mean SSIM：`-0.001462`
- W4A8 final 相对 E007 W4A32 final：
  - mean SNR：`-0.3361 dB`
  - mean SSIM：`-0.001269`
- Activation reconstruction 相对 pre-act：
  - mean SNR：`+0.0768 dB`
  - mean SSIM：`+0.000195`

By-source：

| source | rows | FP32 SNR mean | W4A8 pre SNR mean | W4A8 final SNR mean | final-FP32 SNR mean | W4A8 final SSIM mean |
|---|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `22.0595` | `21.2895` | `21.2873` | `-0.7722` | `0.991631` |
| Kerry3D | `400` | `9.6085` | `9.5817` | `9.5704` | `-0.0381` | `0.948695` |
| Shots0001 | `9675` | `17.3538` | `16.9358` | `17.0315` | `-0.3223` | `0.957880` |

结论：

- normalized 协议下，tensor-wise W4A8 不再复现旧 raw 协议里的严重崩坏，但仍明显低于 E007 W4A32。
- W4A8 final 相比 FP32 的 full-grid mean SNR gap 为 `-0.3834 dB`，相比 E007 W4A32 的 gap 为 `-0.3361 dB`。
- Activation reconstruction 在 full grid 上带来小幅正收益，但不足以消除 W4A8 gap。
- by-source 上 Anisotropic 掉点最大，Shots0001 次之，Kerry3D 基本接近 FP32。
- 下一步进入 NE001 diagnostics，重点检查 activation delta 合法性、Conv2d/Linear fake-quant error 和 stage/role/source 分布。

## 2026-05-09 NE000_1 packed deployment equivalence 计划

在进入 NE001 前，追加 `NE000_1`。该实验只验证部署 artifact 等价性，不做 activation diagnostics；activation int8 level 分布、fake-quant error、Conv2d/Linear/stage/role 诊断仍归入 NE001。

动机：

- NE000 的 `.pth` 是恢复型 PyTorch checkpoint，保存的是 FP32 权重、AdaRound alpha、delta/zero_point 和 activation quantizer 状态。
- 虽然 NE000 的推理路径已经确认会执行 W4A8 fake quant，但 `.pth` 本身不是具体 packed 整数部署文件。
- 因此需要验证：把权重转成具体 packed 整数值，并从部署 artifact 恢复 activation qparams 后，是否仍能复现 checkpoint 的 full-grid 指标。

实验拆分：

| 子实验 | 输入 checkpoint | 量化状态 | 验证目的 |
|---|---|---|---|
| NE000_1a | `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth` | W4A32，`weight_quant=true, act_quant=false` | 验证默认 E007 单卡 W4A32 weight-recon checkpoint 的 packed 权重导出和恢复是否对齐。 |
| NE000_1b | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction/quant/20260509_213701_normalized_w4a8_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq.pth` | W4A8，`weight_quant=true, act_quant=true` | 验证 NE000 W4A8 final checkpoint 在 packed 权重整数化并恢复 activation qparams 后是否仍对齐。 |

关于 W4A32 对照的解释：

- NE000 W4A8 不会产出一个新的独立 W4A32 weight-recon checkpoint。
- NE000 的 W4 权重继承自 E007 单卡 W4A32；在 NE000 checkpoint 上关闭 activation quantization 得到的 W4A32 行为只是同一权重的 sanity/reference。
- 所以 NE000_1 的 W4A32 部署验证对象应是 E007 单卡最佳 checkpoint，而不是从 NE000 额外派生一个新 W4A32 checkpoint。

计划动作：

1. 对 E007 W4A32 final checkpoint 运行 packed export。
2. 对 NE000 W4A8 final checkpoint 运行 packed export。
3. 分别恢复两个 packed artifact：
   - `weights.bin` 存储 packed uint4/uint8 权重整数；
   - `aux_fp32.bin` 存储 weight delta/zero_point、activation delta/zero_point、bias 和必要 FP32 参数；
   - 恢复后关闭 weight fake quant，因为权重已经是部署量化值；
   - W4A8 恢复 activation qparams，并保持 activation fake quant。
4. 使用 normalized `478 x 25` grid 做部署等价评估：
   - E007 W4A32 checkpoint final vs W4A32 packed-restored；
   - NE000 W4A8 checkpoint final vs W4A8 packed-restored。
5. 记录部署 artifact 体积和理论压缩收益。

输出目录建议：

- W4A32 packed export：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a32_packed`
- W4A8 packed export：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a8_packed`
- Grid eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/eval`

必须记录：

- export summary：
  - `weights.bin` size；
  - `aux_fp32.bin` size；
  - `manifest.json` size；
  - `summary.json` size；
  - raw deployment payload size；
  - total export size；
  - estimated compression ratio。
- restore summary：
  - restored quantized layers；
  - restored non-quantized tensors；
  - restored activation quantizers；
  - final quant state。
- full-grid 对齐：
  - row count：`11950`；
  - checkpoint SNR/SSIM mean/median；
  - packed-restored SNR/SSIM mean/median；
  - packed minus checkpoint 的 SNR/SSIM delta；
  - prediction diff 的 MSE / mean abs / max abs。

验收标准：

- W4A32 packed-restored 必须与 E007 W4A32 checkpoint final 高度对齐。
- W4A8 packed-restored 必须与 NE000 W4A8 checkpoint final 高度对齐。
- 若 W4A32 对齐但 W4A8 不对齐，优先定位 activation qparams restore 或 W4A8 packed eval 状态。
- 若两者都不对齐，优先定位 packed weight integer export/restore。
- 若两者都对齐，则可认为 NE000 的结果不仅存在于恢复型 `.pth` fake-quant checkpoint 中，可以继续进入 NE001 diagnostics。

## 2026-05-09 NE000_1 packed grid evaluator

为 NE000_1 实现 packed deployment full-grid 等价评估入口。本次只新增评估工具，不运行正式 packed export/eval 结论。

新增文件：

- `SCRN_BRECQ_app/scrn_brecq/cli/evaluate_packed_scrn_grid.py`
- `SCRN_BRECQ_app/scrn_brecq/tests/test_evaluate_packed_scrn_grid.py`

评估语义：

- reference checkpoint：
  - 使用 checkpoint 的 final quant state。
  - W4A32 为 `weight_quant=true, act_quant=false`。
  - W4A8 为 `weight_quant=true, act_quant=true`。
- packed-restored artifact：
  - 从 `weights.bin` 解包权重整数并反量化写回模型；
  - 从 `aux_fp32.bin` 恢复 weight / activation qparams 和必要 FP32 参数；
  - 推理时使用 `weight_quant=false`；
  - `act_quant` 从 packed manifest final state / quant config 继承。

默认协议：

- Test set：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- Grid：
  - SNR settings：`-2,-1,1,5,10`
  - Missing rates：`0.02,0.08,0.18,0.28,0.38`
  - Seed：`20260507`

输出和指标：

- `per_sample_metrics.jsonl`
- `metrics.json`
- `config.json`
- `summary.md`
- 每行记录：
  - `fp32_snr_db / fp32_ssim`
  - `checkpoint_snr_db / checkpoint_ssim`
  - `packed_snr_db / packed_ssim`
  - `packed_minus_checkpoint_*`
  - `packed_vs_checkpoint_mse`
  - `packed_vs_checkpoint_mean_abs_diff`
  - `packed_vs_checkpoint_max_abs_diff`
- 聚合维度：
  - overall
  - by source
  - by SNR setting
  - by missing rate
  - by condition

测试记录：

- TDD RED：
  - 新增 test 后先运行：
    - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_packed_scrn_grid -v`
  - 预期失败：
    - `ModuleNotFoundError: No module named 'SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn_grid'`
- GREEN：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_packed_scrn_grid -v`
  - 5 tests OK
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_packed_scrn_grid SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_packed_scrn SCRN_BRECQ_app.scrn_brecq.tests.test_packed_deployment -v`
  - 13 tests OK
  - `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/evaluate_packed_scrn_grid.py`
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn_grid --help`

下一步：

- 提交工具后，继续执行 NE000_1a / NE000_1b packed export。
- 然后使用该 evaluator 跑 W4A32 和 W4A8 的 normalized `478 x 25` packed equivalence。

## 2026-05-09 NE000_1 packed deployment equivalence 结果

完成 NE000_1a / NE000_1b packed deployment equivalence。该实验只验证部署 artifact 能否复现 checkpoint 指标，不做 activation diagnostics。

预检：

- Branch：`main`
- Worktree：开始时 clean。
- GPU：0/1/2/3 均空闲，仅 Xorg 占用约 `4 MiB`。
- Full-grid eval 使用物理 GPU `1`，通过 `CUDA_VISIBLE_DEVICES=1` 暴露为进程内 `cuda:0`。

### NE000_1a：E007 W4A32 packed equivalence

输入 checkpoint：

- `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`

Packed export：

- Export dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a32_packed/e007_normalized_w4a32_single_gpu1_final`

Artifact size：

| item | bytes | MiB |
|---|---:|---:|
| source `.pth` | `5,320,020` | `5.0736` |
| `weights.bin` | `213,632` | `0.2037` |
| `aux_fp32.bin` | `40,860` | `0.0390` |
| `manifest.json` | `94,254` | `0.0899` |
| `summary.json` | `2,695` | `0.0026` |
| raw deployment payload | `254,492` | `0.2427` |
| total export files | `348,746` | `0.3326` |

Export summary：

- `estimated_model_compression_ratio=6.765430740455495`
- `quantized_layer_count=52`
- `activation_quantizer_count=0`
- `final_quant_state={"weight_quant": true, "act_quant": false}`

Packed grid eval：

- Eval run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/eval/20260509_225448_ne000_1a_w4a32_packed_grid478_seed20260507`
- Row count：
  - `11950`
- Restore summary：
  - `restored_quantized_layers=52`
  - `restored_non_quantized_tensors=70`
  - `restored_activation_quantizers=0`
  - `final_quant_state={"weight_quant": true, "act_quant": false}`

Overall：

| metric | checkpoint | packed-restored | packed-checkpoint |
|---|---:|---:|---:|
| SNR mean | `17.785581719` | `17.785577215` | `-0.000004504` |
| SNR median | `18.112756971` | `18.112657055` | `-0.000000972` |
| SSIM mean | `0.964137084` | `0.964137092` | `0.000000008` |
| SSIM median | `0.978461333` | `0.978461489` | `-0.000000009` |

Prediction diff：

- `packed_vs_checkpoint_mse_mean=3.9482866637798976e-10`
- `packed_vs_checkpoint_mean_abs_diff_mean=1.2906794042757771e-05`
- `packed_vs_checkpoint_max_abs_diff_mean=0.0002254785246958091`
- `packed_vs_checkpoint_max_abs_diff_max=0.0005853455513715744`

By-source：

| source | rows | checkpoint SNR mean | packed SNR mean | packed-checkpoint SNR mean |
|---|---:|---:|---:|---:|
| Anisotropic | `1875` | `21.971754726` | `21.971745046` | `-0.000009680` |
| Kerry3D | `400` | `9.608162795` | `9.608165681` | `0.000002886` |
| Shots0001 | `9675` | `17.312392384` | `17.312388578` | `-0.000003806` |

### NE000_1b：NE000 W4A8 packed equivalence

输入 checkpoint：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction/quant/20260509_213701_normalized_w4a8_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq.pth`

Packed export：

- Export dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a8_packed/ne000_normalized_w4a8_tensor_a5000_final`

Artifact size：

| item | bytes | MiB |
|---|---:|---:|
| source `.pth` | `5,359,500` | `5.1112` |
| `weights.bin` | `213,632` | `0.2037` |
| `aux_fp32.bin` | `41,276` | `0.0394` |
| `manifest.json` | `121,554` | `0.1159` |
| `summary.json` | `2,709` | `0.0026` |
| raw deployment payload | `254,908` | `0.2431` |
| total export files | `376,462` | `0.3590` |

Export summary：

- `estimated_model_compression_ratio=6.765430740455495`
- `quantized_layer_count=52`
- `activation_quantizer_count=52`
- `final_quant_state={"weight_quant": true, "act_quant": true}`

Packed grid eval：

- Eval run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/eval/20260509_225927_ne000_1b_w4a8_packed_grid478_seed20260507`
- Row count：
  - `11950`
- Restore summary：
  - `restored_quantized_layers=52`
  - `restored_non_quantized_tensors=70`
  - `restored_activation_quantizers=52`
  - `final_quant_state={"weight_quant": true, "act_quant": true}`

Overall：

| metric | checkpoint | packed-restored | packed-checkpoint |
|---|---:|---:|---:|
| SNR mean | `17.449507172` | `17.449533809` | `0.000026637` |
| SNR median | `17.877688616` | `17.880496749` | `0.000000000` |
| SSIM mean | `0.962868418` | `0.962868814` | `0.000000396` |
| SSIM median | `0.977292375` | `0.977292636` | `0.000000000` |

Prediction diff：

- `packed_vs_checkpoint_mse_mean=3.7573586346050463e-07`
- `packed_vs_checkpoint_mean_abs_diff_mean=0.00012680872421293757`
- `packed_vs_checkpoint_max_abs_diff_mean=0.0025956151830850787`
- `packed_vs_checkpoint_max_abs_diff_max=0.018667370080947876`

By-source：

| source | rows | checkpoint SNR mean | packed SNR mean | packed-checkpoint SNR mean |
|---|---:|---:|---:|---:|
| Anisotropic | `1875` | `21.287276634` | `21.287250178` | `-0.000026456` |
| Kerry3D | `400` | `9.570384048` | `9.570566086` | `0.000182038` |
| Shots0001 | `9675` | `17.031505261` | `17.031535762` | `0.000030501` |

结论：

- W4A32 packed-restored 与 E007 W4A32 checkpoint final 高度对齐，mean SNR delta 约 `-4.5e-06 dB`。
- W4A8 packed-restored 与 NE000 W4A8 checkpoint final 高度对齐，mean SNR delta 约 `+2.66e-05 dB`。
- 两者均远小于 `0.01 dB` 验收阈值。
- NE000 的 W4A8 结果不仅存在于恢复型 `.pth` fake-quant checkpoint 中；packed 整数权重导出、activation qparams 恢复和部署视角 PyTorch restore 链路均通过 full-grid 等价验证。
- 下一步可以进入 NE001 diagnostics。

## 2026-05-09 NE000 目录排序重命名

为让 NE000 baseline 目录在排序中位于 `NE000_1_packed_deployment_equivalence` 之前，将 NE000 原始目录重命名：

- 旧目录：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_normalized_w4a8_activation_reconstruction`
- 新目录：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction`

同步处理：

- `SCRN_BRECQ_app/` 下文本型记录文件中的路径已替换为新目录名。
- 本日志和 `DEVELOPMENT_LOG.md` 中的 NE000 checkpoint / eval 路径已同步更新。
- 二进制 `.pth` checkpoint 内部历史运行配置不做原地改写；模型加载、验证和 grid evaluation 不依赖该路径字段。

当前 NE000 排序：

- `NE000_0_normalized_w4a8_activation_reconstruction`
- `NE000_1_packed_deployment_equivalence`

## 2026-05-09 NE000_2 归一化 W4A4 激活量化探针

在 NE001 diagnostics 前补做 W4A4 压力探针：保持 E007 W4A32 权重起点和 NE000 activation reconstruction 设置不变，仅将 activation bitwidth 从 A8 改成 A4，观察 normalized 协议下 tensor-wise W4A4 的可用性。

### 执行口径

- 实验名：
  - `NE000_2_normalized_w4a4_activation_reconstruction_probe`
- 原始 E007 W4A32 checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- A4 seed metadata checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- 说明：
  - 现有 `activation_only_quantize_scrn.py` 的 CLI 没有 `--n-bits-a` 参数。
  - E007 checkpoint 的 `quant_config.n_bits_a=8` 会在配置合并时覆盖外部 config。
  - 因此本次生成一个仅修改 `quant_config.n_bits_a: 8 -> 4` 的 metadata seed；权重张量保持 E007 不变，不代表新的 W4 weight reconstruction。

核心配置：

- `n_bits_w=4`
- `n_bits_a=4`
- `activation_granularity=tensor`
- `activation_range_method=none`
- `num_samples=1024`
- `batch_size=16`
- `init_batch_size=64`
- `iters_a=5000`
- `activation_lr=0.0004`
- `lp_norm=2.4`
- GPU：物理 GPU `1`

### Quant run

- Run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/quant/20260509_232313_normalized_w4a4_tensor_a5000_1024cali_single_gpu1`
- Pre-act checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/quant/20260509_232313_normalized_w4a4_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- Final checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/quant/20260509_232313_normalized_w4a4_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq.pth`

时间：

- Activation initialization：`24.8694 s`
- Activation reconstruction：`1328.2162 s` / `22.1369 min`
- Total elapsed：`1356.1941 s` / `22.6032 min`

Single-sample sanity：

| metric | value |
|---|---:|
| `post_weight_snr` | `13.8918` |
| `pre_act_snr` | `13.1723` |
| `act_init_delta` | `-0.7195` |
| `quant_post_act_recon_snr_db` | `13.2697` |
| `quant_act_recon_snr_gain_db` | `0.0974` |
| `non_positive_delta_count` | `0` |

### Checkpoint verification

Pre-act：

- `passed=true`
- `final_quant_state={"weight_quant": true, "act_quant": true}`
- `quant_config.n_bits_a=4`
- `activation_delta_count=52`
- `activation_zero_point_count=52`
- `weight_bit_counts={"4": 50, "8": 2}`
- `level_offender_count=0`

Final：

- `passed=true`
- `final_quant_state={"weight_quant": true, "act_quant": true}`
- `quant_config.n_bits_a=4`
- `activation_delta_count=52`
- `activation_zero_point_count=52`
- `weight_bit_counts={"4": 50, "8": 2}`
- `level_offender_count=0`

额外 delta 检查：

| checkpoint | delta count | non-positive | min delta | max delta |
|---|---:|---:|---:|---:|
| pre-act | `52` | `0` | `0.0088924468` | `0.5482574105` |
| final | `52` | `0` | `0.0088924468` | `0.4546860754` |

### Normalized 478x25 grid eval

- Eval run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/eval/20260509_234948_normalized_w4a4_tensor_a5000_grid478_seed20260507`
- Row count：`11950`
- Eval elapsed：`238.4515 s`
- Test set：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- Grid：
  - SNR settings：`-2,-1,1,5,10`
  - missing rates：`0.02,0.08,0.18,0.28,0.38`
  - seed：`20260507`

Overall：

| model | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| FP32 | `17.832885` | `18.174198` | `0.964330` | `0.978794` |
| E007 W4A32 final | `17.785582` | `18.112757` | `0.964137` | `0.978461` |
| NE000 W4A8 final | `17.449507` | `17.877689` | `0.962868` | `0.977292` |
| NE000_2 W4A4 pre-act | `11.172733` | `11.157722` | `0.941492` | `0.955772` |
| NE000_2 W4A4 final | `12.914963` | `13.118019` | `0.939563` | `0.954078` |

W4A4 final gaps：

| comparison | SNR mean gap | SNR median gap | SSIM mean gap | SSIM median gap |
|---|---:|---:|---:|---:|
| W4A4 final - FP32 | `-4.917922` | `-4.152035` | `-0.024767` | `-0.025209` |
| W4A4 final - E007 W4A32 | `-4.870618` | `-4.994738` | `-0.024574` | `-0.024384` |
| W4A4 final - NE000 W4A8 | `-4.534544` | `-4.759669` | `-0.023305` | `-0.023215` |
| W4A4 final - W4A4 pre-act | `+1.742231` | `+1.861192` | `-0.001929` | `-0.002785` |

By-source：

| source | rows | FP32 SNR mean | W4A4 pre SNR mean | W4A4 final SNR mean | final-FP32 SNR mean | W4A4 final SSIM mean |
|---|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `22.059463` | `10.712972` | `13.280217` | `-8.779246` | `0.966844` |
| Kerry3D | `400` | `9.608488` | `9.108445` | `9.033177` | `-0.575312` | `0.929975` |
| Shots0001 | `9675` | `17.353808` | `11.347179` | `13.004665` | `-4.349143` | `0.934673` |

### 判断

- W4A4 是真实 A4 activation fake quant：checkpoint `quant_config.n_bits_a=4`，52 个 activation quantizer 均有 delta/zero_point，pre/final reload verification 均通过。
- W4A4 没有完全崩坏，但相对 NE000 W4A8 final 低约 `4.53 dB` mean SNR，相对 E007 W4A32 低约 `4.87 dB` mean SNR。
- Activation reconstruction 对 W4A4 的 full-grid SNR 有明显帮助（`+1.74 dB` mean），但 SSIM 轻微变差，说明当前 reconstruction 目标对 A4 的结构保持不够稳。
- 暂不把 W4A4 进入 packed deployment；NE001 主线仍以 NE000 W4A8 为诊断对象，W4A4 作为后续 A4 range / granularity / mixed precision 的压力对照保留。

## 2026-05-10 NE000_1c：NE000_2 W4A4 packed deployment 等价验证

本节补齐 W4A4 的部署等价验证：对 `NE000_2` W4A4 final checkpoint 执行 packed export，并在 normalized `478 x 25` grid 上比较 packed-restored 输出与原 checkpoint final 输出。

### Packed export

- 输入 checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/quant/20260509_232313_normalized_w4a4_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- Export dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a4_packed/ne000_2_normalized_w4a4_tensor_a5000_final`
- Manifest / summary：
  - `final_quant_state={"weight_quant": true, "act_quant": true}`
  - `quant_config.n_bits_a=4`
  - `quantized_layer_count=52`
  - `activation_quantizer_count=52`
  - `estimated_model_compression_ratio=6.765430740455495`

Artifact sizes：

| item | bytes | MiB |
|---|---:|---:|
| source `.pth` | `5359500` | `5.1112` |
| `weights.bin` | `213632` | `0.2037` |
| `aux_fp32.bin` | `41276` | `0.0394` |
| `manifest.json` | `121590` | `0.1160` |
| `summary.json` | `2708` | `0.0026` |
| raw deployment payload | `254908` | `0.2431` |
| export files including `summary.json` | `379206` | `0.3616` |

### Packed grid eval

- Eval run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/eval/20260510_001507_ne000_1c_w4a4_packed_grid478_seed20260507`
- Row count：`11950`
- Eval elapsed：`237.7457 s`
- Restore summary：
  - `restored_quantized_layers=52`
  - `restored_non_quantized_tensors=70`
  - `restored_activation_quantizers=52`
  - `final_quant_state={"weight_quant": true, "act_quant": true}`

Overall equivalence：

| metric | checkpoint | packed-restored | packed-checkpoint |
|---|---:|---:|---:|
| SNR mean | `12.914963390` | `12.915036109` | `0.000072719` |
| SNR median | `13.118019299` | `13.120819498` | `0.000000000` |
| SSIM mean | `0.939563064` | `0.939564812` | `0.000001748` |
| SSIM median | `0.954077528` | `0.954055301` | `0.000000000` |

Prediction diff：

- `packed_vs_checkpoint_mse_mean=3.893126945986073e-06`
- `packed_vs_checkpoint_mean_abs_diff_mean=0.0003266090435713198`
- `packed_vs_checkpoint_max_abs_diff_mean=0.009579205599552113`
- `packed_vs_checkpoint_max_abs_diff_max=0.07725390791893005`

By-source：

| source | rows | checkpoint SNR mean | packed SNR mean | packed-checkpoint SNR mean | checkpoint SSIM mean | packed SSIM mean |
|---|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `13.280216794` | `13.280417211` | `0.000200416` | `0.966843563` | `0.966847291` |
| Kerry3D | `400` | `9.033176792` | `9.033380279` | `0.000203487` | `0.929975485` | `0.929969113` |
| Shots0001 | `9675` | `13.004665148` | `13.004707713` | `0.000042565` | `0.934672531` | `0.934674231` |

### 后续核心对比对象

| object | SNR mean | SNR median | SSIM mean | SSIM median | role |
|---|---:|---:|---:|---:|---|
| FP32 | `17.832885` | `18.174198` | `0.964330` | `0.978794` | normalized upper baseline |
| E007 W4A32 final | `17.785582` | `18.112757` | `0.964137` | `0.978461` | default W4 weight-only baseline |
| NE000 W4A8 final | `17.449507` | `17.877689` | `0.962868` | `0.977292` | current usable activation quantization baseline |
| NE000_2 W4A4 final | `12.914963` | `13.118019` | `0.939563` | `0.954078` | A4 pressure reference, not deployment candidate |

### 判断

- W4A4 packed-restored 与 NE000_2 W4A4 checkpoint final 高度对齐，mean SNR delta 约 `7.27e-05 dB`，远小于 `0.01 dB` 阈值。
- W4A4 是合法、可导出、可恢复的 A4 activation 压力对照，但当前 full-grid 指标明显低于 W4A8，不作为部署候选。
- W4A8 是当前最重要的可用 activation quantization baseline；NE001 主线仍应优先诊断 NE000 W4A8。

## 2026-05-10 四个核心结果与重建前后结论汇总

本节补充记录 NE000 系列到目前为止最重要的横向结论。这里的“四个核心结果”指：

1. normalized FP32 baseline；
2. E007 W4A32 final，当前默认 weight-only baseline；
3. NE000 W4A8 final，当前最重要的可用 activation quantization baseline；
4. NE000_2 W4A4 final，合法可部署恢复的 A4 压力对照。

### 四个核心结果的最终指标

| object | SNR mean | SNR median | SSIM mean | SSIM median | FP32 SNR gap mean |
|---|---:|---:|---:|---:|---:|
| FP32 | `17.832885` | `18.174198` | `0.964330` | `0.978794` | `0.000000` |
| E007 W4A32 final | `17.785582` | `18.112757` | `0.964137` | `0.978461` | `-0.047304` |
| NE000 W4A8 final | `17.449507` | `17.877689` | `0.962868` | `0.977292` | `-0.383378` |
| NE000_2 W4A4 final | `12.914963` | `13.118019` | `0.939563` | `0.954078` | `-4.917922` |

跨对象差距：

| comparison | SNR mean delta | SNR median delta | SSIM mean delta |
|---|---:|---:|---:|
| E007 W4A32 final - FP32 | `-0.047304` | `-0.061441` | `-0.000193` |
| NE000 W4A8 final - E007 W4A32 final | `-0.336075` | `-0.235068` | `-0.001269` |
| NE000_2 W4A4 final - NE000 W4A8 final | `-4.534544` | `-4.759669` | `-0.023305` |
| NE000_2 W4A4 final - E007 W4A32 final | `-4.870618` | `-4.994738` | `-0.024574` |

### 三个重建实验的 pre/post 改变量

这三个模型都涉及重建，pre/post 改变量是判断该阶段是否真正有效的关键指标。

| model | pre SNR mean | post SNR mean | SNR gain | pre SSIM mean | post SSIM mean | SSIM gain | post-FP32 SNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| E007 W4A32 weight recon | `16.614250` | `17.785582` | `+1.171332` | `0.935462` | `0.964137` | `+0.028675` | `-0.047304` |
| NE000 W4A8 act recon | `17.372734` | `17.449507` | `+0.076773` | `0.962674` | `0.962868` | `+0.000195` | `-0.383378` |
| NE000_2 W4A4 act recon | `11.172733` | `12.914963` | `+1.742231` | `0.941492` | `0.939563` | `-0.001929` | `-4.917922` |

Median 视角：

| model | pre SNR median | post SNR median | SNR gain median | pre SSIM median | post SSIM median | SSIM gain median |
|---|---:|---:|---:|---:|---:|---:|
| E007 W4A32 weight recon | `17.177490` | `18.112757` | `+0.805456` | `0.949638` | `0.978461` | `+0.023764` |
| NE000 W4A8 act recon | `17.785509` | `17.877689` | `+0.006535` | `0.977030` | `0.977292` | `+0.000069` |
| NE000_2 W4A4 act recon | `11.157722` | `13.118019` | `+1.861192` | `0.955772` | `0.954078` | `-0.002785` |

结论：

- E007 W4A32 的 weight reconstruction 是强有效步骤：mean SNR `+1.17 dB`，mean SSIM `+0.0287`，最终距离 FP32 只剩 `0.047 dB`，因此它是后续所有 activation quantization 的稳固起点。
- NE000 W4A8 的 activation reconstruction 是小幅精修：A8 init 本身已经接近 final，mean SNR 只再提升 `0.0768 dB`，说明新 normalized 协议下 A8 的主要改善来自数据/幅值协议和合法量化状态，而不是 activation reconstruction 大幅救回。
- NE000_2 W4A4 的 activation reconstruction 对 SNR 有强正收益：mean SNR `+1.742 dB`，但 SSIM 反而下降 `0.00193`，说明当前 reconstruction 目标对 A4 会优先优化能量误差，不一定同步改善结构相似性。
- A4 final 虽然合法且可恢复，但仍比 W4A8 final 低 `4.53 dB` mean SNR；A4 后续需要 range/granularity/mixed precision 的专门路线，不能直接当部署候选。

### 按 source 看重建收益

E007 W4A32：

| source | rows | pre SNR | post SNR | SNR gain | post-FP32 SNR | pre SSIM | post SSIM | SSIM gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `20.003475` | `21.971755` | `+1.968280` | `-0.087708` | `0.981766` | `0.992832` | `+0.011066` |
| Kerry3D | `400` | `9.437826` | `9.608163` | `+0.170337` | `-0.000326` | `0.950352` | `0.949887` | `-0.000465` |
| Shots0001 | `9675` | `16.254123` | `17.312392` | `+1.058270` | `-0.041416` | `0.925873` | `0.959165` | `+0.033292` |

NE000 W4A8：

| source | rows | pre SNR | post SNR | SNR gain | post-FP32 SNR | pre SSIM | post SSIM | SSIM gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `21.289510` | `21.287277` | `-0.002234` | `-0.772186` | `0.991605` | `0.991631` | `+0.000026` |
| Kerry3D | `400` | `9.581705` | `9.570384` | `-0.011321` | `-0.038104` | `0.948793` | `0.948695` | `-0.000098` |
| Shots0001 | `9675` | `16.935779` | `17.031505` | `+0.095726` | `-0.322303` | `0.957641` | `0.957880` | `+0.000239` |

NE000_2 W4A4：

| source | rows | pre SNR | post SNR | SNR gain | post-FP32 SNR | pre SSIM | post SSIM | SSIM gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `10.712972` | `13.280217` | `+2.567245` | `-8.779246` | `0.969940` | `0.966844` | `-0.003096` |
| Kerry3D | `400` | `9.108445` | `9.033177` | `-0.075269` | `-0.575312` | `0.939263` | `0.929975` | `-0.009288` |
| Shots0001 | `9675` | `11.347179` | `13.004665` | `+1.657486` | `-4.349143` | `0.936071` | `0.934673` | `-0.001399` |

Source 结论：

- `Anisotropic` 是 activation bitwidth 压力最强的 source：W4A8 final 已有 `-0.772 dB` gap，W4A4 final 扩大到 `-8.779 dB`。后续 NE001/NE004 应重点检查它对应样本上的 activation range、Conv2d fake-quant error 和 stage 输出误差。
- `Kerry3D` 对 activation bitwidth 最不敏感：W4A8 gap 只有 `-0.038 dB`，W4A4 gap 也只有 `-0.575 dB`，说明 A4 崩坏不是所有 source 的统一现象。
- `Shots0001` 是总体结论的主导 source，因为样本数 `9675/11950`；W4A8 在这里仍有 `-0.322 dB` gap，W4A4 为 `-4.349 dB`。
- W4A8 activation reconstruction 对 Anisotropic/Kerry3D 的 SNR 没有正收益，主要收益来自 Shots0001；因此 NE001 不应只看 overall gain。
- W4A4 activation reconstruction 对 Anisotropic 和 Shots0001 的 SNR 有大幅恢复，但 Kerry3D 轻微变差，并且三个 source 的 SSIM gain 都为负，说明 A4 的重建目标存在 source-dependent tradeoff。

### packed deployment 等价链路

| packed object | checkpoint SNR mean | packed SNR mean | packed-checkpoint SNR mean | pred MSE mean | max abs diff mean |
|---|---:|---:|---:|---:|---:|
| E007 W4A32 packed | `17.785581719` | `17.785577215` | `-0.000004504` | `3.94828666378e-10` | `0.000225479` |
| NE000 W4A8 packed | `17.449507172` | `17.449533809` | `0.000026637` | `3.75735863461e-07` | `0.002595615` |
| NE000_2 W4A4 packed | `12.914963390` | `12.915036109` | `0.000072719` | `3.89312694599e-06` | `0.009579206` |

Deployment 结论：

- W4A32、W4A8、W4A4 三个量化模型都已经通过 packed export / restore / full-grid equivalence；这些结果不只是 `.pth` fake-quant checkpoint 内的偶然状态。
- packed-vs-checkpoint 差异随 activation bitwidth 变低而增大，但三者 mean SNR delta 都远低于 `0.01 dB` 验收阈值。
- 后续比较精度时可以优先引用 checkpoint final 的 grid 指标；讨论部署链路时引用 packed equivalence 证明。

### 模型大小与部署存储对比

Checkpoint / estimated packed size：

| object | checkpoint MiB | estimated packed MiB | estimated compression | quantized weight MiB | weight compression | act quantizers |
|---|---:|---:|---:|---:|---:|---:|
| E007 W4A32 | `5.0736` | `0.2427` | `6.7654x` | `0.2037` | `7.9784x` | `0` |
| NE000 W4A8 | `5.1112` | `0.2427` | `6.7654x` | `0.2037` | `7.9784x` | `52` |
| NE000_2 W4A4 | `5.1112` | `0.2427` | `6.7654x` | `0.2037` | `7.9784x` | `52` |

Packed export files：

| object | `weights.bin` MiB | `aux_fp32.bin` MiB | raw payload MiB | `manifest.json` MiB | `summary.json` MiB | total export MiB |
|---|---:|---:|---:|---:|---:|---:|
| E007 W4A32 | `0.2037` | `0.0390` | `0.2427` | `0.0899` | `0.0026` | `0.3352` |
| NE000 W4A8 | `0.2037` | `0.0394` | `0.2431` | `0.1159` | `0.0026` | `0.3616` |
| NE000_2 W4A4 | `0.2037` | `0.0394` | `0.2431` | `0.1160` | `0.0026` | `0.3616` |

Parameter / layer accounting：

| object | base params | quantized weight params | non-quantized params | 4bit layers | 8bit layers |
|---|---:|---:|---:|---:|---:|
| E007 W4A32 | `430437` | `426112` | `4325` | `50` | `2` |
| NE000 W4A8 | `430437` | `426112` | `4325` | `50` | `2` |
| NE000_2 W4A4 | `430437` | `426112` | `4325` | `50` | `2` |

Size 结论：

- 三个量化模型的权重存储完全一致：`weights.bin=0.2037 MiB`，因为它们共享同一套 W4 weight reconstruction，且 head/tail 保持 8bit。
- W4A8/W4A4 的 estimated packed model size 与 W4A32 基本相同，都是 `0.2427 MiB`；当前 packed 估算主要统计权重和必要 FP32 aux，不把 activation bitwidth 带来的运行时激活张量存储作为模型文件大小收益。
- W4A8/W4A4 比 W4A32 多出 activation qparams，因此 checkpoint 从 `5.0736 MiB` 增至 `5.1112 MiB`，`aux_fp32.bin` 从 `0.0390 MiB` 增至 `0.0394 MiB`。
- W4A8 与 W4A4 的模型文件大小几乎相同，因为两者都保存 52 个 activation `delta/zero_point` 浮点参数；activation bitwidth 影响推理 fake quant 网格和精度，不显著改变当前 packed artifact 的文件体积。
- 实际 export 文件总大小大于 raw payload，是因为 `manifest.json` 和 `summary.json` 记录了层级元数据、量化状态和可复现实验信息；部署时真正的二进制 payload 主要是 `weights.bin + aux_fp32.bin`。

### 当前研究判断和后续优先级

- 默认对比表应固定为 FP32 / E007 W4A32 / NE000 W4A8 / NE000_2 W4A4。
- 后续 NE001 diagnostics 的主对象仍是 NE000 W4A8，因为它是当前最接近可用部署的 activation quantization 结果。
- NE000_2 W4A4 应作为压力对照进入诊断视野，但不应把主线改成 A4 优化；A4 的主要价值是放大 activation quantization 热点。
- 下一步诊断必须同时报告 pre/post reconstruction，而不是只报告 final。特别是 W4A8 的 final gain 很小，单看 final 容易误判 activation reconstruction 的作用。
- 如果 NE001 发现 W4A8 和 W4A4 的误差热点高度重合，NE004/NE006 的 Conv2d sensitivity 和结构化 granularity 搜索可能同时解释 A8 gap 和 A4 崩坏。
- 如果 W4A4 的热点与 W4A8 不重合，则 A4 应另开 A4-specific range/granularity/mixed precision 路线，不能用 W4A8 的诊断结论直接外推。

## 2026-05-10 NE001-NE006 W4A4 主线实验计划

在正式开始 NE001 前，重新记录后续实验主线。NE000 和 NE000_1/2 已经确认：W4A8 在 normalized 协议下表现很好，W4A4 合法、可恢复、可 packed 等价，但仍有约 `4.53 dB` mean SNR 的 A4 activation gap。因此后续主对象从“修 W4A8”调整为“以 W4A8 为成功参照，重点攻 W4A4”。

本节是对上一节“当前研究判断和后续优先级”的更新：旧文字保留为历史判断，本节之后执行 NE001-NE006 时以 W4A4 主线为准。

### 固定实验对象

| object | role | 使用方式 |
|---|---|---|
| FP32 | 上限基准 | 不做 activation quantization 实验，只作为 full-grid 指标、逐样本输出和可视化上限参考。 |
| E007 W4A32 final | 权重量化起点 | 后续 A4/A8 activation quantization 都从该 checkpoint 出发；同时作为 weight-only gap 参考。 |
| NE000 W4A8 final | 成功参照与护栏 | 不作为主要优化对象；用于判断 W4A4 热点是否和 A8 一致，并验证新策略不能明显破坏 A8。 |
| NE000_2 W4A4 final | 主研究对象 | 后续诊断、敏感性、range、granularity 和 mixed precision 优先围绕它展开。 |

### 全局执行口径

- 数据协议固定为 `paper5_energy_filtered_perpatch_absmax`。
- 正式判断固定使用 normalized `478 x 25` grid：SNR settings `-2,-1,1,5,10`，missing rates `0.02,0.08,0.18,0.28,0.38`，seed `20260507`。
- 每个涉及 reconstruction 的实验必须报告 pre/post 改变量，不能只报告 final。
- 单样本和代表图只作为解释材料，不能替代 full-grid mean/median、by-source、by-condition 指标。
- 运行资源优先使用 GPU；单卡默认优先级仍为 `1 -> 2 -> 3 -> 0`，可并行 sweep 时优先做 job-level 多 GPU 并行。
- 所有新产物只写入 `SCRN_BRECQ_app/` 下；涉及 `scrn_brecq/` 代码或文档时同步更新两份日志并提交。

### 后续实验安排

| 编号 | 实验对象 | 实验内容 | 实验方法 | 实验目标 | 主要输出 / 决策 |
|---|---|---|---|---|---|
| NE001 | 主：NE000_2 W4A4 pre/final；参照：NE000 W4A8 pre/final、E007 W4A32、FP32 | W4A4-centered activation diagnostics。 | 对 W4A4 pre-act 与 final checkpoint 跑 activation quantizer 诊断，统计 activation qparams、delta/zero_point 合法性、fake-quant error、effective levels、module type、stage、role、source 和 condition 分布；同时抽 W4A8 做同口径对照。 | 找到 W4A4 主要误差热点，判断是否和 W4A8 热点重合，明确后续该优先修 Conv2d、Linear/transformer、stage output 还是特定 source/condition。 | quantizer 清单、热点表、by-source/stage/role error ranking、W4A4 vs W4A8 热点重合度；决定 NE004 的 sensitivity 分组。 |
| NE002 | 主：NE000_2 W4A4；参照：NE000 W4A8、E007 W4A32 packed | 合法状态和 checkpoint / deployment sanity。 | 复核 `verify_quantized_scrn`、packed manifest、summary、reload 后 final quant state、activation bitwidth、52 个 activation quantizer、`non_positive_delta_count=0`、checkpoint-vs-packed 等价；必要时补充 state toggle sanity。 | 排除“结果看起来好/坏是因为未真正启用 activation quantization、reload 状态错误、packed restore 不一致或 bitwidth 配置错位”。 | W4A4/A8 legality 表；若发现状态问题，先修状态再继续；若无问题，确认 NE003-NE006 都可使用当前 checkpoint。 |
| NE003 | 四个核心对象 | 固定评估和代表图口径。 | 复核 full-grid 聚合、by-source、by-SNR、by-missing-rate、by-condition；选择固定代表样本，使用 seismic colormap 和反归一化/幅值一致显示，展示 FP32 / W4A32 / W4A8 / W4A4 以及误差图。 | 让后续所有改进都有统一可视化和数值解释口径，避免单样本偶然性或显示尺度误导。 | 代表样本清单、图像输出目录、四对象同图对比、W4A4 最差/中位/最好样本案例；后续报告固定复用。 |
| NE004 | 主：W4A4；参照：W4A8 | Activation quantizer 敏感性。 | 基于 NE001 热点做分组关闭/保留实验，优先按 Conv2d vs Linear/transformer、split_proj、merge_proj、stage_output_conv、stage1-5、source-sensitive groups 分组；先跑小规模 scout，再对关键组跑 full-grid。 | 判断 W4A4 的 `4.53 dB` gap 由哪些 activation quantizer 组主导，并验证旧协议“Conv2d 多层累积误差”在 A4 下是否仍成立。 | 分组恢复表、group ranking、source-specific recovery；决定 NE005 是否值得做 range，NE006 应优先处理哪些结构。 |
| NE005 | 主：W4A4 的 NE004 目标分组 | Range、clipping、outlier 和 calibration 代表性检查。 | 在不改变 granularity 的前提下，测试 tensor-wise max、percentile、MSE-grid、source/condition-aware calibration、目标分组 range variants；只对 NE004 指出的关键组做完整验证。 | 判断 A4 主要问题是否能通过 range / clipping / calibration 修复，还是必须进入结构化粒度或 mixed precision。 | range variant 对比表；若 tensor-wise range 有效，进入更精细 range；若无效，收束到 NE006 粒度/混精度。 |
| NE006 | 主：W4A4；护栏：W4A8 | 结构化 activation granularity / mixed precision 搜索。 | 从旧 E006 强候选开始：all Conv2d per-channel、split_proj + merge_proj + stage_output_conv per-channel、对应 group-wise `g4`、stage_output_conv `g4`；在 A4 下必要时加入 selective A8 fallback 或 selective FP32。 | 尝试把 W4A4 从压力对照推进到可用候选；同时确认策略不会明显破坏 W4A8。 | W4A4 候选策略表、pre/post 指标、packed 可行性判断；若出现接近 W4A8 的 A4 候选，再做 packed export/equivalence。 |

### 阶段性判断规则

- 如果 NE001 显示 W4A4 和 W4A8 热点高度重合，后续优先复用旧 E006 的结构化 Conv2d 粒度路线，只是把目标 bitwidth 改成 A4。
- 如果 W4A4 热点明显不同于 W4A8，后续不要把 W4A8 的好结果外推到 A4；NE004-NE006 必须以 A4-specific 分组和 source-specific 结论为准。
- 如果 W4A4 的 SSIM 持续随 reconstruction 或 range 改善而下降，需要单独记录 SNR/SSIM tradeoff，并考虑 reconstruction loss 或代表样本可视化，而不是只追求 SNR。
- W4A8 目前不需要主动优化，但每个最终候选策略都应至少用 W4A8 做护栏验证，防止为了 A4 引入会破坏 A8 的工程默认配置。

## 2026-05-10 NE001 full-reference activation diagnostics

完成 `NE001`：以 `NE000_2 W4A4 pre/final` 为主诊断对象，同时纳入 `NE000 W4A8 pre/final`、`E007 W4A32 pre/final` 和 FP32 上限指标，建立完整参考链条。

### 运行口径

- Branch：`main`
- 开始时 worktree：clean，本地领先 `origin/main` 28 个提交。
- 数据协议：`paper5_energy_filtered_perpatch_absmax`
- Calibration：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- Diagnostics 参数：
  - `num_samples=1024`
  - `batch_size=16`
  - `num_workers=0`
  - `seed=1005`
- 输出根目录：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics`

GPU 说明：

- 预检时 GPU `0/1/2/3` 基本空闲，默认优先选择 GPU `1`。
- 首次按计划使用 GPU `1` 跑 `W4A4 pre-act` diagnostics 时，sandbox 外 CUDA 可用，但现有 `diagnose_activation_quantization` 会把 1024 个 calibration samples 整批送入一次 forward；`batch_size=16` 只影响 DataLoader，不影响 diagnostics forward。
- GPU run 在 full-batch activation diagnostics 中 OOM：
  - 错误点：`_per_channel_absmax_stats`
  - 现象：尝试额外分配约 `4.00 GiB`，进程已占约 `12.38 GiB`
- 为保持“不改代码、不降 num_samples”的计划约束，本次 6 个 diagnostics 全部改用 CPU fallback 完成，并在 run name 中显式标注 `cpu_fallback`。

### Diagnostics run directories

| object | run dir |
|---|---|
| W4A4 pre-act | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics/20260510_015710_ne001a_w4a4_pre_act_diagnostics_1024cali_cpu_fallback` |
| W4A4 final | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics/20260510_020106_ne001b_w4a4_final_diagnostics_1024cali_cpu_fallback` |
| W4A8 pre-act | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics/20260510_020454_ne001c_w4a8_pre_act_reference_diagnostics_1024cali_cpu_fallback` |
| W4A8 final | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics/20260510_020847_ne001d_w4a8_final_reference_diagnostics_1024cali_cpu_fallback` |
| W4A32 pre-recon | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics/20260510_021257_ne001e_w4a32_pre_recon_reference_diagnostics_1024cali_cpu_fallback` |
| W4A32 final | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics/20260510_021648_ne001f_w4a32_final_reference_diagnostics_1024cali_cpu_fallback` |

每个 run 均生成：

- `config.json`
- `summary.json`
- `summary.md`
- `quantizers.csv`
- `activation_stats.jsonl`
- `offender_layers.json`

### 八个对象的 fixed-grid 指标

FP32 checkpoint：

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`

| object | SNR mean | SNR median | SSIM mean | SSIM median | post/pre gain SNR mean |
|---|---:|---:|---:|---:|---:|
| FP32 | `17.832885` | `18.174198` | `0.964330` | `0.978794` | N/A |
| E007 W4A32 pre-recon | `16.614250` | `17.177490` | `0.935462` | `0.949638` | N/A |
| E007 W4A32 final | `17.785582` | `18.112757` | `0.964137` | `0.978461` | `+1.171332` |
| NE000 W4A8 pre-act | `17.372734` | `17.785509` | `0.962674` | `0.977030` | N/A |
| NE000 W4A8 final | `17.449507` | `17.877689` | `0.962868` | `0.977292` | `+0.076773` |
| NE000_2 W4A4 pre-act | `11.172733` | `11.157722` | `0.941492` | `0.955772` | N/A |
| NE000_2 W4A4 final | `12.914963` | `13.118019` | `0.939563` | `0.954078` | `+1.742231` |

### Legality / diagnostics summary

| object | final state | act bits | delta/init | non-positive delta | delta min/max | fake-quant MSE mean/max | effective levels min/max |
|---|---|---:|---:|---:|---:|---:|---:|
| W4A4 pre | `W=true A=true` | `4/8` | `52/52` | `0` | `0.00889245 / 0.548257` | `0.00612955 / 0.027745` | `13 / 246` |
| W4A4 final | `W=true A=true` | `4/8` | `52/52` | `0` | `0.00889245 / 0.454686` | `0.00468151 / 0.0464743` | `14 / 235` |
| W4A8 pre | `W=true A=true` | `8` | `52/52` | `0` | `0.00416813 / 0.0480133` | `5.65246e-05 / 0.000505416` | `168 / 256` |
| W4A8 final | `W=true A=true` | `8` | `52/52` | `0` | `0.00401271 / 0.0526452` | `4.43473e-05 / 0.000241724` | `153 / 256` |
| W4A32 pre | `W=true A=false` | `8` | `0/0` | `0` | `None / None` | `None / None` | `None / None` |
| W4A32 final | `W=true A=false` | `8` | `0/0` | `0` | `None / None` | `None / None` | `None / None` |

结论：

- W4A4/W4A8 checkpoint 都是真正启用 activation quantization 的状态：`act_quant=true`，52 个 activation quantizer 均初始化，`non_positive_delta_count=0`。
- W4A32 pre/final 是合法 weight-only 对照：`act_quant=false`，activation delta count 为 `0`，fake-quant MSE 为 `None` 符合预期。
- A4 的 fake-quant MSE 比 A8 高约两个数量级；A4 final 的 mean MSE 比 A4 pre 降低，但 max MSE 从 `0.027745` 升到 `0.0464743`，说明 reconstruction 改善总体误差时把最坏层压力集中到少数层。
- A4 final effective levels 最低只有 `14`，A8 final 最低为 `153`，A4 的核心问题是有效量化级别严重不足，而不是非法 scale。

### Module type 对比

| object | Conv2d MSE mean/max | Conv2d relative MSE mean/max | Linear MSE mean/max | Linear relative MSE mean/max |
|---|---:|---:|---:|---:|
| W4A4 pre | `0.00688291 / 0.0222065` | `0.0351968 / 0.0674264` | `0.00492416 / 0.027745` | `0.0216285 / 0.0524481` |
| W4A4 final | `0.00448758 / 0.0136385` | `0.0273810 / 0.0546570` | `0.00499179 / 0.0464743` | `0.0211767 / 0.0470783` |
| W4A8 pre | `7.25379e-05 / 0.000505416` | `0.000491862 / 0.00160065` | `3.09033e-05 / 0.000137540` | `0.000302101 / 0.000998137` |
| W4A8 final | `5.08725e-05 / 0.000138683` | `0.000451410 / 0.00165058` | `3.39071e-05 / 0.000241724` | `0.000212343 / 0.000597379` |

判断：

- W4A4 reconstruction 明显降低 Conv2d mean/max MSE，但 Linear max MSE 升高，主要来自 attention qkv。
- 旧协议里 “Conv2d 是主瓶颈” 不能直接完整外推到 W4A4。A4 下需要同时看：
  - Conv2d / stage output / fusion 的 relative error；
  - attention qkv 的 absolute MSE；
  - stage5 transformer 的 low effective-level 和 relative error。

### W4A4 final 热点

W4A4 final absolute MSE top layers：

| layer | type | role | stage | fake-quant MSE |
|---|---|---|---|---:|
| `model.stage3.0.block.trans_branch.attn.qkv` | Linear | attention_qkv | stage3 | `0.0464743` |
| `model.stage2.0.block.trans_branch.attn.qkv` | Linear | attention_qkv | stage2 | `0.0226427` |
| `model.stage1.0.block.conv_branch.3` | Conv2d | conv | stage1 | `0.0136385` |
| `model.stage2.0.block.conv_branch.3` | Conv2d | conv | stage2 | `0.0114776` |
| `model.stage4.0.block.trans_branch.attn.qkv` | Linear | attention_qkv | stage4 | `0.0109560` |
| `model.stage3.0.block.conv_branch.0` | Conv2d | conv | stage3 | `0.0107701` |
| `model.stage3.0.block.conv_branch.3` | Conv2d | conv | stage3 | `0.0099101` |
| `model.stage1.0.block.conv_branch.0` | Conv2d | conv | stage1 | `0.00983864` |

W4A4 final relative MSE top layers：

| layer | type | role | stage | relative MSE |
|---|---|---|---|---:|
| `model.stage1.1` | Conv2d | stage_output_conv | stage1 | `0.0546570` |
| `model.stage5.0.block.trans_branch.attn.proj` | Linear | attention_proj | stage5 | `0.0470783` |
| `model.stage5.0.block.trans_branch.mlp.2` | Linear | mlp | stage5 | `0.0442115` |
| `model.stage2.1` | Conv2d | stage_output_conv | stage2 | `0.0439959` |
| `model.stage1.0.block.merge_proj` | Conv2d | merge_proj | stage1 | `0.0431563` |
| `model.stage4.1` | Conv2d | stage_output_conv | stage4 | `0.0419215` |
| `model.stage1.0.block.trans_branch.mlp.2` | Linear | mlp | stage1 | `0.0399722` |
| `model.stage2.0.block.merge_proj` | Conv2d | merge_proj | stage2 | `0.0388929` |

W4A4 final role summary，按 relative MSE mean 排序：

| role | count | MSE mean | relative MSE mean | effective levels min |
|---|---:|---:|---:|---:|
| merge_proj | `5` | `0.00134185` | `0.0388603` | `16` |
| tail | `1` | `0.000599201` | `0.0355458` | `16` |
| stage_output_conv | `5` | `0.000991273` | `0.0342028` | `16` |
| split_proj | `5` | `0.000734917` | `0.0284331` | `16` |
| attention_proj | `5` | `0.000465102` | `0.0240871` | `14` |
| mlp | `10` | `0.000415627` | `0.0232609` | `16` |
| conv | `15` | `0.00851042` | `0.0221592` | `16` |
| attention_qkv | `5` | `0.0186708` | `0.0140978` | `16` |
| head | `1` | `6.80622e-06` | `0.000778224` | `235` |

### W4A4 vs W4A8 热点重合

- W4A4 final 与 W4A8 final 的 absolute MSE top10 有 `7/10` 层重合。
- 重合层：
  - `model.stage1.0.block.conv_branch.3`
  - `model.stage2.0.block.conv_branch.3`
  - `model.stage2.0.block.trans_branch.attn.qkv`
  - `model.stage3.0.block.conv_branch.0`
  - `model.stage3.0.block.trans_branch.attn.qkv`
  - `model.stage4.0.block.conv_branch.3`
  - `model.stage4.0.block.trans_branch.attn.qkv`

判断：

- W4A4 和 W4A8 的热点结构高度重合，但 A4 把这些热点放大到部署不可接受的量级。
- A4 的 absolute MSE 首要热点是 `attention_qkv` 和部分 CNN conv；relative MSE 首要热点是 `stage_output_conv`、`merge_proj`、`stage5 attention_proj/mlp`。
- NE004 不应只复刻旧 E004 的 all Conv2d 关闭实验；应该把 W4A4 分组 sensitivity 设计成：
  1. `attention_qkv`；
  2. CNN `conv_branch`；
  3. `stage_output_conv`；
  4. `split_proj + merge_proj`；
  5. stage5 transformer sanity；
  6. all Conv2d 与 all Linear/transformer 作为全局对照。

### NE001 结论

- W4A4 低指标不是 activation delta 非法、checkpoint reload 错误或没有真正启用 activation quantization；它是真实 A4 bitwidth 压力。
- W4A4 activation reconstruction 对 full-grid SNR 有明显收益，但 diagnostics 显示它更像“整体误差再分配”：Conv2d 误差下降，attention qkv 最坏层压力上升。
- W4A8 当前足够好，不需要把 NE001-NE006 主线用于继续修 W4A8；W4A8 应作为成功参照和护栏。
- 下一步进入 NE004 前，建议先做一个轻量 NE002/NE003 收尾：
  - NE002：确认 full-reference legality / state 表已经满足可继续条件；
  - NE003：固定代表样本图和 full-grid 可视化口径；
  - 然后 NE004 直接以 W4A4 分组 sensitivity 为主。

## 2026-05-11 NE001 结果解释与旧 E 系列对照

本节把 NE001 的诊断结果整理成人类可读结论，重点说明各指标的含义、NE001 与旧 E001-E006 的一致/不一致之处，以及对后续实验的直接指导。

### 一句话结论

NE001 的核心结论是：

> `W4A4` 低指标不是因为 checkpoint 状态错误、activation delta 非法、没有真正启用 activation quantization，或者 packed/reload 失败；它是真实的 A4 activation bitwidth 压力。`W4A8` 的热点结构与 `W4A4` 高度重合，但 A8 有足够量化级别，因此误差仍可接受；A4 把同一批敏感结构的误差放大到不可部署的量级。

后续主线应从“继续修 W4A8”转为：

> 以 W4A8 为成功参照和护栏，重点定位并优化 W4A4 的 A4-specific activation bottleneck。

### NE001 中各类指标分别说明什么

| 指标 | 读法 | NE001 观察 | 说明 |
|---|---|---|---|
| full-grid SNR / SSIM | 最终任务质量，决定模型是否真的可用。 | W4A32 final `17.7856 dB`，W4A8 final `17.4495 dB`，W4A4 final `12.9150 dB`。 | W4A32 几乎贴近 FP32；W4A8 可用；W4A4 有约 `4.53 dB` A4 gap，是主攻对象。 |
| pre/final gain | 重建阶段是否有效。 | W4A4 act recon `+1.7422 dB`，W4A8 act recon `+0.0768 dB`，W4A32 weight recon `+1.1713 dB`。 | W4A4 reconstruction 有明显 SNR 收益，但仍不足；W4A8 reconstruction 只是微调。 |
| `activation_quantizers` | 模型中被包装的 activation quantizer 数量。 | A4/A8/W4A32 都统计到 `52` 个 QuantModule。 | 结构插入位置一致，便于横向对比。 |
| `activation_delta_count` / `initialized_activation_quantizers` | activation quantizer 是否真正初始化。 | W4A4/W4A8 为 `52/52`；W4A32 为 `0/0`。 | W4A4/W4A8 确实启用了 activation quantization；W4A32 正确保持 weight-only。 |
| `non_positive_delta_count` | 是否存在非法 scale。 | A4/A8 全部为 `0`。 | 当前问题不是旧 E001 中的负 delta bug。 |
| fake-quant MSE mean/max | 局部 activation 被量化前后的绝对误差。 | W4A4 final `0.00468 / 0.04647`，W4A8 final `4.43e-05 / 0.000242`。 | A4 局部误差比 A8 高约两个数量级。 |
| effective int levels | 实际使用了多少整数格点。 | W4A4 final min `14`，W4A8 final min `153`。 | A4 的核心瓶颈是表达级别太少；A8 仍有足够离散级别。 |
| absolute MSE top layers | 哪些层制造最大绝对误差。 | W4A4 final top 是 `stage3/stage2/stage4 attention_qkv` 和部分 CNN conv。 | A4 下 attention qkv 不能忽略。 |
| relative MSE top layers | 哪些层相对自身信号最脆弱。 | W4A4 final top 是 `stage_output_conv`、`merge_proj`、`stage5 attention_proj/mlp`。 | A4 的弱点不只在大误差层，也在结构边界和融合层。 |
| top10 热点重合 | A4/A8 是否是同一种结构问题。 | W4A4 final 与 W4A8 final absolute MSE top10 重合 `7/10`。 | 热点位置相似，但 A4 误差量级远大于 A8。 |

### 为什么 W4A4 final SNR 提升但 SSIM 下降

W4A4 activation reconstruction 让 mean SNR 从 `11.1727` 提升到 `12.9150`，但 mean SSIM 从 `0.941492` 降到 `0.939563`。

这说明：

- 当前 reconstruction 目标更接近局部 MSE / 能量误差优化。
- SNR 受能量误差影响较大，因此可以明显提高。
- SSIM 更关注结构相似性；A4 下如果 reconstruction 把误差重新分配到结构敏感位置，SSIM 可能下降。
- 后续 A4 优化不能只看 SNR；NE003 代表图和 by-source SSIM 必须保留。

### 与旧 E 系列的一致之处

| 旧 E 系列结论 | NE001 是否支持 | 说明 |
|---|---|---|
| activation quantization 需要固定多样本评估，单样本不可靠。 | 支持。 | NE001 继续使用 normalized `478 x 25` full-grid 作为最终质量口径，diagnostics 只解释局部机制。 |
| 负 delta 是必须修复的合法性 bug，但不是全部问题。 | 支持。 | NE001 中 A4/A8 都没有负 delta，但 A4 仍明显掉点。 |
| Conv2d activation quantization 是重要误差来源。 | 部分支持。 | W4A4 Conv2d relative MSE 仍高，Conv branch 多层进入 top list。 |
| split/merge/stage_output 是结构化粒度的重要候选。 | 支持且加强。 | W4A4 relative MSE top 明确包含 `stage_output_conv`、`merge_proj`、`split_proj`。 |
| stage5 独立策略需要谨慎。 | 支持。 | W4A4 relative MSE 中 stage5 attention_proj/mlp 很突出，但旧 E006c 发现 stage5 独立细粒度可能有害，后续需要作为 sanity 而不是直接默认优化。 |

### 与旧 E 系列不完全一致的地方

| 旧 E 系列结论 | NE001 新观察 | 是否矛盾 | 当前解释 |
|---|---|---|---|
| 旧 raw 协议下 W4A8 一做 activation quantization 就严重崩坏。 | normalized 协议下 W4A8 final 只比 W4A32 低约 `0.336 dB`。 | 不是直接矛盾。 | 数据协议、幅值归一化、FP32 起点、eval 口径都变了；旧数值不能迁移。normalized per-patch absmax 极大缓解了 A8 activation range 压力。 |
| 旧 E004d 认为 W4A8 A8 init 崩坏主因是 Conv2d，Linear/transformer 不是主因。 | NE001 中 W4A4 final absolute MSE top 是 attention_qkv，Linear max MSE 高于 Conv2d。 | 不完全一致，但不是直接冲突。 | 旧 E004d 是 W4A8 raw 协议的 full-grid sensitivity；NE001 是 W4A4 normalized 协议的局部 diagnostics。bitwidth、数据协议、指标语义都不同。 |
| 旧 E004/E006 主线从 transformer 转向 Conv2d。 | NE001 显示 A4 下 transformer attention qkv/proj 重新变得重要。 | 是 A4-specific 新信号。 | A8 有 153+ effective levels 时 transformer 误差可接受；A4 只有约 14-16 levels 时 transformer 和融合边界会重新成为瓶颈。 |
| 旧 E005 tensor-wise range/clipping 基本无效。 | NE001 显示 A4 effective levels 很低，似乎 range 可能重要。 | 不能直接推翻旧结论。 | 低 effective levels 说明 A4 表达力不足，但不等于 tensor-wise clipping 能修复。必须先做 NE004 sensitivity，再对目标组做 NE005 range。 |
| 旧 E006c 最强候选是 selective split/merge/stage_output Conv2d 粒度。 | NE001 的 relative MSE 明确指向 stage_output/merge/split。 | 一致。 | 这条旧结论在新协议下仍有机制支持，值得优先复测，但目标应从 W4A8 扩展到 W4A4。 |

### 关键区别：diagnostics 不是 sensitivity

NE001 的 fake-quant MSE / relative MSE 是局部指标，它回答：

- 哪些层本地量化误差大？
- 哪些层相对自身信号最脆弱？
- A4 和 A8 的局部误差结构是否相似？

它不能直接回答：

- 关闭某组 activation quantizer 后 full-grid SNR 能恢复多少？
- 某层局部误差大是否一定导致最终输出差？
- 某个结构是否应该保留 FP32、改 per-channel，还是改 range？

因此 NE001 只能给 NE004 排序和分组优先级，不能替代 NE004 sensitivity。

### 对 NE004 的直接指导

NE004 不应只复刻旧 E004 的 `all Conv2d` / `all Linear` 二分实验。新 W4A4 主线需要同时覆盖 absolute MSE 和 relative MSE 两类热点。

建议 NE004 第一轮分组：

| 优先级 | group | 来源 | 要回答的问题 |
|---:|---|---|---|
| 1 | `attention_qkv` | W4A4 absolute MSE top | qkv 的巨大局部误差是否真的主导 full-grid A4 gap？ |
| 2 | `cnn conv_branch` | W4A4 absolute MSE top + 旧 E004 Conv2d 结论 | CNN conv 累积误差是否仍是 A4 的主要 SNR 损失来源？ |
| 3 | `stage_output_conv` | W4A4 relative MSE top + 旧 E006c 强候选 | stage 输出卷积是否解释 A4 结构性失真？ |
| 4 | `split_proj + merge_proj` | W4A4 relative MSE top + 旧 E006c 强候选 | 融合投影是否是 A4 下最值得做细粒度的结构？ |
| 5 | `stage5 attention_proj/mlp` | W4A4 relative MSE top + 旧 stage5 风险 | stage5 transformer 是否是 A4-specific 风险点，还是旧 E006c 中“独立 stage5 有害”的延续？ |
| 6 | `all Conv2d` / `all Linear` | 旧 E004 对照 | 用全局组验证 A4 是否仍满足“Conv2d 主导”假设。 |

NE004 判读规则：

- 如果 `attention_qkv` 关闭后 full-grid 恢复很小，说明 attention qkv 的 high absolute MSE 可能被后续结构吸收，主线回到 Conv2d/fusion/stage_output。
- 如果 `attention_qkv` 关闭后恢复很大，W4A4 需要 A4-specific transformer 策略，不能只沿用旧 E006 Conv2d selective granularity。
- 如果 `stage_output_conv` 或 `split+merge` 恢复大，旧 E006c 的 selective granularity 候选应直接迁移到 W4A4。
- 如果 `all Conv2d` 仍远强于细分组，说明 A4 可能存在更广泛的 Conv2d 累积误差，NE006 要考虑 all Conv2d per-channel 作为上限。

### 对 NE005 / NE006 的影响

NE005：

- 不应马上大范围做 percentile / MSE-grid range sweep。
- 只应在 NE004 证明敏感的组上做 range/clipping。
- 需要同时看 SNR 和 SSIM，因为 W4A4 reconstruction 已经表现出 SNR/SSIM tradeoff。

NE006：

- 旧 E006c 的 `split_proj + merge_proj + stage_output_conv` selective per-channel / g4 仍是优先候选。
- 但 W4A4 还应加入 `attention_qkv` 或 transformer qkv/proj 的 A4-specific 候选：
  - selective A8 fallback；
  - qkv-only higher precision；
  - qkv per-channel / group-wise；
  - qkv 保持 FP32 作为上限反事实。
- W4A8 不再作为主要优化对象，但每个 W4A4 策略都应至少验证不会明显破坏 W4A8。

### 当前对“冲突”的最终判断

NE001 和旧 E 系列不是简单互相否定，而是说明：

1. 旧 E 系列回答的是 raw-amplitude 旧协议下的 W4A8 崩坏机制。
2. NE001 回答的是 normalized 新协议下 W4A4 的 A4-specific 压力机制。
3. 旧结论中 “Conv2d / split / merge / stage_output 很重要” 仍然成立。
4. 旧结论中 “Linear/transformer 不是主因” 不能直接外推到 W4A4。
5. 后续实验必须以 full-grid sensitivity 验证 NE001 的局部诊断，不能只凭 diagnostics 排序决定最终策略。

## 2026-05-11 NE001 统计项横向对比与指标用途

本节补充 NE001 中此前没有展开的统计项。NE001 实际记录了大量 activation distribution / quantization error / structure grouping 统计，但不同统计项回答的问题不同，不能全部按同一权重解释。

### 指标分层

| 指标组 | 包含字段 | 回答的问题 | 后续主要服务 |
|---|---|---|---|
| 任务质量 | SNR、SSIM、pre/final gain | 模型最终是否可用，重建是否真的提升输出质量。 | 所有实验的最终验收。 |
| 合法性 | quant state、delta count、initialized count、non-positive delta | checkpoint 是否真的启用 activation quantization，scale 是否非法。 | NE002 sanity / state verification。 |
| 量化误差 | fake-quant MSE、relative MSE | 哪些层量化前后局部误差大，哪些层相对自身信号最脆弱。 | NE004 sensitivity 分组。 |
| 有效级别 | effective int levels | activation 实际利用了多少整数格点。 | 判断 A4 是否是 bitwidth 表达力瓶颈。 |
| 分布离群 | min/max/std、p99、p99.9、p99.99、absmax、absmax/p99 | range 是否被长尾或离群值撑大。 | NE005 range / clipping。 |
| 通道不均衡 | per-channel absmax ratio | 同一层内通道幅值差异是否大，tensor-wise scale 是否过粗。 | NE006 per-channel / group-wise granularity。 |
| 结构分组 | module type、stage、branch、role summary | 误差是否集中在 Conv2d、Linear、stage output、fusion、attention 等结构。 | NE004 分组、NE006 selective strategy。 |

因此，NE001 的正确读法不是“谁的数最大就直接修谁”，而是：

1. 先用 full-grid SNR/SSIM 判断现象是否真实重要；
2. 用 legality 排除状态错误；
3. 用 fake-quant MSE / relative MSE 形成 NE004 sensitivity 候选；
4. 用 outlier 和 per-channel ratio 分别决定 NE005 / NE006 是否值得做。

### 六个 diagnostics 对象的横向可比性

| 对象 | 可比较字段 | 不应比较字段 | 原因 |
|---|---|---|---|
| W4A4 pre/final | 全部 activation diagnostics 字段。 | N/A | 主对象，`act_quant=true` 且 52 个 activation quantizer 均初始化。 |
| W4A8 pre/final | 全部 activation diagnostics 字段。 | N/A | 成功参照，`act_quant=true` 且 52 个 activation quantizer 均初始化。 |
| W4A32 pre/final | activation distribution、结构分组、最终 SNR/SSIM。 | fake-quant MSE、effective levels、delta stats。 | W4A32 是 weight-only，`act_quant=false`，没有 activation delta。 |
| FP32 | 最终 SNR/SSIM、可视化上限。 | activation quantizer diagnostics。 | FP32 不是 quantized checkpoint，不存在 activation quantizer。 |

### 六对象统计摘要

| object | absmax/p99 max | top outlier layer | per-channel ratio max | top channel-imbalance layer | fake-quant MSE mean/max | relative MSE max | levels min/max |
|---|---:|---|---:|---|---:|---:|---:|
| W4A4 pre | `9.83285` | `model.stage5.1` | `2.68474` | `model.stage1.0.block.conv_branch.6` | `0.00612955 / 0.027745` | `0.0674264` | `13 / 246` |
| W4A4 final | `9.83285` | `model.stage5.1` | `2.68474` | `model.stage1.0.block.conv_branch.6` | `0.00468151 / 0.0464743` | `0.0546570` | `14 / 235` |
| W4A8 pre | `9.83285` | `model.stage5.1` | `2.68474` | `model.stage1.0.block.conv_branch.6` | `5.65246e-05 / 0.000505416` | `0.00160065` | `168 / 256` |
| W4A8 final | `9.83285` | `model.stage5.1` | `2.68474` | `model.stage1.0.block.conv_branch.6` | `4.43473e-05 / 0.000241724` | `0.00165058` | `153 / 256` |
| W4A32 pre | `9.08709` | `model.stage5.1` | `2.62299` | `model.stage1.0.block.conv_branch.6` | `None / None` | `None` | `None / None` |
| W4A32 final | `9.83285` | `model.stage5.1` | `2.68474` | `model.stage1.0.block.conv_branch.6` | `None / None` | `None` | `None / None` |

读法：

- `absmax/p99 max` 和 `per-channel ratio max` 在 W4A4/W4A8/W4A32 final 中基本一致，说明这些分布形态主要来自模型结构和输入分布，不是 A4 特有状态错误。
- W4A4 与 W4A8 的分布离群位置相同，但 fake-quant MSE 和 effective levels 差异巨大；这说明 A4 的主要问题不是“看到了不同分布”，而是同样分布在 A4 网格下表达不够。
- `model.stage5.1` 是最强 outlier 层，`model.stage1.0.block.conv_branch.6` 是最强 per-channel imbalance 层。它们不是最终 SNR 结论本身，但分别是 NE005 range 和 NE006 granularity 的重点候选。

### W4A4 pre -> final：reconstruction 改善了什么，又恶化了什么

| role | pre MSE mean | final MSE mean | MSE delta | pre relative MSE | final relative MSE | relative delta | final levels min |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention_proj | `0.000525719` | `0.000465102` | `-0.0000606` | `0.0266923` | `0.0240871` | `-0.002605` | `14` |
| attention_qkv | `0.0184496` | `0.0186708` | `+0.000221` | `0.0165272` | `0.0140978` | `-0.002429` | `16` |
| conv | `0.0135093` | `0.00851042` | `-0.004999` | `0.0364380` | `0.0221592` | `-0.014279` | `16` |
| merge_proj | `0.00165389` | `0.00134185` | `-0.000312` | `0.0464129` | `0.0388603` | `-0.007553` | `16` |
| mlp | `0.000360641` | `0.000415627` | `+0.0000550` | `0.0216472` | `0.0232609` | `+0.001614` | `16` |
| split_proj | `0.000727764` | `0.000734917` | `+0.00000715` | `0.0277123` | `0.0284331` | `+0.000721` | `16` |
| stage_output_conv | `0.00101980` | `0.000991273` | `-0.0000285` | `0.0345558` | `0.0342028` | `-0.000353` | `16` |

读法：

- W4A4 reconstruction 最大的正收益来自 `conv`：MSE mean 从 `0.0135093` 降到 `0.00851042`，relative MSE mean 从 `0.0364380` 降到 `0.0221592`。
- `merge_proj`、`attention_proj` 也有改善。
- `attention_qkv` 的 relative MSE 下降，但 absolute MSE 略升，并且 final 的 worst layer 变成 `stage3 attention_qkv`。这解释了为什么日志中说 reconstruction 像是“整体误差再分配”。
- `mlp` 和 `split_proj` 略有变差。虽然幅度不大，但它们提示后续不能只看 mean SNR，需要保留 role-level 表。

### W4A4 final vs W4A8 final：同结构下 A4 放大了多少误差

| role | A4 MSE mean | A8 MSE mean | A4/A8 MSE ratio | A4 relative mean | A8 relative mean | A4/A8 relative ratio |
|---|---:|---:|---:|---:|---:|---:|
| attention_proj | `0.000465102` | `3.72275e-06` | `124.9x` | `0.0240871` | `0.000202129` | `119.2x` |
| attention_qkv | `0.0186708` | `0.000122226` | `152.8x` | `0.0140978` | `0.000105402` | `133.8x` |
| conv | `0.00851042` | `0.0000895306` | `95.1x` | `0.0221592` | `0.000236041` | `93.9x` |
| merge_proj | `0.00134185` | `0.0000241400` | `55.6x` | `0.0388603` | `0.000666999` | `58.3x` |
| mlp | `0.000415627` | `0.00000483982` | `85.9x` | `0.0232609` | `0.000270922` | `85.9x` |
| split_proj | `0.000734917` | `0.0000107998` | `68.0x` | `0.0284331` | `0.000418082` | `68.0x` |
| stage_output_conv | `0.000991273` | `0.0000192701` | `51.4x` | `0.0342028` | `0.000855874` | `40.0x` |
| tail | `0.000599201` | `0.00000710523` | `84.3x` | `0.0355458` | `0.000421496` | `84.3x` |

读法：

- A4 并不是只在某一个结构上比 A8 差，而是几乎所有 role 都被放大几十到一百多倍。
- `attention_qkv` 的 A4/A8 MSE ratio 最高，说明 A4 的 transformer qkv 压力是新协议下必须单独验证的点。
- `stage_output_conv` 和 `merge_proj` 的 relative MSE 很高，即使 MSE 绝对值不如 qkv，也可能对结构输出质量更敏感。

### W4A4 final 的 stage 视角

| stage | count | MSE mean/max | relative MSE mean/max | absmax/p99 mean/max | per-channel ratio mean/max | levels min |
|---|---:|---:|---:|---:|---:|---:|
| head | `1` | `6.806e-06 / 6.806e-06` | `0.000778 / 0.000778` | `6.5907 / 6.5907` | `2.2708 / 2.2708` | `235` |
| stage1 | `10` | `0.004106 / 0.013639` | `0.027554 / 0.054657` | `5.3266 / 6.7599` | `1.6807 / 2.6847` | `16` |
| stage2 | `10` | `0.005423 / 0.022643` | `0.026152 / 0.043996` | `4.3956 / 6.7357` | `1.4086 / 2.0229` | `16` |
| stage3 | `10` | `0.008114 / 0.046474` | `0.024563 / 0.035186` | `4.4163 / 5.9881` | `1.4648 / 1.9141` | `16` |
| stage4 | `10` | `0.004042 / 0.010956` | `0.023266 / 0.041922` | `5.4073 / 7.7720` | `1.5144 / 2.0534` | `16` |
| stage5 | `10` | `0.002597 / 0.006698` | `0.024806 / 0.047078` | `6.1928 / 9.8329` | `1.6723 / 2.5266` | `14` |
| tail | `1` | `0.000599 / 0.000599` | `0.035546 / 0.035546` | `1.9711 / 1.9711` | `1.0000 / 1.0000` | `16` |

读法：

- stage3 的 absolute MSE 最大，主要由 `stage3 attention_qkv` 拉高。
- stage1 的 relative MSE max 最大，对应 `stage1.1 stage_output_conv`。
- stage5 的 outlier 和 effective levels 最差，包含 `stage5.1`、`stage5 attention_proj/mlp` 等风险点。
- 因此 “stage5 有信号” 和旧 E006c “stage5 独立细粒度有害” 不矛盾；它说明 stage5 需要作为 sanity check，而不是默认单独优化。

### outlier 与 per-channel imbalance 的具体候选

W4A4 final top per-channel imbalance：

| rank | layer | type | role | stage | ratio |
|---:|---|---|---|---|---:|
| 1 | `model.stage1.0.block.conv_branch.6` | Conv2d | conv | stage1 | `2.68474` |
| 2 | `model.stage5.1` | Conv2d | stage_output_conv | stage5 | `2.52657` |
| 3 | `model.stage5.0.block.conv_branch.6` | Conv2d | conv | stage5 | `2.50420` |
| 4 | `model.stage1.0.block.split_proj` | Conv2d | split_proj | stage1 | `2.41581` |
| 5 | `model.head` | Conv2d | head | head | `2.27078` |
| 6 | `model.stage5.0.block.merge_proj` | Conv2d | merge_proj | stage5 | `2.15788` |

W4A4 final top outlier：

| rank | layer | type | role | stage | absmax/p99 |
|---:|---|---|---|---|---:|
| 1 | `model.stage5.1` | Conv2d | stage_output_conv | stage5 | `9.83285` |
| 2 | `model.stage5.0.block.conv_branch.6` | Conv2d | conv | stage5 | `8.37609` |
| 3 | `model.stage4.0.block.conv_branch.3` | Conv2d | conv | stage4 | `7.77198` |
| 4 | `model.stage5.0.block.conv_branch.3` | Conv2d | conv | stage5 | `7.33216` |
| 5 | `model.stage4.0.block.conv_branch.6` | Conv2d | conv | stage4 | `7.15543` |
| 6 | `model.stage5.0.block.merge_proj` | Conv2d | merge_proj | stage5 | `7.12966` |

读法：

- per-channel imbalance 指向 NE006：这些层更可能从 per-channel / group-wise activation granularity 中获益。
- outlier 指向 NE005：这些层更值得做 percentile / MSE-grid / structured clipping。
- 两张表高度偏向 Conv2d、stage output、split/merge，而不是 qkv；这说明 qkv 的问题更像 bitwidth/absolute MSE，Conv/fusion/stage output 的问题更像 range/granularity。

### 这次完整对比对后续实验的具体影响

1. NE004 sensitivity 必须先验证 diagnostics 排名是否会转化为 full-grid SNR 恢复。
2. NE004 不能只做旧 E004 的 all Conv2d / all Linear，应加入：
   - `attention_qkv`
   - `cnn conv_branch`
   - `stage_output_conv`
   - `split_proj + merge_proj`
   - `stage5 attention_proj/mlp`
3. NE005 range/clipping 不应全模型乱扫，应优先对 top outlier 中的 stage5/stage4 Conv2d、stage_output、merge_proj 做。
4. NE006 granularity 应优先覆盖 per-channel imbalance top layers 所属结构：
   - conv_branch.6
   - stage_output_conv
   - split_proj / merge_proj
5. qkv 的 high absolute MSE 不一定能被 Conv2d per-channel 修复，因此 NE006 需要保留 qkv-only A8 fallback / higher precision / group-wise 的反事实候选。
6. W4A8 只作为护栏：若某个 W4A4 策略让 W4A8 明显变差，则不能作为通用默认策略。

### 结论更新

NE001 的完整统计并没有推翻前面的简化结论，而是把它细化成三条并行线索：

1. **bitwidth 线索**：A4 effective levels 只有约 `14-16`，导致所有结构的误差相对 A8 放大几十到一百多倍。
2. **range 线索**：stage5/stage4 Conv2d、stage_output、merge_proj 有明显 outlier，后续 NE005 只应优先处理这些目标组。
3. **granularity 线索**：conv_branch.6、stage_output_conv、split/merge 有明显 per-channel imbalance，后续 NE006 的 selective per-channel / g4 应从这些结构开始。

因此，进入下一部分实验前的优先级是：先做 NE004 W4A4 分组 sensitivity，验证这些局部统计是否真的对应 full-grid 恢复；再决定 NE005 range 和 NE006 granularity 的具体展开顺序。

## 2026-05-11 NE002 与 NE003 新计划记录

本节把 NE002 / NE003 的新语义固定下来，避免机械复刻旧 E002 / E003。旧 E002 的核心是修复 activation `delta` 负数；旧 E003 的核心是建立 128-sample 多样本评估口径。当前 NE 系列已经换成 normalized 协议，且 W4A8 已经可用、W4A4 成为主对象，因此 NE002 / NE003 应作为进入 NE004 前的状态与解释口径收口，而不是优化实验。

### NE002：合法状态、checkpoint reload 与 packed deployment sanity

实验目标：

- 排除 W4A4 / W4A8 结果由 checkpoint 状态错误、activation bitwidth 配错、未真正启用 activation quantization、reload 状态丢失或 packed restore 不一致造成的可能。
- 确认 NE004-NE006 可以把差异归因到 activation quantization 策略本身，而不是工程状态问题。
- 将 W4A4 低指标正式定性为真实 A4 activation bitwidth 压力，而不是合法性 bug。

实验对象：

| 类别 | 对象 | 用途 |
|---|---|---|
| 主对象 | `NE000_2 W4A4 pre_act_recon`、`NE000_2 W4A4 final` | 检查 A4 activation quantization 在重建前后的状态是否合法。 |
| 成功参照 | `NE000 W4A8 pre_act_recon`、`NE000 W4A8 final` | 确认 A8 好结果不是因为 activation quantization 没有真正启用。 |
| weight-only 参照 | `E007 W4A32 pre_recon`、`E007 W4A32 final` | 确认 W4A32 正确保持 `act_quant=false`，只作为 weight-only 对照。 |
| 部署参照 | W4A32 packed、W4A8 packed、W4A4 packed | 确认 packed 权重整数化和恢复后仍与 fake-quant checkpoint 对齐。 |
| 上限参照 | FP32 checkpoint | 只记录 full-grid 指标，不做 quantizer sanity。 |

实验方法：

1. 对 6 个 quantized checkpoint 统一运行 `verify_quantized_scrn`，输出 verification JSON。
2. 检查每个 checkpoint 的 `final_quant_state`、weight bit counts、activation bitwidth、activation quantizer count、initialized activation quantizer count、`non_positive_delta_count`、`level_offender_count`。
3. 对 W4A32 / W4A8 / W4A4 packed artifact 检查 `manifest.json` 和 `summary.json`，确认 `weights.bin`、`aux_fp32.bin`、activation qparams、bitwidth 和 final quant state 完整。
4. 复核 packed-vs-checkpoint full-grid equivalence：W4A32、W4A8、W4A4 的 packed restored mean SNR delta 应保持在 `< 0.01 dB` 量级。
5. 如现有结果仍不能充分证明 state toggle 正确，再补一个小规模 state toggle sanity：同一 checkpoint 分别强制 `act_quant=false/true`，确认输出会按预期切换。

验收标准：

| 对象 | 预期状态 |
|---|---|
| W4A4 pre/final | `weight_quant=true`、`act_quant=true`、activation bitwidth `4`、52 个 activation quantizer 初始化、`non_positive_delta_count=0`。 |
| W4A8 pre/final | `weight_quant=true`、`act_quant=true`、activation bitwidth `8`、52 个 activation quantizer 初始化、`non_positive_delta_count=0`。 |
| W4A32 pre/final | `weight_quant=true`、`act_quant=false`、无 activation delta，作为 weight-only 参照。 |
| packed W4A32/W4A8/W4A4 | packed restored 与原 checkpoint full-grid 指标对齐，mean SNR delta 绝对值 `< 0.01 dB`。 |

NE002 不做：

- 不做新的 activation reconstruction。
- 不做 range / clipping。
- 不做 granularity / mixed precision。
- 不以单样本 SNR 判断结果好坏。

NE002 完成后的判断：

- 若所有 sanity 均通过，进入 NE003 和 NE004。
- 若发现 bitwidth、quant state、reload 或 packed restore 问题，先修复状态问题并重新验证，不进入 NE004。
- 若只发现 packed 等价失败但 fake-quant checkpoint 合法，则 NE004 可继续使用 fake-quant checkpoint，但部署候选必须暂停。

### NE003：固定 full-grid 解释口径和代表图口径

实验目标：

- 固定后续 W4A4 优化实验的数值和可视化解释口径。
- 避免后续再次被单样本、显示尺度、反归一化方式或 source/condition 偶然性误导。
- 为 NE004-NE006 每个候选策略提供统一的可读对比模板。

实验对象：

| 对象 | 角色 |
|---|---|
| FP32 | normalized 协议上限。 |
| E007 W4A32 pre/final | weight-only 重建前后参照。 |
| NE000 W4A8 pre/final | 可用 activation quantization 成功参照。 |
| NE000_2 W4A4 pre/final | W4A4 主对象，重点看 activation reconstruction 前后差异。 |

实验方法：

1. 复核已有 normalized `478 x 25` full-grid 指标，统一记录 overall、by-source、by-SNR、by-missing-rate、by-condition。
2. 固定一组代表样本：
   - W4A4 final 最差样本；
   - W4A4 final 中位样本；
   - W4A4 final 较好样本；
   - W4A4 reconstruction 提升最大样本；
   - W4A4 reconstruction 变差或 SSIM 下降样本；
   - 默认单样本 sanity 图，仅作为历史可视化参照，不作为正式指标。
3. 使用一致显示规范：
   - seismic colormap；
   - normalized 样本使用反归一化或统一幅值尺度；
   - 同一图内 FP32 / W4A32 / W4A8 / W4A4 使用一致 `vmin/vmax`；
   - 同时输出误差图，误差图单独固定尺度。
4. 每个代表样本记录 source、patch index、SNR setting、missing rate、condition index、FP32/W4A32/W4A8/W4A4 SNR/SSIM。

验收标准：

- 代表样本清单可复现，记录 sample id / source / condition。
- 图像输出全部位于 `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE003_*` 下。
- 记录 W4A4 pre -> final 的 SNR/SSIM 变化，特别保留“W4A4 SNR 提升但 SSIM 下降”的案例。
- 后续 NE004-NE006 的每个候选策略都复用同一 full-grid 和代表图口径。

NE003 不做：

- 不改变 checkpoint。
- 不改变量化参数。
- 不做 sensitivity 或 range / granularity 搜索。

### 与旧 E002 / E003 的区别

| 阶段 | 旧 E 系列语义 | NE 系列新语义 |
|---|---|---|
| E002 / NE002 | 修复 activation `delta` 负数，并验证正 scale clamp 是否恢复 W4A8。 | 复核 W4A4/W4A8/W4A32 的 quant state、bitwidth、reload 和 packed deployment 等价。 |
| E003 / NE003 | 建立 128-sample eval，证明单样本 SNR 不可靠。 | 固定 normalized `478 x 25` full-grid 解释口径、代表样本和 seismic 反归一化可视化规范。 |

当前建议顺序：

1. 先执行 NE002，正式锁定 checkpoint 和 packed deployment 的合法性。
2. 再执行 NE003，固定代表图和 full-grid 解释口径。
3. 然后进入 NE004，以 W4A4 分组 sensitivity 验证 NE001 的局部诊断是否真的能转化为 full-grid 恢复。

## 2026-05-11 NE002 checkpoint / deployment sanity 结果

完成 NE002 合法状态与部署等价 sanity。NE002 不改变 checkpoint、不重新量化、不做优化，只确认当前 W4A4 / W4A8 / W4A32 结果是否可以作为后续 NE003 / NE004 的可靠输入。

执行环境：

- branch：`main`
- repo root：`/home/data1/hanwen/project/Project/SCRN_Quant`
- 初始 worktree：clean
- GPU：物理 GPU `1`
- 备注：普通沙箱内 `torch.cuda.is_available() == False`，实际 GPU 命令按同参数在沙箱外执行；GPU 2 当时有外部 `swinir` 进程，未使用。

### NE002 verification 产物

输出目录：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE002_checkpoint_deployment_sanity/verification`

| object | verification JSON | passed | weight_quant | act_quant | n_bits_a | weight bits | level offenders | activation delta / initialized |
|---|---|---:|---:|---:|---:|---|---:|---|
| W4A4 pre | `ne002_w4a4_pre_verification.json` | true | true | true | 4 | `{"4":50,"8":2}` | 0 | `52 / 52` |
| W4A4 final | `ne002_w4a4_final_verification.json` | true | true | true | 4 | `{"4":50,"8":2}` | 0 | `52 / 52` |
| W4A8 pre | `ne002_w4a8_pre_verification.json` | true | true | true | 8 | `{"4":50,"8":2}` | 0 | `52 / 52` |
| W4A8 final | `ne002_w4a8_final_verification.json` | true | true | true | 8 | `{"4":50,"8":2}` | 0 | `52 / 52` |
| W4A32 pre | `ne002_w4a32_pre_verification.json` | true | true | false | 8 | `{"4":50,"8":2}` | 0 | `0 / 0` |
| W4A32 final | `ne002_w4a32_final_verification.json` | true | true | false | 8 | `{"4":50,"8":2}` | 0 | `0 / 0` |

非正 activation delta 复核：

- `verify_quantized_scrn` 当前报告 activation delta 是否恢复，但不单独输出 `non_positive_delta_count` 字段。
- 复用 NE001 同 checkpoint diagnostics summary 交叉确认：
  - W4A4 pre：`non_positive_delta_count=0`
  - W4A4 final：`non_positive_delta_count=0`
  - W4A8 pre：`non_positive_delta_count=0`
  - W4A8 final：`non_positive_delta_count=0`
  - W4A32 pre/final：`act_quant=false`，activation delta 为 `0/0`，不适用。

判断：

- W4A4/W4A8 checkpoint 都是真正的 W4 activation + A4/A8 activation fake-quant checkpoint。
- W4A32 pre/final 都是合法 W4A32 weight-only checkpoint。
- 6 个 checkpoint 均无 weight level offender。

### NE002 state toggle smoke

输出目录：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE002_checkpoint_deployment_sanity/state_toggle`

设置：

- eval dataset：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- `num_eval_samples=16`
- `batch_size=16`
- `seed=20260507`
- GPU：物理 GPU `1`
- 说明：这只是 activation state toggle sanity，不作为正式 full-grid 指标。

| run | mode | selected quantizers | samples | FP32 SNR mean | quant SNR mean | quant SSIM mean | quant-FP32 SNR |
|---|---|---:|---:|---:|---:|---:|---:|
| `20260511_125156_ne002_w4a4_final_all_on` | all_on | 51 | 16 | 16.4840 | 12.0236 | 0.930772 | -4.4604 |
| `20260511_125203_ne002_w4a4_final_all_off` | all_off | 51 | 16 | 16.4840 | 16.4281 | 0.953357 | -0.0559 |
| `20260511_125210_ne002_w4a8_final_all_on` | all_on | 51 | 16 | 16.4840 | 16.1103 | 0.951832 | -0.3737 |
| `20260511_125216_ne002_w4a8_final_all_off` | all_off | 51 | 16 | 16.4840 | 16.4281 | 0.953357 | -0.0559 |

判断：

- `all_on` / `all_off` 都选中 51 个候选 activation quantizer，符合默认排除 output quantizer 的预期。
- W4A4 与 W4A8 的 `all_on` 和 `all_off` 输出明显不同，说明 activation quantizer 开关确实生效。
- W4A4 `all_on` 明显低于 W4A8 `all_on`，且二者 `all_off` 回到相同的 W4A32-like 行为；这支持 W4A4 问题来自 activation A4，而不是 weight checkpoint 或 eval 流程。

### NE002 packed deployment equivalence 复核

已复核的 packed export 目录：

- W4A32：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a32_packed/e007_normalized_w4a32_single_gpu1_final`
- W4A8：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a8_packed/ne000_normalized_w4a8_tensor_a5000_final`
- W4A4：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a4_packed/ne000_2_normalized_w4a4_tensor_a5000_final`

三组 packed artifact 均包含：

- `weights.bin`
- `aux_fp32.bin`
- `manifest.json`
- `summary.json`

| object | weights.bin | aux_fp32.bin | manifest.json | raw payload | total export | quantized layers | activation quantizers |
|---|---:|---:|---:|---:|---:|---:|---:|
| W4A32 packed | 213632 | 40860 | 94254 | 254492 | 348746 | 52 | 0 |
| W4A8 packed | 213632 | 41276 | 121554 | 254908 | 376462 | 52 | 52 |
| W4A4 packed | 213632 | 41276 | 121590 | 254908 | 376498 | 52 | 52 |

Packed full-grid equivalence：

| object | eval rows | packed runtime state | checkpoint SNR mean | packed SNR mean | packed-checkpoint SNR | checkpoint SSIM mean | packed SSIM mean | pred MSE mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| W4A32 | 11950 | `weight_quant=false, act_quant=false` | 17.785582 | 17.785577 | -0.000005 | 0.964137 | 0.964137 | 3.95e-10 |
| W4A8 | 11950 | `weight_quant=false, act_quant=true` | 17.449507 | 17.449534 | +0.000027 | 0.962868 | 0.962869 | 3.76e-07 |
| W4A4 | 11950 | `weight_quant=false, act_quant=true` | 12.914963 | 12.915036 | +0.000073 | 0.939563 | 0.939565 | 3.89e-06 |

判断：

- 三个 packed restored 模型的 mean SNR delta 绝对值都远小于 `0.01 dB`，通过部署等价阈值。
- W4A8 / W4A4 packed restore 使用 `weight_quant=false, act_quant=true`，语义正确：权重已经恢复为部署量化值，activation fake quant 仍开启。
- W4A4 packed 部署等价通过，因此 W4A4 可作为合法 A4 压力对照继续进入后续实验。

### NE002 最终结论

- W4A4 低指标不是 checkpoint reload 错误、bitwidth 配错、activation quantizer 未启用、负 scale 或 packed restore 不一致导致的。
- W4A8 好结果也不是因为 activation quantization 没有真正开启；W4A8 final 是合法 W4A8 activation quantization baseline。
- E007 W4A32 pre/final 是合法 weight-only 参照。
- W4A32 / W4A8 / W4A4 packed deployment 均与对应 fake-quant checkpoint full-grid 对齐。
- NE002 sanity 通过，可以进入 NE003 固定代表图口径，并随后进入 NE004 W4A4 分组 sensitivity。

## 2026-05-11 NE003 fixed grid visual protocol 结果

完成 NE003：固定后续 W4A4 activation quantization 优化实验的 full-grid 数值解释与代表图口径。本轮只生成统一可视化和摘要，不改 checkpoint、不重新量化、不做 sensitivity。

新增 CLI 与测试：

- `SCRN_BRECQ_app/scrn_brecq/cli/visualize_quantized_scrn_grid.py`
- `SCRN_BRECQ_app/scrn_brecq/tests/test_visualize_quantized_scrn_grid.py`

验证：

- unit test：`5 tests OK`
- `py_compile`：通过
- CLI `--help`：通过

运行产物：

- run dir：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE003_fixed_grid_visual_protocol/20260511_140941_ne003_w4a4_w4a8_w4a32_seismic_denorm_seed20260507`
- GPU：物理 GPU `1`；普通沙箱内 CUDA 不可用，按同参数升级执行。
- 输出：`config.json`、`metrics_summary.json`、`selected_samples.json`、`summary.md`、`figures/*.png`
- 图像数量：`16`
  - normalized representative：`15`
  - default single sanity：`1`

### NE003 固定显示协议

- 正式数值指标继续使用 normalized `478 x 25` grid。
- normalized 样本显示使用 manifest 反归一化：`display = normalized * normalization_scale`。
- 图像 colormap 固定为 `seismic`，并以 0 为中心对称显示。
- 每张正式代表图包含：
  - Clean、Input、FP32、W4A32 final、W4A8 final、W4A4 final；
  - W4A32 pre/final、W4A8 pre/final、W4A4 pre/final；
  - Input / FP32 / W4A32 final / W4A8 final / W4A4 final error map。
- prediction panel 使用同图统一尺度；error panel 使用独立统一误差尺度。
- 默认单样本 `SCRN-main/test_data` 只保留历史 sanity 作用，不纳入 normalized full-grid 结论。

### NE003 核心 full-grid 对比

| object | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| FP32 | 17.832885 | 18.174198 | 0.964330 | 0.978794 |
| E007 W4A32 pre | 16.614250 | 17.177490 | 0.935462 | 0.949638 |
| E007 W4A32 final | 17.785582 | 18.112757 | 0.964137 | 0.978461 |
| NE000 W4A8 pre | 17.372734 | 17.785509 | 0.962674 | 0.977030 |
| NE000 W4A8 final | 17.449507 | 17.877689 | 0.962868 | 0.977292 |
| NE000_2 W4A4 pre | 11.172733 | 11.157722 | 0.941492 | 0.955772 |
| NE000_2 W4A4 final | 12.914963 | 13.118019 | 0.939563 | 0.954078 |

重建前后变化：

| object | SNR mean delta | SNR median delta | SSIM mean delta | SSIM median delta |
|---|---:|---:|---:|---:|
| E007 W4A32 | +1.171332 | +0.805456 | +0.028675 | +0.023764 |
| NE000 W4A8 | +0.076773 | +0.006535 | +0.000195 | +0.000069 |
| NE000_2 W4A4 | +1.742231 | +1.861192 | -0.001929 | -0.002785 |

关键解释：

- W4A8 final 已经很接近 W4A32 / FP32，后续继续提高 W4A8 的空间有限；它更适合作为“A8 成功参照”。
- W4A4 final 仍有明显 SNR gap，但 activation reconstruction 对 W4A4 的 SNR 恢复幅度大于 W4A8，说明 A4 仍有可优化空间。
- W4A4 的 SSIM 平均略降，不能只按 SNR 选择策略。后续 NE004-NE006 必须同时报告 SNR、SSIM 和 NE003 error map。
- Kerry3D 的 W4A4 代表样本出现 reconstruction 变差，是后续 sensitivity 中优先排查的 source / condition。

### NE003 代表图集合

| # | selection | source | patch | condition | W4A4 final SNR | W4A4 SNR gain |
|---:|---|---|---|---:|---:|---:|
| 1 | E012 continuity low | Anisotropic | `test_000044.npy` | 4 | 13.516 | +3.153 |
| 2 | E012 continuity mid | Anisotropic | `test_000013.npy` | 12 | 14.195 | +3.404 |
| 3 | E012 continuity high | Anisotropic | `test_000025.npy` | 20 | 16.486 | +3.086 |
| 4 | E012 continuity low | Kerry3D | `test_000081.npy` | 4 | 5.825 | +0.076 |
| 5 | E012 continuity mid | Kerry3D | `test_000084.npy` | 12 | 10.817 | +0.181 |
| 6 | E012 continuity high | Kerry3D | `test_000077.npy` | 20 | 7.928 | -0.254 |
| 7 | E012 continuity low | Shots0001 | `test_000374.npy` | 4 | 10.554 | +2.087 |
| 8 | E012 continuity mid | Shots0001 | `test_000395.npy` | 12 | 13.752 | +2.063 |
| 9 | E012 continuity high | Shots0001 | `test_000349.npy` | 20 | 16.642 | +1.028 |
| 10 | W4A4 worst final SNR | Anisotropic | `test_000034.npy` | 18 | -1.504 | -0.153 |
| 11 | W4A4 best final SNR | Shots0001 | `test_000334.npy` | 20 | 18.571 | +1.803 |
| 12 | W4A4 median final SNR | Shots0001 | `test_000372.npy` | 8 | 13.118 | +2.394 |
| 13 | W4A4 max recon gain | Anisotropic | `test_000037.npy` | 22 | 16.447 | +3.912 |
| 14 | W4A4 worst recon change | Kerry3D | `test_000091.npy` | 20 | 13.302 | -1.363 |
| 15 | W4A4 max SSIM drop | Kerry3D | `test_000086.npy` | 22 | 12.714 | -1.107 |
| 16 | default single sanity | `SCRN-main/test_data` | `clear.npy` | -1 | 13.270 | +0.097 |

NE003 对后续实验的约束：

- 后续 W4A4 策略候选必须优先用 normalized `478 x 25` 指标判断，不使用默认单样本做结论。
- 后续每个重要候选都应复用 NE003 代表图集合，特别检查：
  - W4A4 worst final SNR 是否被修复；
  - max recon gain 是否继续保持；
  - Kerry3D worst recon change / max SSIM drop 是否改善；
  - W4A8 / W4A32 参照是否保持稳定。
- NE004 应以 W4A4 sensitivity 为主线；W4A8 只作为成功参照，不作为主要优化对象。

## 2026-05-11 NE002 / NE003 整理版与 NE004-NE006 后续计划

本节把 NE002、NE003 压缩为后续实验判断所需的核心事实，并明确 NE004、NE005、NE006 的实验内容、方法、目标和对象。当前主对象为 `NE000_2 W4A4`；`NE000 W4A8` 是成功参照，`E007 W4A32` 和 FP32 是上下限参照。

### 已固定的实验事实

| 类别 | 已确认结论 | 影响 |
|---|---|---|
| 合法性 | W4A4/W4A8/W4A32 pre/final verification 全部通过。 | 后续不再把主问题归因于 checkpoint 损坏或 reload 错误。 |
| activation 状态 | W4A4/W4A8 均启用 activation quantization；W4A32 保持 `act_quant=false`。 | W4A4 低指标是真实 A4 activation 压力。 |
| state toggle | W4A4/W4A8 `all_on` 和 `all_off` 指标明显不同，`all_off` 回到 W4A32-like。 | activation quantizer 开关可信，可以做 NE004 sensitivity。 |
| packed 等价 | W4A32/W4A8/W4A4 packed restored 与 fake-quant checkpoint 的 mean SNR delta 都小于 `0.01 dB`。 | 三个结果都可作为部署等价已通过的参照。 |
| full-grid 口径 | NE003 固定 normalized `478 x 25` 为正式判断口径。 | 后续任何策略都必须报告同一 grid。 |
| 可视化口径 | NE003 固定 `seismic`、manifest 反归一化、15 个 normalized representative + 1 个 default sanity。 | 后续策略图像必须可横向比较。 |

核心指标：

| object | SNR mean / median | SSIM mean / median | 定位 |
|---|---:|---:|---|
| FP32 | 17.832885 / 18.174198 | 0.964330 / 0.978794 | 上限。 |
| E007 W4A32 final | 17.785582 / 18.112757 | 0.964137 / 0.978461 | weight-only 近上限参照。 |
| NE000 W4A8 final | 17.449507 / 17.877689 | 0.962868 / 0.977292 | A8 成功参照。 |
| NE000_2 W4A4 final | 12.914963 / 13.118019 | 0.939563 / 0.954078 | 后续主优化对象。 |

重建前后变化：

| object | SNR mean delta | SSIM mean delta | 结论 |
|---|---:|---:|---|
| E007 W4A32 | +1.171332 | +0.028675 | weight reconstruction 稳定有效。 |
| NE000 W4A8 | +0.076773 | +0.000195 | A8 init 已经足够好，重建只小幅微调。 |
| NE000_2 W4A4 | +1.742231 | -0.001929 | A4 重建显著恢复 SNR，但 SSIM 略降。 |

后续解读原则：

- W4A8 不再是主优化对象；它用于判断某个现象是否 A4 特有。
- W4A4 是合法且 packed 可恢复的 A4 baseline，但与 W4A32 final 的 mean SNR gap 仍约 `4.87 dB`。
- W4A4 后续优化不能只看 SNR；SSIM、by-source 和 NE003 error map 都必须同步看。
- Kerry3D 的 W4A4 reconstruction 变差和 SSIM drop 样本是后续定位的重点。

### NE004：W4A4 activation quantizer 分组 sensitivity

实验目标：

- 定位 W4A4 gap 主要来自哪些 activation quantizer 组。
- 验证旧 E004 “Conv2d activation 多层累积误差主导”在新 normalized 数据和 A4 bitwidth 下是否仍成立。
- 把 NE001 的局部统计热点转化为 full-grid 行为证据。

实验对象：

| 对象 | 用途 |
|---|---|
| `NE000_2 W4A4 final` | 主诊断对象。 |
| `NE000_2 W4A4 pre` | 可选对照，判断 reconstruction 是否改变热点。 |
| `NE000 W4A8 final` | 成功参照，验证同一分组在 A8 下是否影响较小。 |
| `E007 W4A32 final` / FP32 | 固定参照，不参与 activation sensitivity。 |

实验方法：

1. 固定 normalized `478 x 25` grid、seed `20260507` 和 NE003 指标口径。
2. 对 W4A4 final 依次测试：
   - `all_on`
   - `all_off`
   - `conv2d_off`
   - `linear_off`
   - `stage_output_conv_off`
   - `split_proj_off`
   - `merge_proj_off`
   - `split_proj + merge_proj + stage_output_conv_off`
   - stage1/stage2/stage3/stage4/stage5 off
   - attention projection / FFN / output quantizer 等 role-wise off，如工具支持。
3. 对 W4A8 final 跑精简同构参照：`all_on`、`all_off`、`conv2d_off`、`split/merge/stage_output_conv_off`。
4. 每个 run 记录 `11950` rows、overall、by-source、by-condition、相对 W4A4 all_on 的 gain。
5. 对 top 2-3 个 sensitivity 结果复用 NE003 代表图输出。

验收标准：

- 明确给出 W4A4 主要瓶颈组，判断是 Conv2d 主导、Linear 主导、stage 主导，还是结构化组合主导。
- 若关闭某组显著恢复 W4A4 且 SSIM 不下降，该组进入 NE006 finer granularity。
- 若某组只改善单一 source 或伤害 SSIM，则只作为诊断证据，不直接作为部署方向。

预期产物：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE004_w4a4_activation_sensitivity/...`
- 分组 sensitivity 表、by-source gain 表、NE003 同口径代表图。

### NE005：W4A4 activation range / clipping / calibration 实验

实验目标：

- 判断 W4A4 gap 是否来自 tensor-wise range 估计、离群值、clipping 或 calibration 分布。
- 复核旧 E005 “range/clipping 对 W4A8 帮助有限”在 A4 下是否仍成立；A4 bitwidth 更低，旧结论不能直接照搬。

实验对象：

| 对象 | 用途 |
|---|---|
| `E007 W4A32 final` | 所有 W4A4 range 变体共同起点。 |
| `NE000_2 W4A4 final` | tensor-wise A4 baseline。 |
| `NE000 W4A8 final` | 判断改动是否只对 A4 有意义。 |

实验方法：

1. 先确认 `activation_only_quantize_scrn` 当前支持哪些 `activation_range_method` / clipping 参数；如不支持目标策略，单独小补丁实现并测试。
2. 保持 NE000_2 其它变量不变：
   - `n_bits_w=4`
   - `n_bits_a=4`
   - `activation_granularity=tensor`
   - `num_samples=1024`
   - `batch_size=16`
   - `init_batch_size=64`
   - `iters_a=5000`
   - `activation_lr=0.0004`
   - `lp_norm=2.4`
3. 只改变 range / clipping / calibration 变量，候选包括：
   - percentile clipping：p99、p99.5、p99.9、p99.99；
   - MSE grid range init；
   - absmax / no-clipping baseline 复跑；
   - NE004 敏感组 selective clipping；
   - 可选 per-source / balanced calibration 对照。
4. 每个候选都评估 pre-act 和 final，记录 reconstruction 前后变化。
5. 对有希望的候选复用 NE003 代表图，重点看 Kerry3D worst recon change 和 max SSIM drop。

验收标准：

- 候选必须在 normalized `478 x 25` 上超过 NE000_2 W4A4 final。
- 判断维度包括：
  - W4A4 final mean/median SNR；
  - SSIM mean/median；
  - by-source，尤其 Kerry3D 是否改善；
  - activation delta / bitwidth / verification 是否合法。
- 若收益有限，正式排除“简单调 range/clipping 解决 A4”的路径，进入 NE006。

预期产物：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping/...`
- range variant 表、pre/final 指标、相对 NE000_2 gain、是否值得 packed 的判断。

### NE006：W4A4 structured granularity / selective fine granularity

实验目标：

- 基于 NE004 定位出的敏感组测试 finer activation granularity。
- 寻找 W4A4 的质量上限候选和部署友好候选。
- 复核旧 E006 “selective fine granularity 优于 all Conv2d fine granularity”的机制在 normalized A4 下是否成立。

实验对象：

| 对象 | 用途 |
|---|---|
| `E007 W4A32 final` | 所有 W4A4 granularity 变体共同起点。 |
| `NE000_2 W4A4 final` | tensor-wise A4 baseline。 |
| `NE000 W4A8 final` | 成功参照。 |

实验方法：

1. 保持 NE000_2 reconstruction 设置，主要改变 activation granularity 和选择组。
2. 候选至少包括：
   - all Conv2d activation per-channel；
   - all Conv2d activation group-wise g4；
   - `split_proj + merge_proj + stage_output_conv` per-channel；
   - `split_proj + merge_proj + stage_output_conv` g4；
   - `stage_output_conv` g4；
   - stage5 单独 finer granularity / stage5 保持 tensor-wise 对照；
   - Linear / transformer sanity。
3. 每个候选从 E007 W4A32 final 出发重跑 activation init + reconstruction。
4. 每个候选做 normalized full-grid eval，top candidates 生成 NE003 代表图。
5. 最终候选必须做 verification 和 packed deployment equivalence。

验收标准：

- W4A4 final 明显缩小与 W4A32 final 的 gap，且 SSIM 不恶化。
- per-channel 可作为质量上限；g4 或 selective g4 若接近 per-channel 且 packed 等价通过，优先作为部署候选。
- 若 all Conv2d 不如 selective，则保留旧 E006 的机制链条。
- 若 stage5 finer granularity 仍有害，明确记录为禁用方向。

预期产物：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_structured_granularity/...`
- granularity strategy 表、top candidates 代表图、verification、packed equivalence。

### NE004-NE006 执行顺序

1. NE004 先定位“该改哪里”。
2. NE005 再判断是否能用低成本 range/clipping 修复。
3. NE006 最后做 structured granularity，寻找真正 W4A4 候选。

阶段性成功标准：

- 短期：W4A4 final mean SNR 明显高于 `12.914963`，SSIM 不再继续下降。
- 中期：找到 full-grid 稳定优于 tensor-wise W4A4 的 selective strategy。
- 长期：给出清晰机制链条：normalized 数据下 A8 为什么成功、A4 gap 来自哪里、什么结构策略能恢复 A4。

## 2026-05-11 NE004 grid sensitivity evaluator 实现记录

为 NE004 新增 normalized fixed-grid activation sensitivity evaluator。旧 `evaluate_activation_sensitivity.py` 仍是旧 multi-sample degradation 口径；NE004 正式结论必须使用 NE003 固定的 normalized `478 x 25` grid，因此新增 grid 版工具。

代码与接口：

- 新增 `SCRN_BRECQ_app/scrn_brecq/cli/evaluate_activation_sensitivity_grid.py`
  - fixed grid 默认：SNR `-2,-1,1,5,10`，missing rate `0.02,0.08,0.18,0.28,0.38`，seed `20260507`。
  - 输出 `per_sample_metrics.jsonl`、`metrics.json`、`config.json`、`summary.md`、`selected_quantizers.csv`。
  - 聚合维度继承 grid evaluator：overall、by-source、by-SNR、by-missing-rate、by-condition。
- 扩展 `activation_sensitivity.py`
  - 新增 plural OR selector：`--stages`、`--branches`、`--roles`、`--module-types`。
  - singular selector 保留；不同 selector 字段之间仍为 AND。
  - 默认继续排除 output quantizer，除非显式 `--include-output-quantizer`。

验证：

- TDD RED：
  - plural selector 缺失时报 `unexpected keyword argument 'roles'`；
  - grid CLI 缺失时报 `ModuleNotFoundError`。
- selector tests：`7 tests OK`
- grid evaluator tests：`5 tests OK`
- `py_compile`：通过
- CLI `--help`：通过

Smoke run：

| item | value |
|---|---|
| checkpoint | `NE000_2 W4A4 final` |
| mode | `disable_group --module-types Conv2d` |
| subset | `2 patches x 1 condition` |
| GPU | physical GPU `1` |
| run dir | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE004_w4a4_activation_sensitivity/smoke/20260511_190920_ne004_smoke_w4a4_conv2d_off_2patch_1cond` |
| selected quantizers | 31 |
| rows | 2 |
| quant SNR mean / median | 19.1097 / 19.1097 |
| quant SSIM mean | 0.973638 |

该 smoke 只验证工具链可用，不作为 NE004 结论。下一步使用同一 CLI 跑 W4A4 full-grid sensitivity matrix 和 W4A8 sanity subset。

## 2026-05-11 NE004 W4A4 activation sensitivity 结果记录

NE004 已完成 normalized `478 x 25 = 11950` full-grid activation sensitivity。正式执行使用物理 GPU `1`；普通沙箱内 CUDA 不可见，所有 full-grid run 均按同一参数升级执行。GPU 2 当时有外部 `swinir` 进程，未参与。

输出根目录：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE004_w4a4_activation_sensitivity/eval`

验收结果：

- W4A4 full matrix：17 个 run 全部完成。
- W4A8 sanity subset：4 个 run 全部完成。
- 每个 run 都有 `per_sample_metrics.jsonl`、`metrics.json`、`config.json`、`summary.md`、`selected_quantizers.csv`。
- 每个 run 都有 `11950` rows。
- selected quantizer count 与模型结构一致：
  - all on/off：`51`
  - Conv2d：`31`
  - Linear / transformer：`20`
  - `split_proj + merge_proj + stage_output_conv`：`15`
  - `stage_output_conv`：`5`
  - 单 stage：`10`

主要 run dirs：

- W4A4 all_on：`20260511_191550_ne004a_w4a4_all_on_grid478_seed20260507`
- W4A4 all_off：`20260511_191826_ne004b_w4a4_all_off_grid478_seed20260507`
- W4A4 Conv2d off：`20260511_192011_ne004c_w4a4_conv2d_off_grid478_seed20260507`
- W4A4 Linear off：`20260511_192222_ne004d_w4a4_linear_off_grid478_seed20260507`
- W4A4 split/merge/stage-output off：`20260511_193421_ne004i_w4a4_split_merge_stage_output_off_grid478_seed20260507`
- W4A4 stage5 off：`20260511_195435_ne004m5_w4a4_stage5_off_grid478_seed20260507`
- W4A8 all_on：`20260511_200058_ne004n_w4a8_all_on_grid478_seed20260507`
- W4A8 all_off：`20260511_200344_ne004o_w4a8_all_off_grid478_seed20260507`
- W4A8 Conv2d off：`20260511_200535_ne004p_w4a8_conv2d_off_grid478_seed20260507`
- W4A8 split/merge/stage-output off：`20260511_200753_ne004q_w4a8_split_merge_stage_output_off_grid478_seed20260507`

W4A4 full-grid sensitivity，相对 gain 均以 W4A4 all_on 为基准：

| group | selected | SNR mean | SNR median | SSIM mean | SNR gain |
|---|---:|---:|---:|---:|---:|
| all_on | 51 | 12.9150 | 13.1180 | 0.939563 | +0.0000 |
| all_off | 51 | 17.7856 | 18.1128 | 0.964137 | +4.8706 |
| Conv2d off | 31 | 17.1117 | 17.6774 | 0.960019 | +4.1968 |
| Linear off | 20 | 13.2336 | 13.4659 | 0.940903 | +0.3187 |
| cnn branch off | 15 | 13.4094 | 13.6834 | 0.943983 | +0.4944 |
| fusion branch off | 10 | 13.5998 | 13.8398 | 0.949349 | +0.6848 |
| transformer branch off | 20 | 13.2336 | 13.4659 | 0.940903 | +0.3187 |
| stage_output_conv off | 5 | 13.8209 | 14.0836 | 0.945819 | +0.9060 |
| split_proj+merge_proj+stage_output_conv off | 15 | 14.8177 | 15.0830 | 0.955504 | +1.9027 |
| conv role off | 15 | 13.4094 | 13.6834 | 0.943983 | +0.4944 |
| attention_qkv+attention_proj off | 10 | 12.9862 | 13.1711 | 0.940272 | +0.0712 |
| mlp off | 10 | 13.1516 | 13.3905 | 0.940516 | +0.2366 |
| stage1 off | 10 | 13.7005 | 13.9380 | 0.944190 | +0.7855 |
| stage2 off | 10 | 12.8254 | 13.0275 | 0.940593 | -0.0895 |
| stage3 off | 10 | 13.0420 | 13.2555 | 0.940532 | +0.1270 |
| stage4 off | 10 | 13.7307 | 14.0081 | 0.943316 | +0.8157 |
| stage5 off | 10 | 14.1329 | 14.4166 | 0.951537 | +1.2179 |

W4A8 sanity subset，相对 gain 均以 W4A8 all_on 为基准：

| group | selected | SNR mean | SNR median | SSIM mean | SNR gain |
|---|---:|---:|---:|---:|---:|
| W4A8 all_on | 51 | 17.4495 | 17.8777 | 0.962868 | +0.0000 |
| W4A8 all_off | 51 | 17.7856 | 18.1128 | 0.964137 | +0.3361 |
| W4A8 Conv2d off | 31 | 17.7775 | 18.1059 | 0.964084 | +0.3279 |
| W4A8 split/merge/stage-output off | 15 | 17.6917 | 18.0479 | 0.963734 | +0.2422 |

By-source 重点对比，相对 gain 均以同 source 的 W4A4 all_on 为基准：

| group | Anisotropic mean / gain | Kerry3D mean / gain | Shots0001 mean / gain |
|---|---:|---:|---:|
| W4A4 all_on | 13.2802 / +0.0000 | 9.0332 / +0.0000 | 13.0047 / +0.0000 |
| W4A4 all_off | 21.9718 / +8.6915 | 9.6082 / +0.5750 | 17.3124 / +4.3077 |
| W4A4 Conv2d off | 20.4664 / +7.1862 | 9.5707 / +0.5375 | 16.7734 / +3.7687 |
| W4A4 split/merge/stage-output off | 16.3668 / +3.0866 | 9.4215 / +0.3883 | 14.7406 / +1.7359 |
| W4A4 stage_output_conv off | 14.8241 / +1.5439 | 9.2018 / +0.1686 | 13.8175 / +0.8128 |
| W4A4 stage5 off | 15.1255 / +1.8453 | 9.3795 / +0.3463 | 14.1370 / +1.1324 |

NE004 结论：

- W4A4 主 gap 来自 activation quantization：`all_off` 回到 E007 W4A32 final，说明权重量化状态正常。
- Conv2d activation 是主因：Conv2d off 追回 `+4.1968 dB`，接近 all_off 的 `+4.8706 dB`；Linear / transformer off 只有 `+0.3187 dB`。
- 最强 selective 小集合是 `split_proj + merge_proj + stage_output_conv`：15 个 quantizer 追回 `+1.9027 dB`，明显强于普通 cnn/fusion/transformer branch。
- `stage_output_conv` 单独 5 个 quantizer 有 `+0.9060 dB`；stage5 单独关闭有 `+1.2179 dB`，stage1/stage4 有中等收益，stage2 单独关闭略负。
- Attention / MLP 不是当前主线：attention 仅 `+0.0712 dB`，MLP 仅 `+0.2366 dB`。
- W4A8 的方向一致但幅度小：W4A8 Conv2d off 追回 `+0.3279 dB`，几乎等于 all_off 的 `+0.3361 dB`。这说明 W4A8 好结果不是 activation quantization 没开，而是 A8 residual gap 本来很小。

与旧 E 系列的关系：

- 旧 E004 的机制判断在 NE004 中仍成立：Conv2d activation 主导，Linear / transformer 不是主瓶颈。
- 旧 E006C 的 selective 方向仍成立：`split_proj + merge_proj + stage_output_conv` 是高价值结构组。
- 新协议下 W4A8 不再崩坏；A4 才把 activation sensitivity 放大出来。因此 NE 系列主线应从“修 W4A8 崩坏”调整为“恢复 W4A4 的 Conv2d activation loss”。
- `stage5 off` 有收益不等于 stage5 finer granularity 一定有效；旧 E006C 曾显示 stage5 独立细粒度可能有害，NE006 需要用 granularity 实验单独确认。

后续建议：

- NE005 继续做 range / clipping sanity，用来排除低成本 range 路线，但不应期待它单独解决全部 A4 gap。
- NE006 的优先候选应是：
  - all Conv2d per-channel / g4；
  - `split_proj + merge_proj + stage_output_conv` per-channel / g4；
  - `stage_output_conv` g4；
  - stage5 单独 finer granularity 与 stage5 保持 tensor-wise 对照。
- 所有后续结论继续使用 normalized `478 x 25` grid，不使用旧 128-sample sensitivity 口径作为正式指标。

## 2026-05-11 NE004 全量结果二次整理

本节把 NE004 所有正式 run 按统一解释口径重新整理。`all_on` 是量化模型原始状态；`all_off` 是同一个 W4A4/W4A8 checkpoint 关闭 activation quantizer 后的 weight-only 参照；`disable_group` 表示只关闭指定 activation quantizer 组，其相对 `all_on` 的 gain 表示该组 activation quantization 对当前损失的贡献上限。sensitivity disable 只用于定位误差来源，不等价于最终部署策略。

W4A4 overall 全量表：

| group | selected | SNR mean | SNR median | SSIM mean | SSIM median | SNR gain | SSIM gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_on | 51 | 12.9150 | 13.1180 | 0.939563 | 0.954078 | +0.0000 | +0.000000 |
| all_off | 51 | 17.7856 | 18.1128 | 0.964137 | 0.978461 | +4.8706 | +0.024574 |
| Conv2d off | 31 | 17.1117 | 17.6774 | 0.960019 | 0.974592 | +4.1968 | +0.020456 |
| Linear off | 20 | 13.2336 | 13.4659 | 0.940903 | 0.954589 | +0.3187 | +0.001340 |
| cnn branch off | 15 | 13.4094 | 13.6834 | 0.943983 | 0.958733 | +0.4944 | +0.004420 |
| fusion branch off | 10 | 13.5998 | 13.8398 | 0.949349 | 0.964306 | +0.6848 | +0.009786 |
| transformer branch off | 20 | 13.2336 | 13.4659 | 0.940903 | 0.954589 | +0.3187 | +0.001340 |
| stage_output_conv off | 5 | 13.8209 | 14.0836 | 0.945819 | 0.960161 | +0.9060 | +0.006256 |
| split+merge+stage_output off | 15 | 14.8177 | 15.0830 | 0.955504 | 0.970028 | +1.9027 | +0.015941 |
| conv role off | 15 | 13.4094 | 13.6834 | 0.943983 | 0.958733 | +0.4944 | +0.004420 |
| attention off | 10 | 12.9862 | 13.1711 | 0.940272 | 0.954257 | +0.0712 | +0.000709 |
| mlp off | 10 | 13.1516 | 13.3905 | 0.940516 | 0.955035 | +0.2366 | +0.000953 |
| stage1 off | 10 | 13.7005 | 13.9380 | 0.944190 | 0.958810 | +0.7855 | +0.004627 |
| stage2 off | 10 | 12.8254 | 13.0275 | 0.940593 | 0.954710 | -0.0895 | +0.001030 |
| stage3 off | 10 | 13.0420 | 13.2555 | 0.940532 | 0.954761 | +0.1270 | +0.000969 |
| stage4 off | 10 | 13.7307 | 14.0081 | 0.943316 | 0.957741 | +0.8157 | +0.003753 |
| stage5 off | 10 | 14.1329 | 14.4166 | 0.951537 | 0.966062 | +1.2179 | +0.011974 |

W4A4 gain 排名和占 all_off gap 比例：

| group | SNR gain | all_off gap share |
|---|---:|---:|
| all_off | +4.8706 | 100.0% |
| Conv2d off | +4.1968 | 86.2% |
| split+merge+stage_output off | +1.9027 | 39.1% |
| stage5 off | +1.2179 | 25.0% |
| stage_output_conv off | +0.9060 | 18.6% |
| stage4 off | +0.8157 | 16.7% |
| stage1 off | +0.7855 | 16.1% |
| fusion branch off | +0.6848 | 14.1% |
| conv role off | +0.4944 | 10.2% |
| cnn branch off | +0.4944 | 10.2% |
| transformer branch off | +0.3187 | 6.5% |
| Linear off | +0.3187 | 6.5% |
| mlp off | +0.2366 | 4.9% |
| stage3 off | +0.1270 | 2.6% |
| attention off | +0.0712 | 1.5% |
| stage2 off | -0.0895 | -1.8% |

W4A4 by-source 全量表，单元格为 `SNR mean / gain`：

| group | Anisotropic | Kerry3D | Shots0001 |
|---|---:|---:|---:|
| all_on | 13.2802 / +0.0000 | 9.0332 / +0.0000 | 13.0047 / +0.0000 |
| all_off | 21.9718 / +8.6915 | 9.6082 / +0.5750 | 17.3124 / +4.3077 |
| Conv2d off | 20.4664 / +7.1862 | 9.5707 / +0.5375 | 16.7734 / +3.7687 |
| Linear off | 13.7033 / +0.4231 | 9.1653 / +0.1321 | 13.3108 / +0.3062 |
| cnn branch off | 13.9821 / +0.7019 | 9.1507 / +0.1175 | 13.4744 / +0.4698 |
| fusion branch off | 14.3097 / +1.0295 | 9.2607 / +0.2275 | 13.6416 / +0.6369 |
| transformer branch off | 13.7033 / +0.4231 | 9.1653 / +0.1321 | 13.3108 / +0.3062 |
| stage_output_conv off | 14.8241 / +1.5439 | 9.2018 / +0.1686 | 13.8175 / +0.8128 |
| split+merge+stage_output off | 16.3668 / +3.0866 | 9.4215 / +0.3883 | 14.7406 / +1.7359 |
| conv role off | 13.9821 / +0.7019 | 9.1507 / +0.1175 | 13.4744 / +0.4698 |
| attention off | 13.3643 / +0.0841 | 9.1376 / +0.1044 | 13.0720 / +0.0673 |
| mlp off | 13.6062 / +0.3260 | 9.0553 / +0.0221 | 13.2328 / +0.2282 |
| stage1 off | 14.5999 / +1.3197 | 9.2133 / +0.1801 | 13.7117 / +0.7071 |
| stage2 off | 13.0996 / -0.1806 | 9.0359 / +0.0027 | 12.9290 / -0.0757 |
| stage3 off | 13.3865 / +0.1062 | 9.0349 / +0.0018 | 13.1409 / +0.1362 |
| stage4 off | 14.6582 / +1.3780 | 9.1442 / +0.1110 | 13.7406 / +0.7359 |
| stage5 off | 15.1255 / +1.8453 | 9.3795 / +0.3463 | 14.1370 / +1.1324 |

W4A8 sanity 全量表：

| group | selected | SNR mean | SNR median | SSIM mean | SSIM median | SNR gain |
|---|---:|---:|---:|---:|---:|---:|
| W4A8 all_on | 51 | 17.4495 | 17.8777 | 0.962868 | 0.977292 | +0.0000 |
| W4A8 all_off | 51 | 17.7856 | 18.1128 | 0.964137 | 0.978461 | +0.3361 |
| W4A8 Conv2d off | 31 | 17.7775 | 18.1059 | 0.964084 | 0.978429 | +0.3279 |
| W4A8 split+merge+stage_output off | 15 | 17.6917 | 18.0479 | 0.963734 | 0.978095 | +0.2422 |

按输入 SNR setting 的关键趋势，括号为相对 W4A4 all_on 的 gain：

| group | -2 dB | -1 dB | 1 dB | 5 dB | 10 dB |
|---|---:|---:|---:|---:|---:|
| all_on | 11.6505 / +0.0000 | 12.0574 / +0.0000 | 12.7252 / +0.0000 | 13.6971 / +0.0000 | 14.4447 / +0.0000 |
| Conv2d off | 14.9635 / +3.3130 | 15.5722 / +3.5148 | 16.6447 / +3.9195 | 18.4045 / +4.7074 | 19.9737 / +5.5291 |
| split+merge+stage_output off | 13.1685 / +1.5180 | 13.6601 / +1.6026 | 14.5119 / +1.7867 | 15.8227 / +2.1256 | 16.9252 / +2.4806 |
| stage5 off | 12.6297 / +0.9793 | 13.1003 / +1.0428 | 13.8837 / +1.1585 | 15.0560 / +1.3589 | 15.9948 / +1.5502 |
| all_off | 15.3934 / +3.7430 | 16.0500 / +3.9926 | 17.2188 / +4.4936 | 19.2096 / +5.5125 | 21.0561 / +6.6115 |

按 missing rate 的关键趋势，括号为相对 W4A4 all_on 的 gain：

| group | 0.02 | 0.08 | 0.18 | 0.28 | 0.38 |
|---|---:|---:|---:|---:|---:|
| all_on | 13.9245 / +0.0000 | 13.6206 / +0.0000 | 13.0601 / +0.0000 | 12.4028 / +0.0000 | 11.5668 / +0.0000 |
| Conv2d off | 18.1624 / +4.2378 | 17.8429 / +4.2223 | 17.2784 / +4.2183 | 16.5907 / +4.1879 | 15.6843 / +4.1175 |
| split+merge+stage_output off | 15.8166 / +1.8920 | 15.5298 / +1.9092 | 14.9793 / +1.9192 | 14.3164 / +1.9136 | 13.4464 / +1.8796 |
| stage5 off | 15.2414 / +1.3169 | 14.8965 / +1.2760 | 14.2799 / +1.2198 | 13.5663 / +1.1635 | 12.6803 / +1.1135 |
| all_off | 18.8276 / +4.9030 | 18.4873 / +4.8667 | 17.9192 / +4.8590 | 17.2641 / +4.8613 | 16.4298 / +4.8630 |

二次分析：

- 模块类型层面，Conv2d off 追回 all_off gap 的 `86.2%`，而 Linear/transformer 只有 `6.5%`。A4 激活误差不是平均分布在所有 quantizer 上，而是高度集中在 Conv2d activation。
- 结构角色层面，`split_proj + merge_proj + stage_output_conv` 追回 `39.1%` 的 all_off gap，是最强的部署可操作小集合。它同时提升 Anisotropic、Kerry3D、Shots0001，没有只对单一 source 有效。
- source 层面，Anisotropic 对 A4 activation 最敏感：all_off 可追回 `+8.6915 dB`，Conv2d off 可追回 `+7.1862 dB`；Shots0001 次之；Kerry3D 的绝对 SNR 低且 activation-off gain 小，后续优化不应只用 Kerry3D 判断策略成败。
- 输入 SNR setting 层面，Conv2d / selective 组在高输入 SNR 条件下 gain 更大，说明 activation quantization 更限制“本来模型能恢复得更好”的样本。
- missing rate 层面，Conv2d 和 split/merge/stage-output 的 gain 在 `0.02` 到 `0.38` 基本稳定，说明 NE004 结论不是由某个缺失率条件偶然驱动。
- stage 层面，stage5 off 的收益最高，但 stage2 off 轻微负收益。该现象提示 activation sensitivity 存在跨层补偿，不能把 disable_group 结果机械外推成“该 stage finer granularity 一定更好”。
- W4A8 sanity 与 W4A4 方向一致但幅度小：W4A8 Conv2d off 追回 `0.3279 / 0.3361 = 97.6%` 的 A8 activation gap。A8 成功不是状态错误，而是 normalized 数据 + 8bit activation 已经把 Conv2d 误差压到较小。

对后续实验的直接约束：

- NE005 range/clipping 可以做，但如果收益不接近 `+1 dB`，应尽快收束，不要在全局 range 参数上长时间扫。
- NE006 的主线应以 Conv2d granularity 为中心，优先顺序为：
  1. `split_proj + merge_proj + stage_output_conv` g4 / per-channel；
  2. all Conv2d g4 / per-channel 上限参考；
  3. `stage_output_conv` g4；
  4. stage5 独立策略作为风险对照；
  5. Linear / attention / MLP 只保留 sanity，不应作为主优化方向。
- 后续报告 W4A4 改进时，必须同时报告 overall、by-source、by-input-SNR、by-missing-rate，避免一个策略只改善单一条件却被误判为通用提升。

## 2026-05-11 关于最终输出 activation quantizer 禁用的说明

当前 W+A 量化实现中，`act_quant=true` 表示中间层激活量化整体开启；但模型最后一个 `QuantModule` 的 activation quantizer 会通过 `disable_network_output_quantization()` 被禁用。也就是说，结构上每个 `QuantModule` 都有 activation quantizer 状态，W4A4/W4A8 checkpoint 中可以看到 52 个 activation quantizer 的 `delta/zero_point`；但实际前向部署状态下，最后网络输出层的 activation quantizer 被 bypass。因此更准确的表述是：中间激活参与量化，最终预测输出不再额外量化。

这样设计的原因：

- SCRN 的最终输出已经是恢复后的单通道地震 patch，不会继续作为下一层输入。对最终输出再做一次 4bit/8bit fake quantization，主要影响的是保存/评估结果本身，而不是后续低精度计算链路。
- SNR/SSIM 和可视化比较都直接基于最终预测 patch。如果最终输出也被量化，指标会混入一层额外的输出网格截断误差，难以区分中间激活量化误差和最终结果格式误差。
- BRECQ/PTQ 的核心目标是让量化网络内部计算尽量接近 FP32 路径；保留最终输出为浮点，更利于和 FP32、W4A32、W4A8/W4A4 之间做公平质量比较。

注意事项：

- 如果未来目标硬件要求最终输出也必须以 INT4/INT8 张量形式落盘或传给后处理算子，那么当前评估会略偏乐观，需要单独增加“output quantized”部署口径。
- checkpoint 中仍保存最后一个 activation quantizer 的状态，容易让人误以为 52 个激活量化器全部实际生效。后续报告应明确区分“结构上存在 52 个 quantizer”和“实际启用约 51 个，中间层启用、最终输出禁用”。
- 对比不同实验时必须保持该策略一致。若某个实验打开最终输出量化，而另一个实验关闭，就不能直接比较 SNR/SSIM。

当前结论：最终输出 activation quantizer 禁用不是状态错误，而是有意的评估和部署口径选择。除非后续明确研究“最终输出也低 bit 存储/传输”的部署场景，否则 W4A4/W4A8 主线应继续保持中间激活量化、最终输出不量化。

## 2026-05-12 NE005 W4A4 range / clipping sanity 结果

目标：验证 normalized 新协议下，`NE000_2 W4A4` 的 A4 activation gap 是否能通过 tensor-wise range / clipping 低成本修复。NE005 不改 granularity、不做 sensitivity、不做 packed，只比较 activation range 初始化策略及其后续 activation reconstruction 的 full-grid 结果。

执行环境与固定设置：

- 代码改动：无。
- GPU：物理 GPU `1`，执行前 GPU 1 空闲；GPU 2 有外部 python 进程占用约 `436 MiB`，未使用。
- 起点 checkpoint：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- Calibration：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- Eval：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- 统一参数：`n_bits_w=4`，`n_bits_a=4`，`activation_granularity=tensor`，`num_samples=1024`，`batch_size=16`，`init_batch_size=64`，`iters_a=5000`，`activation_lr=0.0004`，`lp_norm=2.4`。
- 正式评估口径：normalized `478 x 25` grid，`11950` rows，seed `20260507`。
- 对照：NE000_2 W4A4 final 为 `12.9150 / 13.1180` SNR mean/median，SSIM mean `0.939563`；E007 W4A32 final 为 `17.7856 / 18.1128` SNR mean/median，SSIM mean `0.964137`。

Run 目录：

| id | range method / selector | selected range quantizers | quant run | eval run |
|---|---|---:|---|---|
| NE005a | max / all activation candidates | 51 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_205121_ne005a_w4a4_range_max_tensor_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_001617_ne005a_w4a4_range_max_grid478_seed20260507` |
| NE005b | percentile 99.9 / all activation candidates | 51 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_211446_ne005b_w4a4_percentile_p999_tensor_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_002037_ne005b_w4a4_percentile_p999_grid478_seed20260507` |
| NE005c | percentile 99.99 / all activation candidates | 51 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_213752_ne005c_w4a4_percentile_p9999_tensor_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_002456_ne005c_w4a4_percentile_p9999_grid478_seed20260507` |
| NE005d | mse_grid / all activation candidates | 51 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_220200_ne005d_w4a4_mse_grid_tensor_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_002916_ne005d_w4a4_mse_grid_grid478_seed20260507` |
| NE005e | mse_grid / Conv2d only | 31 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_224146_ne005e_w4a4_mse_grid_conv2d_only_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_003336_ne005e_w4a4_mse_grid_conv2d_only_grid478_seed20260507` |
| NE005f | mse_grid / split_proj + merge_proj + stage_output_conv | 15 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_230441_ne005f_w4a4_mse_grid_split_merge_stage_output_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_003756_ne005f_w4a4_mse_grid_split_merge_stage_output_grid478_seed20260507` |
| NE005g | percentile 99.9 / split_proj + merge_proj + stage_output_conv | 15 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_232726_ne005g_w4a4_percentile_p999_split_merge_stage_output_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_004215_ne005g_w4a4_percentile_p999_split_merge_stage_output_grid478_seed20260507` |

合法性验证：

- 7 个 final checkpoint 均 `passed=true`。
- 7 个 final checkpoint 均为 `weight_quant=true, act_quant=true`。
- activation bitwidth 均为 `4`，`activation_delta_count=52`，`initialized_activation_quantizers=52`。
- weight bit counts 均保持 `4bit=50, 8bit=2`。
- `non_positive_delta_count=0`，`level_offender_count=0`。
- 7 个 grid eval 均生成 `11950` rows，并包含 `metrics.json`、`config.json`、`summary.md`、`per_sample_metrics.jsonl`。

总体结果，正式数值来自 normalized `478 x 25` grid：

| variant | selected | rows | pre SNR/SSIM mean | final SNR/SSIM mean | final SNR/SSIM median | recon gain SNR/SSIM | final vs NE000_2 W4A4 | remaining SNR gap to E007 W4A32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NE005a max | 51 | 11950 | 7.8501 / 0.748581 | 12.9330 / 0.938922 | 13.1377 / 0.953237 | +5.0829 / +0.190341 | +0.0180 / -0.000641 | -4.8526 |
| NE005b p99.9 | 51 | 11950 | 10.5794 / 0.925247 | 12.2617 / 0.936723 | 12.4995 / 0.950616 | +1.6824 / +0.011476 | -0.6532 / -0.002840 | -5.5238 |
| NE005c p99.99 | 51 | 11950 | 11.8503 / 0.892493 | 12.8394 / 0.937749 | 13.0574 / 0.951726 | +0.9892 / +0.045256 | -0.0755 / -0.001814 | -4.9462 |
| NE005d mse_grid all | 51 | 11950 | 12.1405 / 0.900725 | 12.8898 / 0.939433 | 13.0843 / 0.954021 | +0.7492 / +0.038708 | -0.0252 / -0.000130 | -4.8958 |
| NE005e mse_grid Conv2d | 31 | 11950 | 12.0340 / 0.901563 | 12.8932 / 0.939358 | 13.0856 / 0.953876 | +0.8592 / +0.037795 | -0.0217 / -0.000205 | -4.8923 |
| NE005f mse_grid selective | 15 | 11950 | 11.0522 / 0.909328 | 12.8956 / 0.939338 | 13.0950 / 0.953766 | +1.8434 / +0.030010 | -0.0194 / -0.000225 | -4.8900 |
| NE005g p99.9 selective | 15 | 11950 | 11.1966 / 0.931744 | 12.8730 / 0.939889 | 13.0629 / 0.954297 | +1.6764 / +0.008145 | -0.0420 / +0.000326 | -4.9126 |

重点 source 对比，单元格为 `final SNR mean / 相对 NE000_2 W4A4 gain`：

| variant | Anisotropic | Kerry3D | Shots0001 | min source gain | max source gain |
|---|---:|---:|---:|---:|---:|
| NE005a max | 13.3243 / +0.0441 | 9.0339 / +0.0008 | 13.0184 / +0.0137 | +0.0008 | +0.0441 |
| NE005b p99.9 | 12.6868 / -0.5934 | 8.8181 / -0.2150 | 12.3217 / -0.6829 | -0.6829 | -0.2150 |
| NE005c p99.99 | 13.2607 / -0.0195 | 8.9967 / -0.0365 | 12.9166 / -0.0880 | -0.0880 | -0.0195 |
| NE005d mse_grid all | 13.2573 / -0.0229 | 9.0292 / -0.0040 | 12.9781 / -0.0265 | -0.0265 | -0.0040 |
| NE005e mse_grid Conv2d | 13.2633 / -0.0169 | 9.0363 / +0.0032 | 12.9810 / -0.0237 | -0.0237 | +0.0032 |
| NE005f mse_grid selective | 13.2668 / -0.0134 | 9.0315 / -0.0017 | 12.9834 / -0.0213 | -0.0213 | -0.0017 |
| NE005g p99.9 selective | 13.2485 / -0.0317 | 9.0577 / +0.0246 | 12.9580 / -0.0467 | -0.0467 | +0.0246 |

最佳 SNR 变体 NE005a 的 by-input-SNR 稳定性：

| input SNR setting | NE005a final SNR | gain vs NE000_2 W4A4 |
|---:|---:|---:|
| -2 | 11.6748 | +0.0243 |
| -1 | 12.0769 | +0.0194 |
| 1 | 12.7424 | +0.0172 |
| 5 | 13.7116 | +0.0145 |
| 10 | 14.4594 | +0.0147 |

最佳 SNR 变体 NE005a 的 by-missing-rate 稳定性：

| missing rate | NE005a final SNR | gain vs NE000_2 W4A4 |
|---:|---:|---:|
| 0.02 | 13.9624 | +0.0378 |
| 0.08 | 13.6566 | +0.0360 |
| 0.18 | 13.0809 | +0.0207 |
| 0.28 | 12.4077 | +0.0049 |
| 0.38 | 11.5574 | -0.0094 |

关键解释：

- 最佳 SNR 变体是 NE005a `max`，但相对 NE000_2 W4A4 final 只有 `+0.0180 dB`，远低于 NE005 预设的 `+0.5 dB` 最低继续投入阈值。
- 最佳 SSIM mean 变体是 NE005g selective p99.9，为 `0.939889`，只比 NE000_2 W4A4 final 高 `+0.000326`。
- p99.9 全局 clipping 明显有害：SNR mean 从 `12.9150` 掉到 `12.2617`。
- mse_grid 能明显改变 pre-act-recon：全量 mse_grid 的 pre SNR mean 达到 `12.1405`，比 NE000_2 原始 pre SNR mean `11.1727` 高约 `+0.9678 dB`；但经过 activation reconstruction 后 final 反而略低于 NE000_2 final。这说明 range 初始化可以改变起点，但当前 activation reconstruction 会把不同 range 初始化收敛到接近且不优于原 baseline 的区域。
- Conv2d-only 和 selective mse_grid 没有带来收益：NE005e `12.8932`，NE005f `12.8956`，均低于 NE000_2 final。这与 NE004 “Conv2d 是误差主因”不冲突；NE004 说明关掉这些 activation quantizer 能恢复大量 gap，NE005 则说明仅在 tensor-wise A4 下重选 range 不能恢复该 gap。

NE005 结论：

- range / clipping 路线收益有限，应收束。最佳 SNR 改进只有 `+0.0180 dB`，不满足 `+0.5 dB` 的继续投入标准。
- NE004 的主因判断仍成立：A4 gap 主要来自 Conv2d activation，尤其 selective 小集合；但 NE005 排除了“在 tensor-wise A4 下只调 range 就能修复”的低成本路径。
- 后续应优先进入 NE006 granularity：先做 `split_proj + merge_proj + stage_output_conv` 的 g4 / per-channel，再做 all Conv2d g4 / per-channel 上限参考，stage_output_conv g4 与 stage5 独立作为对照。
- 如果未来需要组合策略，range 只适合作为 granularity 之后的小幅辅助项，而不是主线。

### NE005 完整结果补充与高可读结论

如何读 NE005 指标：

- `pre SNR/SSIM`：只做 activation range 初始化、尚未做 activation reconstruction 的 W4A4 结果。它反映 range 初始化质量，但不能单独代表最终模型。
- `final SNR/SSIM`：activation reconstruction 后的正式 W4A4 结果，后续决策只看这个字段。
- `recon gain`：final 相对 pre 的变化。gain 大不代表最终好，只说明重建从差起点追回更多。
- `final vs NE000_2 W4A4`：本实验最重要字段，表示 range/clipping 是否真的超过原始 W4A4 baseline。
- `gap to E007 W4A32`：距离 weight-only 上限还差多少。NE005 的所有变体都仍有约 `4.85 dB` 以上 SNR gap。

含 NE000_2 baseline 的总表：

| variant | range | selected | pre SNR mean/median | final SNR mean/median | final SSIM mean/median | recon gain SNR/SSIM | final vs NE000_2 | gap to W4A32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NE000_2 baseline | none baseline | 51 | 11.1727 / 11.1577 | 12.9150 / 13.1180 | 0.939563 / 0.954078 | +1.7422 / -0.001929 | +0.0000 / +0.000000 | -4.8706 / -0.024574 |
| NE005a max | max all | 51 | 7.8501 / 8.2980 | 12.9330 / 13.1377 | 0.938922 / 0.953237 | +5.0829 / +0.190341 | +0.0180 / -0.000641 | -4.8526 / -0.025215 |
| NE005b p99.9 | p99.9 all | 51 | 10.5794 / 10.7291 | 12.2617 / 12.4995 | 0.936723 / 0.950616 | +1.6824 / +0.011476 | -0.6532 / -0.002840 | -5.5238 / -0.027414 |
| NE005c p99.99 | p99.99 all | 51 | 11.8503 / 12.3437 | 12.8394 / 13.0574 | 0.937749 / 0.951726 | +0.9892 / +0.045256 | -0.0755 / -0.001814 | -4.9462 / -0.026388 |
| NE005d mse all | mse all | 51 | 12.1405 / 12.6383 | 12.8898 / 13.0843 | 0.939433 / 0.954021 | +0.7492 / +0.038708 | -0.0252 / -0.000130 | -4.8958 / -0.024704 |
| NE005e mse Conv2d | mse Conv2d | 31 | 12.0340 / 12.4996 | 12.8932 / 13.0856 | 0.939358 / 0.953876 | +0.8592 / +0.037795 | -0.0217 / -0.000205 | -4.8923 / -0.024779 |
| NE005f mse selective | mse split+merge+stage_output | 15 | 11.0522 / 11.2294 | 12.8956 / 13.0950 | 0.939338 / 0.953766 | +1.8434 / +0.030010 | -0.0194 / -0.000225 | -4.8900 / -0.024799 |
| NE005g p99.9 selective | p99.9 split+merge+stage_output | 15 | 11.1966 / 11.2490 | 12.8730 / 13.0629 | 0.939889 / 0.954297 | +1.6764 / +0.008145 | -0.0420 / +0.000326 | -4.9126 / -0.024248 |

运行时间与 single-sample sanity。该表只用于解释运行成本和单样本误导风险，正式判断仍以上表 full-grid 为准：

| variant | init s | recon s | elapsed min | single pre SNR/SSIM | single final SNR/SSIM | single recon gain |
|---|---:|---:|---:|---:|---:|---:|
| NE005a max | 25.2 | 1369.4 | 23.32 | 10.2431 / 0.792973 | 13.2465 / 0.902703 | +3.0033 / +0.109730 |
| NE005b p99.9 | 25.5 | 1351.8 | 23.04 | 12.3065 / 0.845850 | 13.1131 / 0.882638 | +0.8066 / +0.036788 |
| NE005c p99.99 | 25.2 | 1414.9 | 24.05 | 12.6385 / 0.877608 | 13.2652 / 0.904508 | +0.6267 / +0.026900 |
| NE005d mse all | 25.4 | 2352.1 | 39.69 | 12.7619 / 0.884633 | 13.2842 / 0.900688 | +0.5223 / +0.016055 |
| NE005e mse Conv2d | 25.6 | 1340.6 | 22.84 | 12.7369 / 0.885862 | 13.2878 / 0.900895 | +0.5509 / +0.015034 |
| NE005f mse selective | 25.3 | 1331.2 | 22.67 | 12.8111 / 0.883907 | 13.2766 / 0.902313 | +0.4655 / +0.018405 |
| NE005g p99.9 selective | 25.4 | 1328.5 | 22.62 | 12.5781 / 0.858694 | 13.2749 / 0.903449 | +0.6967 / +0.044754 |

single-sample 与 full-grid 的差异很重要：单样本上 NE005e/NE005d/NE005f 看起来都比 NE005a 好，但 full-grid 上只有 NE005a 在 SNR mean 上略微超过 baseline，而且幅度只有 `+0.0180 dB`。因此 NE 系列后续仍必须坚持 `478 x 25` grid，不能用单样本选择策略。

by-source 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`。当前 normalized test manifest 只有 `Anisotropic`、`Kerry3D`、`Shots0001` 三个 source：

| variant | Anisotropic | Kerry3D | Shots0001 |
|---|---:|---:|---:|
| NE000_2 baseline | 13.28 / +0.00 | 9.03 / +0.00 | 13.00 / +0.00 |
| NE005a max | 13.32 / +0.04 | 9.03 / +0.00 | 13.02 / +0.01 |
| NE005b p99.9 | 12.69 / -0.59 | 8.82 / -0.22 | 12.32 / -0.68 |
| NE005c p99.99 | 13.26 / -0.02 | 9.00 / -0.04 | 12.92 / -0.09 |
| NE005d mse all | 13.26 / -0.02 | 9.03 / -0.00 | 12.98 / -0.03 |
| NE005e mse Conv2d | 13.26 / -0.02 | 9.04 / +0.00 | 12.98 / -0.02 |
| NE005f mse selective | 13.27 / -0.01 | 9.03 / -0.00 | 12.98 / -0.02 |
| NE005g p99.9 selective | 13.25 / -0.03 | 9.06 / +0.02 | 12.96 / -0.05 |

by-source 结论：没有任何变体在三个 source 上形成稳定有效提升。NE005a 的提升主要是 Anisotropic 上 `+0.04 dB` 的极小改动；NE005g 的 SSIM mean 略高，但 SNR 在 Anisotropic 和 Shots0001 都下降。range/clipping 没有解决 NE004 暴露的 source-sensitive activation gap。

by-input-SNR 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | -2 dB | -1 dB | 1 dB | 5 dB | 10 dB |
|---|---:|---:|---:|---:|---:|
| NE000_2 baseline | 11.65 / +0.00 | 12.06 / +0.00 | 12.73 / +0.00 | 13.70 / +0.00 | 14.44 / +0.00 |
| NE005a max | 11.67 / +0.02 | 12.08 / +0.02 | 12.74 / +0.02 | 13.71 / +0.01 | 14.46 / +0.01 |
| NE005b p99.9 | 11.12 / -0.53 | 11.50 / -0.56 | 12.11 / -0.61 | 12.98 / -0.72 | 13.61 / -0.84 |
| NE005c p99.99 | 11.60 / -0.06 | 12.00 / -0.06 | 12.66 / -0.06 | 13.61 / -0.08 | 14.33 / -0.11 |
| NE005d mse all | 11.64 / -0.01 | 12.04 / -0.02 | 12.70 / -0.02 | 13.66 / -0.03 | 14.41 / -0.04 |
| NE005e mse Conv2d | 11.64 / -0.01 | 12.04 / -0.02 | 12.70 / -0.02 | 13.67 / -0.03 | 14.41 / -0.03 |
| NE005f mse selective | 11.64 / -0.01 | 12.04 / -0.01 | 12.71 / -0.02 | 13.67 / -0.03 | 14.42 / -0.03 |
| NE005g p99.9 selective | 11.63 / -0.02 | 12.02 / -0.03 | 12.69 / -0.04 | 13.64 / -0.05 | 14.39 / -0.06 |

by-input-SNR 结论：NE005a 的微小收益在各 SNR setting 上都只有 `+0.01` 到 `+0.02 dB`，没有实质意义；p99.9 全局 clipping 对高输入 SNR 样本伤害更大，`10 dB` 条件下下降 `-0.84 dB`，说明强 clipping 会破坏本来恢复潜力更高的样本。

by-missing-rate 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | 0.02 | 0.08 | 0.18 | 0.28 | 0.38 |
|---|---:|---:|---:|---:|---:|
| NE000_2 baseline | 13.92 / +0.00 | 13.62 / +0.00 | 13.06 / +0.00 | 12.40 / +0.00 | 11.57 / +0.00 |
| NE005a max | 13.96 / +0.04 | 13.66 / +0.04 | 13.08 / +0.02 | 12.41 / +0.00 | 11.56 / -0.01 |
| NE005b p99.9 | 12.99 / -0.93 | 12.80 / -0.82 | 12.41 / -0.65 | 11.90 / -0.50 | 11.21 / -0.36 |
| NE005c p99.99 | 13.82 / -0.11 | 13.53 / -0.10 | 12.98 / -0.08 | 12.34 / -0.06 | 11.53 / -0.04 |
| NE005d mse all | 13.92 / -0.01 | 13.61 / -0.01 | 13.04 / -0.02 | 12.37 / -0.04 | 11.52 / -0.05 |
| NE005e mse Conv2d | 13.92 / -0.01 | 13.61 / -0.01 | 13.04 / -0.02 | 12.37 / -0.03 | 11.53 / -0.04 |
| NE005f mse selective | 13.92 / -0.01 | 13.61 / -0.01 | 13.04 / -0.02 | 12.37 / -0.03 | 11.53 / -0.04 |
| NE005g p99.9 selective | 13.90 / -0.02 | 13.59 / -0.03 | 13.02 / -0.04 | 12.35 / -0.05 | 11.51 / -0.06 |

by-missing-rate 结论：NE005a 的微小收益主要集中在低 missing rate 条件，`0.38` 时已经变成 `-0.01 dB`；p99.9 全局 clipping 对低 missing rate 伤害最大，`0.02` 时下降 `-0.93 dB`。这再次说明 clipping 不是稳定修复策略。

按可读性整理的最终判断：

1. NE005 没有找到有效 range/clipping 修复。最佳 SNR 只提升 `+0.0180 dB`，最佳 SSIM 只提升 `+0.000326`，都属于实验噪声级别。
2. range 初始化会改变 pre-act 结果，但 activation reconstruction 后 final 几乎回到同一水平，说明当前瓶颈不是简单的 tensor-wise range 选得不够好。
3. 强 clipping 明确有害，尤其伤害高输入 SNR 和低 missing rate 条件，这些本来是模型恢复空间较大的样本。
4. mse_grid 成本更高，尤其全量 mse_grid 用时 `39.69 min`，但 final SNR 仍低于 baseline；没有继续扩大 mse_grid 搜索的价值。
5. NE004 与 NE005 合起来给出的机制判断是：A4 gap 的主因确实在 Conv2d activation，但不是通过 tensor-wise range/clipping 可以解决，而需要改变 activation quantization granularity 或 bit allocation。
6. 下一步 NE006 应直接做 granularity：优先 `split_proj + merge_proj + stage_output_conv` 的 g4 / per-channel，再做 all Conv2d 上限参考；range/clipping 暂时只保留为 granularity 后的辅助组合，不再作为主线。

## 2026-05-12 NE006 W4A4 activation granularity 计划

NE006 的核心问题：在 normalized 新协议下，`NE000_2 W4A4` 的主要 gap 是否来自 Conv2d activation tensor-wise A4 粒度过粗；如果是，能否找到比 all Conv2d per-channel 更可部署的 selective granularity 策略。

为什么现在应该做 granularity：

- NE004 sensitivity 已经证明 W4A4 all_off 可从 `12.9150 dB` 回到接近 W4A32 的 `17.7856 dB`，activation gap 很真实。
- NE004 中 Conv2d off 追回 all_off gap 的 `86.2%`，Linear / transformer 只有 `6.5%`，主因集中在 Conv2d activation。
- NE004 中 `split_proj + merge_proj + stage_output_conv` off 可追回 `39.1%` 的 all_off gap，是最强的部署可操作小集合。
- NE005 排除了 tensor-wise range / clipping：最佳 range 只提升 `+0.0180 dB`，强 clipping 有害，mse_grid 成本高但 final 不优。
- 旧 E006C 在旧协议下也显示 selective Conv2d granularity 比 all Conv2d per-channel 更有潜力；NE006 要在新协议和 A4 主对象上重新验证这一点。

实验主对象：

- 主线对象：`NE000_2 W4A4 final` 的 A4 activation 量化策略。
- 起点：沿用 E007 W4A32 single-GPU best checkpoint / A4 metadata seed，保持与 NE000_2、NE005 相同的 W4 weight reconstruction 起点。
- 评估：normalized `478 x 25` grid，seed `20260507`，正式结论只看 full-grid mean/median SNR、SSIM、by-source、by-input-SNR、by-missing-rate。
- W4A8：只做 sanity subset，不作为优化主线，因为 NE000 W4A8 final 已接近 W4A32。

NE006 分层实验矩阵：

| priority | experiment | granularity | selected group | 目的 | 预期判断 |
|---:|---|---|---|---|---|
| 1 | NE006a | group-wise g4 | `split_proj + merge_proj + stage_output_conv` | 最重要部署候选，验证 selective 小集合能否低成本追回 gap | 若 `>= +1 dB`，进入代表图和 packed 候选 |
| 2 | NE006b | per-channel | `split_proj + merge_proj + stage_output_conv` | selective 上限，复验旧 E006C 的关键结论 | 若显著强于 g4，再考虑 g2/g4/g8 sweep |
| 3 | NE006c | group-wise g4 | all Conv2d | 部署友好的 Conv2d 上限参考 | 判断 selective 是否接近 all Conv2d |
| 4 | NE006d | per-channel | all Conv2d | 理论上限参考 | 如果这个也弱，说明 granularity 不是主瓶颈 |
| 5 | NE006e | group-wise g4 | `stage_output_conv` | 拆解 selective 组合，测试 stage output 单独贡献 | 若接近 NE006a，说明 stage output 是核心 |
| 6 | NE006f | group-wise g4 | `split_proj + merge_proj` | 拆解 fusion 投影贡献 | 判断 split/merge 是否需要与 stage output 绑定 |
| 7 | NE006g | group-wise g4 | stage5 | 风险对照；NE004 stage5 off 强，但旧 E006C 中 stage5 独立细粒度可能有害 | 只作为风险验证，不默认主线 |
| 8 | NE006h | per-channel | stage5 | stage5 上限风险对照 | 若有害，明确禁止 stage5 独立策略 |
| 9 | NE006i | group-wise g4 | Linear / transformer sanity | 排除项 | 只确认 Linear/transformer 不是主方向 |
| 10 | NE006j | W4A8 sanity subset | all Conv2d 与 selective 小集合 | 检查 A8 与 A4 方向是否一致 | 不投入大规模 A8 优化 |

建议先执行的最小闭环：

1. `split_proj + merge_proj + stage_output_conv g4`
2. `split_proj + merge_proj + stage_output_conv per-channel`
3. `all Conv2d g4`
4. `all Conv2d per-channel`

这 4 个实验可以直接回答 NE006 的主问题：selective 细粒度是否有效、g4 是否足够、selective 是否接近 all Conv2d、per-channel 上限有多高。只有这 4 个有明确收益后，再拆 `stage_output_conv`、`split+merge`、`stage5`。

固定实验设置建议：

- `n_bits_w=4`，`n_bits_a=4`。
- 保持 `num_samples=1024`、`batch_size=16`、`init_batch_size=64`、`iters_a=5000`、`activation_lr=0.0004`、`lp_norm=2.4`，与 NE000_2 / NE005 对齐。
- 执行计划修正：细粒度 activation delta shape 需要通过 `activation_range_method=mse_grid` 写入，因此 NE006 实际使用 `mse_grid`；解释上仍把主变量限定为 granularity，不把 NE005 的 tensor-wise range/clipping 作为主线。
- 使用物理 GPU 优先级 `1 -> 2 -> 3 -> 0`；单卡默认 GPU 1。
- 输出目录建议：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant` 和 `.../eval`。

正式验收与解释标准：

- 每个 final checkpoint 必须通过 `verify_quantized_scrn`：`passed=true`，`weight_quant=true`，`act_quant=true`，activation bitwidth `4`，`activation_delta_count=52`，`non_positive_delta_count=0`，`level_offender_count=0`。
- 每个 grid eval 必须为 `11950` rows。
- 与 NE000_2 W4A4 final 对比：
  - `< +0.5 dB`：不作为主线；
  - `+0.5 ~ +1.0 dB`：辅助价值，但不足以成为核心策略；
  - `>= +1.0 dB`：强信号，进入代表图、packed equivalence 和后续组合优化；
  - `>= +2.0 dB`：优先级提高，可考虑直接作为 W4A4 主候选。
- 同时报告：
  - overall mean/median SNR、SSIM；
  - by-source，重点 `Anisotropic`、`Kerry3D`、`Shots0001`；
  - by-input-SNR，确认高输入 SNR 是否追回更多；
  - by-missing-rate，确认策略不是只改善单一 missing 条件；
  - pre-act 与 final，判断 activation reconstruction 是否放大或抵消 granularity 收益。

NE006 的关键决策树：

- 如果 `selective g4` 已经接近 `selective per-channel`，优先保留 `g4`，因为部署更友好。
- 如果 `selective per-channel` 强而 `selective g4` 弱，再做 g2/g4/g8 小 sweep。
- 如果 `all Conv2d` 明显强于 selective，说明当前 selective 小集合还不够，需要回到 NE004 的 group 分解继续扩集合。
- 如果 `all Conv2d per-channel` 仍提升有限，说明 A4 问题不只是 granularity，应考虑 mixed precision 或 bit allocation。
- 如果 `stage5` 独立策略有害，则后续禁止把 stage5 单独作为 fine granularity 默认策略，只能作为组合中谨慎纳入。

预期结论方向：

- 最可能成为 W4A4 主候选的是 `split_proj + merge_proj + stage_output_conv g4` 或其 per-channel 版本。
- `all Conv2d per-channel` 主要是上限，不一定适合部署。
- `stage5` 需要单独验证风险，不应因为 NE004 stage5 off gain 高就直接作为优化策略。
- W4A8 只做 sanity，不应挤占 W4A4 主线资源。

## 2026-05-12 NE006 W4A4 activation granularity 核心结果

本轮完成 NE006 的 4 个最小闭环实验，目标是验证 W4A4 A4 activation gap 是否能通过 Conv2d selective granularity 修复。实际执行时按实现要求使用 `--activation-range-method mse_grid` 生成 per-channel / group-wise activation delta shape；解释上仍归类为 granularity 实验，不把 NE005 的 tensor-wise range/clipping 路线重新打开。

固定设置：

- 起点：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- calibration：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- eval：normalized `478 x 25` grid，seed `20260507`
- activation reconstruction：`num_samples=1024`、`batch_size=16`、`init_batch_size=64`、`iters_a=5000`、`activation_lr=0.0004`、`lp_norm=2.4`
- GPU：NE006a 用 GPU 1；NE006b 用 GPU 3；NE006c 用 GPU 0；NE006d 因 GPU 2 持续有外部进程，转用 GPU 3。NE006c 运行期间 GPU 0 出现外部 `swinir` 进程约 `868 MiB`，未 OOM，实验参数未改。

产物目录与合法性：

| variant | quant dir | eval dir | selected | verification |
|---|---|---|---:|---|
| NE006a selective g4 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_012630_ne006a_w4a4_g4_split_merge_stage_output_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_113739_ne006a_w4a4_g4_split_merge_stage_output_grid478_seed20260507` | 15 | passed |
| NE006b selective per-channel | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_012630_ne006b_w4a4_pc_split_merge_stage_output_a5000_1024cali_gpu3` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_113739_ne006b_w4a4_pc_split_merge_stage_output_grid478_seed20260507` | 15 | passed |
| NE006c all Conv2d g4 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_012629_ne006c_w4a4_g4_all_conv2d_a5000_1024cali_gpu0` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_113739_ne006c_w4a4_g4_all_conv2d_grid478_seed20260507` | 31 | passed |
| NE006d all Conv2d per-channel | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_102116_ne006d_w4a4_pc_all_conv2d_a5000_1024cali_gpu3` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_114157_ne006d_w4a4_pc_all_conv2d_grid478_seed20260507` | 31 | passed |

四个 final checkpoint 均为合法 W4A4：`weight_quant=true`、`act_quant=true`、`n_bits_a=4`、`activation_delta_count=52`、`initialized_activation_quantizers=52`、`non_positive_delta_count=0`、`level_offender_count=0`、weight bit counts 为 `4bit=50 / 8bit=2`。四个 grid eval 均为 `11950` rows。

overall full-grid 结果：

| variant | granularity | selected | pre SNR mean/median | final SNR mean/median | final SSIM mean/median | recon gain SNR/SSIM | vs NE000_2 W4A4 | gap to E007 W4A32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NE000_2 baseline | tensor | 51 | 11.1727 / 11.1577 | 12.9150 / 13.1180 | 0.939563 / 0.954078 | +1.7422 / -0.001929 | +0.0000 / +0.000000 | -4.8706 / -0.024574 |
| NE006a selective g4 | g4 | 15 | 11.6854 / 11.7630 | 13.2824 / 13.4767 | 0.943762 / 0.958305 | +1.5970 / +0.011829 | +0.3674 / +0.004199 | -4.5032 / -0.020375 |
| NE006b selective per-channel | per-channel | 15 | 11.7721 / 11.8035 | 13.2731 / 13.4541 | 0.944148 / 0.958580 | +1.5010 / +0.004267 | +0.3582 / +0.004585 | -4.5125 / -0.019989 |
| NE006c all Conv2d g4 | g4 | 31 | 12.6877 / 12.8746 | 13.7231 / 13.9766 | 0.945327 / 0.960126 | +1.0354 / +0.014151 | +0.8082 / +0.005764 | -4.0624 / -0.018810 |
| NE006d all Conv2d per-channel | per-channel | 31 | 12.5608 / 12.6432 | 13.6752 / 13.9117 | 0.945889 / 0.960663 | +1.1144 / +0.002959 | +0.7603 / +0.006326 | -4.1104 / -0.018248 |

single-sample sanity 只用于确认运行，没有用于策略优劣判断：

| variant | init s | recon s | elapsed min | single pre SNR/SSIM | single final SNR/SSIM | single recon gain |
|---|---:|---:|---:|---:|---:|---:|
| NE006a selective g4 | 26.3 | 1367.4 | 23.39 | 13.2416 / 0.893237 | 13.3777 / 0.907454 | +0.1361 / +0.014217 |
| NE006b selective per-channel | 29.1 | 1366.9 | 23.43 | 13.3452 / 0.901292 | 13.4085 / 0.904635 | +0.0633 / +0.003342 |
| NE006c all Conv2d g4 | 26.9 | 1968.7 | 33.41 | 13.0798 / 0.897449 | 13.4598 / 0.909815 | +0.3800 / +0.012366 |
| NE006d all Conv2d per-channel | 29.4 | 1314.0 | 22.44 | 13.3368 / 0.908258 | 13.4437 / 0.910167 | +0.1070 / +0.001909 |

by-source 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | Anisotropic | Kerry3D | Shots0001 |
|---|---:|---:|---:|
| NE000_2 baseline | 13.2802 / +0.00 | 9.0332 / +0.00 | 13.0047 / +0.00 |
| NE006a selective g4 | 13.8332 / +0.55 | 9.1187 / +0.09 | 13.3477 / +0.34 |
| NE006b selective per-channel | 13.7811 / +0.50 | 9.1485 / +0.12 | 13.3452 / +0.34 |
| NE006c all Conv2d g4 | 14.6337 / +1.35 | 9.1944 / +0.16 | 13.7339 / +0.73 |
| NE006d all Conv2d per-channel | 14.5110 / +1.23 | 9.2155 / +0.18 | 13.6976 / +0.69 |

by-input-SNR 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | -2 dB | -1 dB | 1 dB | 5 dB | 10 dB |
|---|---:|---:|---:|---:|---:|
| NE000_2 baseline | 11.6505 / +0.00 | 12.0574 / +0.00 | 12.7252 / +0.00 | 13.6971 / +0.00 | 14.4447 / +0.00 |
| NE006a selective g4 | 11.9312 / +0.28 | 12.3545 / +0.30 | 13.0656 / +0.34 | 14.1120 / +0.41 | 14.9485 / +0.50 |
| NE006b selective per-channel | 11.9029 / +0.25 | 12.3312 / +0.27 | 13.0518 / +0.33 | 14.1140 / +0.42 | 14.9656 / +0.52 |
| NE006c all Conv2d g4 | 12.3396 / +0.69 | 12.7811 / +0.72 | 13.5151 / +0.79 | 14.5793 / +0.88 | 15.4006 / +0.96 |
| NE006d all Conv2d per-channel | 12.2798 / +0.63 | 12.7257 / +0.67 | 13.4628 / +0.74 | 14.5364 / +0.84 | 15.3714 / +0.93 |

by-missing-rate 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | 0.02 | 0.08 | 0.18 | 0.28 | 0.38 |
|---|---:|---:|---:|---:|---:|
| NE000_2 baseline | 13.9245 / +0.00 | 13.6206 / +0.00 | 13.0601 / +0.00 | 12.4028 / +0.00 | 11.5668 / +0.00 |
| NE006a selective g4 | 14.3411 / +0.42 | 14.0199 / +0.40 | 13.4314 / +0.37 | 12.7460 / +0.34 | 11.8734 / +0.31 |
| NE006b selective per-channel | 14.3468 / +0.42 | 14.0236 / +0.40 | 13.4200 / +0.36 | 12.7241 / +0.32 | 11.8510 / +0.28 |
| NE006c all Conv2d g4 | 14.8362 / +0.91 | 14.5018 / +0.88 | 13.8812 / +0.82 | 13.1596 / +0.76 | 12.2368 / +0.67 |
| NE006d all Conv2d per-channel | 14.8009 / +0.88 | 14.4599 / +0.84 | 13.8326 / +0.77 | 13.1000 / +0.70 | 12.1826 / +0.62 |

人类可读结论：

1. NE006 证明 granularity 比 NE005 的 range/clipping 更有效。NE005 最好只有 `+0.018 dB`，NE006 selective 可到 `+0.36 dB`，all Conv2d 可到 `+0.76 ~ +0.81 dB`。
2. 但这 4 个核心实验都没有达到 `+1 dB` 强信号阈值，也没有接近 W4A32。即使最好的 NE006c，距离 E007 W4A32 final 仍有 `-4.0624 dB` mean SNR gap。
3. `split_proj + merge_proj + stage_output_conv` selective 组合有稳定正收益，但幅度不足：g4 `+0.3674 dB`，per-channel `+0.3582 dB`。这说明旧 E006C 的 selective 结论在新 normalized A4 场景下不完全复现，当前 selective 小集合不是足够强的 A4 主策略。
4. g4 不弱于 per-channel。本轮两个对照里，selective g4 略高于 selective per-channel，all Conv2d g4 也高于 all Conv2d per-channel。对 W4A4 来说，更细的 per-channel 并没有自动带来更高 full-grid SNR，后续不应默认把 per-channel 当上限最优。
5. all Conv2d 明显强于 selective，说明 NE004 里 Conv2d 主导的判断成立，但当前 selective 集合只覆盖了一部分关键误差。需要继续拆分或扩展 Conv2d 组，而不是直接把 selective g4 标为候选。
6. by-source 显示 Anisotropic 获益最大，all Conv2d g4 达到 `+1.35 dB`；Kerry3D 只有 `+0.16 ~ +0.18 dB`。W4A4 的残余 gap 仍有强 source 依赖，后续不能只看 overall。
7. by-input-SNR 显示输入越干净收益越大：all Conv2d g4 从 `-2 dB` 的 `+0.69 dB` 增至 `10 dB` 的 `+0.96 dB`，说明细粒度主要恢复高质量输入场景中的 activation 表达能力。
8. by-missing-rate 显示低 missing rate 收益更大，高 missing rate 收益下降：all Conv2d g4 从 `0.02` 的 `+0.91 dB` 降至 `0.38` 的 `+0.67 dB`。细粒度不是只改善单一条件，但在困难缺失条件下仍不足。
9. activation reconstruction 仍然重要。四个变体 final-pre mean SNR gain 都为正，`+1.04 ~ +1.60 dB`；但 selective 组 pre-act 不低，final 仍弱，说明仅在 15 个 Conv2d quantizer 上细粒度化不能充分释放 reconstruction。

后续决策：

- 暂不做 NE006 packed export，因为最佳 all Conv2d g4 只有 `+0.8082 dB`，未达到 `>= +1 dB` 主候选阈值。
- 继续 NE006e-h 拆分实验是有必要的，但优先级应调整为“寻找缺失的 Conv2d 关键组”而不是验证原 selective 小集合。
- 下一批建议：
  1. `stage_output_conv g4` 和 `split_proj + merge_proj g4` 拆分，判断 selective 小集合内谁贡献主要收益。
  2. `conv role g4`、`fusion branch g4`、`cnn branch g4`，从 NE004 的 Conv2d 主导结果里寻找比当前 selective 更接近 all Conv2d 的较小集合。
  3. `stage4/stage5 g4` 风险对照，验证旧 E006 中 stage5 独立细粒度有害的现象在 A4 normalized 协议下是否仍存在。
  4. 若仍没有 `>= +1 dB`，转向 mixed precision / selective A8，而不是继续扩大 tensor-wise range/clipping。

## 2026-05-12 NE006 后续展开计划修订

NE006 核心四组实验之后，后续路线需要从“验证旧 selective 策略”调整为“寻找 all Conv2d g4 收益中的关键缺失组”。当前证据是：

- 旧重点组合 `split_proj + merge_proj + stage_output_conv` 有稳定收益，但只有 `+0.36 dB` 左右，不足以成为 W4A4 主策略。
- `all Conv2d g4` 达到 `+0.8082 dB`，明显强于 selective，但仍未达到 `+1 dB` 强信号阈值。
- `g4` 不弱于 per-channel，甚至在 selective 和 all Conv2d 两组对照里都略高，因此下一批优先做 g4 拆分，不优先做 per-channel sweep。
- NE005 已排除 tensor-wise range/clipping 主线，后续不再回到 range/clipping，除非与 granularity 或 mixed precision 组合验证。

下一批 NE006 建议按 g4 拆分执行：

| priority | experiment | selector | 目的 |
|---:|---|---|---|
| 1 | NE006e | `stage_output_conv g4` | 判断原 selective 小集合里 stage output 是否贡献主要收益 |
| 2 | NE006f | `split_proj + merge_proj g4` | 判断 fusion 投影是否需要与 stage output 绑定 |
| 3 | NE006g | `conv role g4` | 覆盖普通 Conv2d role，寻找当前 selective 漏掉的收益 |
| 4 | NE006h | `fusion branch g4` | 从 branch 维度验证 fusion 是否比 role-based 更集中 |
| 5 | NE006i | `cnn branch g4` | 检查早期 CNN branch 是否贡献大 |
| 6 | NE006j | `stage1/2/3/4/5 Conv2d g4` | 找 stage 层级热点，特别复核 stage5 风险 |
| 7 | NE006k | `all Conv2d except stage5 g4` | 如果 stage5 独立细粒度不稳定，验证排除 stage5 后是否更稳 |

判断标准：

- 如果某个小集合达到 `+0.7 ~ +0.8 dB`，接近 all Conv2d g4，则优先围绕该集合做 g2/g4/g8 或少量 per-channel 对照。
- 如果只有 all Conv2d 有收益，小集合都明显弱，说明 A4 activation 误差更分散，后续应转向 mixed precision / selective A8，而不是继续找单一小集合。
- 如果任何拆分组合达到 `>= +1 dB`，再进入代表图、packed equivalence 和部署候选评估。
- 如果 stage5 独立策略有害，应避免把 stage5 单独作为 fine granularity 默认策略，只在组合里谨慎纳入或显式排除。

当前不建议立即做：

- 不做 packed export：最佳结果未达到 `>= +1 dB` 主候选阈值。
- 不做 per-channel 大 sweep：当前 per-channel 未优于 g4。
- 不做 range/clipping 扩展：NE005 已显示收益接近噪声或有害。

下一步实际执行应优先启动 NE006e-h 的 g4 拆分；如果这些拆分仍不能逼近 all Conv2d g4，则进入 mixed precision / selective A8 方向。

## 2026-05-12 NE006e-h W4A4 g4 granularity 拆分结果

本批完成 4 个 g4 拆分实验，用于定位 `all Conv2d g4` 的 `+0.8082 dB` 收益来自哪些 Conv2d 子组。本批不改代码，不做 packed，不做 per-channel sweep。

固定设置：

- 起点：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- calibration：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- eval：normalized `478 x 25` grid，seed `20260507`
- activation reconstruction：`num_samples=1024`、`batch_size=16`、`init_batch_size=64`、`iters_a=5000`、`activation_lr=0.0004`、`lp_norm=2.4`
- granularity：`group_wise g4`
- range shape 写入：`activation_range_method=mse_grid`、`range_loss_p=2.4`、`range_max_values_per_layer=500000`
- GPU：NE006e 用 GPU 1；NE006f 用 GPU 2；NE006g 用 GPU 3；NE006h 用 GPU 0。沙箱内 CUDA 不可用，按完全相同参数提升权限执行。

产物目录与合法性：

| variant | selector | selected | quant dir | eval dir | verification |
|---|---|---:|---|---|---|
| NE006e stage_output_conv g4 | `role=stage_output_conv,module_type=Conv2d` | 5 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_120537_ne006e_w4a4_g4_stage_output_conv_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_123059_ne006e_w4a4_g4_stage_output_conv_grid478_seed20260507` | passed |
| NE006f split+merge g4 | `split_proj + merge_proj` | 10 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_120537_ne006f_w4a4_g4_split_merge_a5000_1024cali_gpu2` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_123059_ne006f_w4a4_g4_split_merge_grid478_seed20260507` | passed |
| NE006g conv role g4 | `role=conv,module_type=Conv2d` | 15 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_120536_ne006g_w4a4_g4_conv_role_a5000_1024cali_gpu3` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_123059_ne006g_w4a4_g4_conv_role_grid478_seed20260507` | passed |
| NE006h fusion branch g4 | `branch=fusion,module_type=Conv2d` | 10 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_120535_ne006h_w4a4_g4_fusion_branch_a5000_1024cali_gpu0` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_123055_ne006h_w4a4_g4_fusion_branch_grid478_seed20260507` | passed |

四个 final checkpoint 均为合法 W4A4：`weight_quant=true`、`act_quant=true`、`n_bits_a=4`、`activation_delta_count=52`、`initialized_activation_quantizers=52`、`non_positive_delta_count=0`、`level_offender_count=0`、weight bit counts 为 `4bit=50 / 8bit=2`。四个 grid eval 均为 `11950` rows。

overall full-grid 结果：

| variant | selected | pre SNR mean/median | final SNR mean/median | final SSIM mean/median | recon gain SNR/SSIM | vs NE000_2 W4A4 | gap to all Conv2d g4 | gap to E007 W4A32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NE000_2 baseline | 51 | 11.1727 / 11.1577 | 12.9150 / 13.1180 | 0.939563 / 0.954078 | +1.7422 / -0.001929 | +0.0000 / +0.000000 | -0.8082 / -0.005764 | -4.8706 / -0.024574 |
| NE006e stage_output_conv g4 | 5 | 11.5374 / 11.5735 | 13.2582 / 13.4803 | 0.940681 / 0.955035 | +1.7208 / +0.002893 | +0.3433 / +0.001118 | -0.4649 / -0.004646 | -4.5274 / -0.023456 |
| NE006f split+merge g4 | 10 | 11.2482 / 11.2813 | 12.9314 / 13.1217 | 0.942282 / 0.956952 | +1.6832 / +0.006632 | +0.0164 / +0.002719 | -0.7917 / -0.003045 | -4.8542 / -0.021855 |
| NE006g conv role g4 | 15 | 11.6154 / 11.6726 | 13.2241 / 13.4930 | 0.941117 / 0.956052 | +1.6087 / +0.001176 | +0.3091 / +0.001554 | -0.4990 / -0.004210 | -4.5615 / -0.023020 |
| NE006h fusion branch g4 | 10 | 11.2482 / 11.2813 | 12.9314 / 13.1217 | 0.942282 / 0.956952 | +1.6832 / +0.006632 | +0.0164 / +0.002719 | -0.7917 / -0.003045 | -4.8542 / -0.021855 |

single-sample sanity 只用于确认运行，不用于策略优劣判断：

| variant | init s | recon s | elapsed min | single pre SNR/SSIM | single final SNR/SSIM | single recon gain |
|---|---:|---:|---:|---:|---:|---:|
| NE006e stage_output_conv g4 | 25.3 | 1327.7 | 22.63 | 13.2265 / 0.907595 | 13.2017 / 0.899682 | -0.0248 / -0.007913 |
| NE006f split+merge g4 | 25.6 | 1328.0 | 22.64 | 13.1519 / 0.893781 | 13.3712 / 0.907287 | +0.2193 / +0.013506 |
| NE006g conv role g4 | 25.3 | 1327.2 | 22.62 | 13.1746 / 0.905096 | 13.2919 / 0.904346 | +0.1173 / -0.000750 |
| NE006h fusion branch g4 | 25.5 | 1324.2 | 22.57 | 13.1519 / 0.893781 | 13.3712 / 0.907287 | +0.2193 / +0.013506 |

by-source 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | Anisotropic | Kerry3D | Shots0001 |
|---|---:|---:|---:|
| NE000_2 baseline | 13.2802 / +0.00 | 9.0332 / +0.00 | 13.0047 / +0.00 |
| NE006e stage_output_conv g4 | 13.8765 / +0.60 | 9.0623 / +0.03 | 13.3119 / +0.31 |
| NE006f split+merge g4 | 13.2709 / -0.01 | 9.0911 / +0.06 | 13.0244 / +0.02 |
| NE006g conv role g4 | 13.7999 / +0.52 | 9.1086 / +0.08 | 13.2827 / +0.28 |
| NE006h fusion branch g4 | 13.2709 / -0.01 | 9.0911 / +0.06 | 13.0244 / +0.02 |

by-input-SNR 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | -2 dB | -1 dB | 1 dB | 5 dB | 10 dB |
|---|---:|---:|---:|---:|---:|
| NE000_2 baseline | 11.6505 / +0.00 | 12.0574 / +0.00 | 12.7252 / +0.00 | 13.6971 / +0.00 | 14.4447 / +0.00 |
| NE006e stage_output_conv g4 | 11.9284 / +0.28 | 12.3515 / +0.29 | 13.0549 / +0.33 | 14.0807 / +0.38 | 14.8756 / +0.43 |
| NE006f split+merge g4 | 11.6522 / +0.00 | 12.0586 / +0.00 | 12.7332 / +0.01 | 13.7145 / +0.02 | 14.4985 / +0.05 |
| NE006g conv role g4 | 11.9770 / +0.33 | 12.3801 / +0.32 | 13.0499 / +0.32 | 14.0011 / +0.30 | 14.7125 / +0.27 |
| NE006h fusion branch g4 | 11.6522 / +0.00 | 12.0586 / +0.00 | 12.7332 / +0.01 | 13.7145 / +0.02 | 14.4985 / +0.05 |

by-missing-rate 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | 0.02 | 0.08 | 0.18 | 0.28 | 0.38 |
|---|---:|---:|---:|---:|---:|
| NE000_2 baseline | 13.9245 / +0.00 | 13.6206 / +0.00 | 13.0601 / +0.00 | 12.4028 / +0.00 | 11.5668 / +0.00 |
| NE006e stage_output_conv g4 | 14.2427 / +0.32 | 13.9623 / +0.34 | 13.4224 / +0.36 | 12.7624 / +0.36 | 11.9012 / +0.33 |
| NE006f split+merge g4 | 13.9721 / +0.05 | 13.6466 / +0.03 | 13.0696 / +0.01 | 12.4036 / +0.00 | 11.5650 / -0.00 |
| NE006g conv role g4 | 14.2861 / +0.36 | 13.9599 / +0.34 | 13.3725 / +0.31 | 12.6849 / +0.28 | 11.8172 / +0.25 |
| NE006h fusion branch g4 | 13.9721 / +0.05 | 13.6466 / +0.03 | 13.0696 / +0.01 | 12.4036 / +0.00 | 11.5650 / -0.00 |

关键结构发现：

- NE006f 和 NE006h 选中的 quantizer 完全相同，都是 5 个 stage 的 `split_proj + merge_proj` 共 10 个 Conv2d quantizer；因此 full-grid 结果完全一致。当前代码结构中 `branch=fusion,module_type=Conv2d` 等价于 `split_proj + merge_proj`。
- NE006e 选中 5 个 stage output Conv2d：`model.stage1.1` 到 `model.stage5.1`。
- NE006g 选中 15 个 `conv_branch` Conv2d。
- NE006c all Conv2d g4 选中 31 个 Conv2d，比 `stage_output_conv + split/merge + conv role` 的 30 个多出 `model.head`。因此 all Conv2d 的额外收益可能来自 head，或来自 head 与其它 Conv2d 组的组合效应。

人类可读结论：

1. stage output 是原 selective 组合里的主要贡献。NE006e 单独达到 `+0.3433 dB`，而 NE006a 的 `split_proj + merge_proj + stage_output_conv` 是 `+0.3674 dB`；加入 split/merge 只多 `+0.024 dB` 左右。
2. split/merge 基本不是有效修复方向。NE006f/NE006h 只有 `+0.0164 dB`，接近噪声级；这与旧 E006C 中 split/merge 重要的结论不一致，是 NE 系列与旧 E 系列的关键差异之一。
3. conv role 有独立正收益但也不够强。NE006g 为 `+0.3091 dB`，略弱于 stage output；它对低输入 SNR 条件收益更强，对高输入 SNR 条件收益反而下降。
4. all Conv2d g4 的 `+0.8082 dB` 不是由单一小组贡献。stage output 和 conv role 各有约 `+0.3 dB`，split/merge 近乎无效；剩余差距可能来自 head 或组合效应。
5. 当前没有任何拆分组合接近 `+0.7 ~ +0.8 dB`，因此还不能把某个小集合作为 W4A4 主候选，也不应做 packed export。
6. full-grid 与 single-sample 再次不一致：NE006e 单样本 final 低于 pre，但 full-grid 是本批最强；后续仍必须以 `478 x 25` 为准。

后续决策：

- 继续 NE006 是必要的，但下一批应从“角色拆分”转向“head 与组合效应”：
  1. `head g4`：验证 all Conv2d 中唯一未被本批覆盖的 `model.head` 是否有显著贡献。
  2. `stage_output_conv + conv role g4`：验证两个有效小组组合后是否接近 all Conv2d。
  3. `all Conv2d except split/merge g4`：验证排除无效 split/merge 后是否保留 all Conv2d 收益。
  4. `all Conv2d except head g4`：验证 head 是否是 all Conv2d 收益的重要来源。
- stage1-5 拆分仍有价值，但优先级低于 head / 组合效应。若组合实验仍不能解释 all Conv2d 收益，再进入 stage-level。
- 不进入 packed export；不做 per-channel sweep；不回到 range/clipping。

## 2026-05-12 NE006e-h 后续影响与下一批计划

NE006e-h 对后续路线的影响很明确：新 normalized W4A4 场景下，不应继续沿旧 E006C 的 `split_proj + merge_proj + stage_output_conv` selective 策略直接加码。旧 selective 组合的收益主要来自 `stage_output_conv`，而不是 `split_proj + merge_proj`。

当前关键判断：

- `split_proj + merge_proj` 从旧实验里的重要对象降级为近似无效对象。NE006f/NE006h full-grid 只有 `+0.0164 dB`，接近噪声级。
- `stage_output_conv` 是当前最有效的小集合，5 个 quantizer 带来 `+0.3433 dB`，基本解释了旧 selective 组合的大部分收益。
- `conv role` 是另一个独立有效组，15 个 quantizer 带来 `+0.3091 dB`，说明 all Conv2d 收益分散在多个 Conv2d 子组。
- `all Conv2d g4` 仍有 `+0.8082 dB`，明显强于任何单独拆分组；这说明剩余收益可能来自 `model.head`、head 与其它 Conv2d 的组合效应，或多组 activation 误差的非线性叠加。
- full-grid 与 single-sample 再次冲突，后续仍必须只用 normalized `478 x 25` grid 做策略判断。

下一批 NE006 建议只做 4 个 head / 组合 / 排除实验：

| priority | experiment | selector | 目的 | 预期解释 |
|---:|---|---|---|---|
| 1 | NE006i | `head g4` | 验证 all Conv2d 中唯一未拆出的 `model.head` 是否关键 | 如果有明显收益，head 需要纳入后续策略 |
| 2 | NE006j | `stage_output_conv + conv role g4` | 组合两个已知有效组 | 如果接近 all Conv2d，说明 split/merge 可排除 |
| 3 | NE006k | `all Conv2d except split/merge g4` | 从 all Conv2d 中移除近似无效 split/merge | 如果接近 all Conv2d，证明 split/merge 不该进主策略 |
| 4 | NE006l | `all Conv2d except head g4` | 从 all Conv2d 中移除 head | 判断 head 是否解释 all Conv2d 的剩余优势 |

执行设置应继续沿用 NE006 固定口径：

- 起点：`e007_w4a32_nbitsa4_metadata_seed.pth`
- `n_bits_w=4`、`n_bits_a=4`
- `activation_granularity=group_wise`、`activation_group_size=4`
- `activation_range_method=mse_grid`，用于写入 g4 activation delta shape
- `iters_a=5000`、`activation_lr=0.0004`、`lp_norm=2.4`
- calibration：normalized 1024 stratified calibration
- eval：normalized `478 x 25` grid，seed `20260507`

判断标准：

- 若 `all Conv2d except split/merge` 接近 NE006c all Conv2d g4，后续策略应明确排除 split/merge。
- 若 `head g4` 或 `all Conv2d except head` 显著改变结果，后续需要围绕 head 做组合验证。
- 若 `stage_output_conv + conv role` 仍明显低于 all Conv2d，说明存在 head 参与效应或多组非线性组合效应。
- 若这 4 个实验仍没有 `>= +1 dB` 的策略，NE006 granularity 单线应降级，下一阶段转向 `W4A4 + selective A8` 或 mixed precision。

当前不建议：

- 不做 packed export：没有达到主候选阈值。
- 不做 per-channel sweep：g4 当前不弱于 per-channel。
- 不回到 range/clipping：NE005 已排除主线价值。
- 不用单样本做任何策略筛选。

## 2026-05-12 NE006i-l W4A4 g4 head / combination 实验结果

本批实验按上一节计划完成 4 个 head / 组合 / 排除实验，目标是解释 NE006c `all Conv2d g4` 的 `+0.8082 dB` 收益是否来自 `head`、`stage_output_conv + conv role` 组合，或是否可以排除 `split_proj + merge_proj`。

固定口径：

- 起点：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- calibration：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- eval：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- eval grid：`478 x 25 = 11950` rows，seed `20260507`
- activation：W4A4，`group_wise`，`group_size=4`，`activation_range_method=mse_grid`
- reconstruction：`iters_a=5000`，`activation_lr=0.0004`，`lp_norm=2.4`

run 目录：

| id | quant run dir | eval run dir | GPU | selector | selected |
|---|---|---|---:|---|---:|
| NE006i | `.../quant/20260512_131527_ne006i_w4a4_g4_head_a5000_1024cali_gpu1` | `.../eval/20260512_134445_ne006i_w4a4_g4_head_grid478_seed20260507` | 1 | `role=head,module_type=Conv2d` | 1 |
| NE006j | `.../quant/20260512_131527_ne006j_w4a4_g4_stage_output_conv_role_a5000_1024cali_gpu2` | `.../eval/20260512_134445_ne006j_w4a4_g4_stage_output_conv_role_grid478_seed20260507` | 2 | `stage_output_conv + conv` | 20 |
| NE006k | `.../quant/20260512_131526_ne006k_w4a4_g4_all_conv2d_except_split_merge_a5000_1024cali_gpu3` | `.../eval/20260512_134444_ne006k_w4a4_g4_all_conv2d_except_split_merge_grid478_seed20260507` | 3 | `all Conv2d except split_proj/merge_proj` | 21 |
| NE006l | `.../quant/20260512_131526_ne006l_w4a4_g4_all_conv2d_except_head_a5000_1024cali_gpu0` | `.../eval/20260512_134444_ne006l_w4a4_g4_all_conv2d_except_head_grid478_seed20260507` | 0 | `all Conv2d except head` | 30 |

验证结果：4 个 final checkpoint 均 `passed=true`，`weight_quant=true`，`act_quant=true`，`n_bits_a=4`，`activation_delta_count=52`，`initialized_activation_quantizers=52`，`level_offender_count=0`。activation-only metrics 中 `non_positive_delta_count=0`，说明本批没有合法性问题。

overall full-grid 结果：

| variant | selected | pre SNR mean/median | final SNR mean/median | final SSIM mean/median | recon gain SNR/SSIM | vs NE000_2 W4A4 | gap to NE006c all Conv2d g4 | gap to E007 W4A32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NE000_2 baseline | 51 | 11.1727 / 11.1577 | 12.9150 / 13.1180 | 0.939563 / 0.954078 | +1.7422 / -0.001929 | +0.0000 / +0.000000 | -0.8082 / -0.005764 | -4.8706 / -0.024574 |
| NE006c all Conv2d g4 | 31 | 12.6877 / 12.8746 | 13.7231 / 13.9766 | 0.945327 / 0.960126 | +1.0354 / +0.003835 | +0.8082 / +0.005764 | +0.0000 / +0.000000 | -4.0624 / -0.018810 |
| NE006i head g4 | 1 | 11.1606 / 11.1444 | 12.9031 / 13.1007 | 0.939624 / 0.954139 | +1.7425 / -0.002001 | -0.0119 / +0.000061 | -0.8200 / -0.005703 | -4.8825 / -0.024513 |
| NE006j stage_output+conv g4 | 20 | 12.1962 / 12.3257 | 13.6186 / 13.8908 | 0.942366 / 0.956958 | +1.4224 / +0.005869 | +0.7036 / +0.002803 | -0.1045 / -0.002961 | -4.1670 / -0.021771 |
| NE006k all Conv2d except split/merge g4 | 21 | 12.1841 / 12.3104 | 13.6097 / 13.8824 | 0.942403 / 0.957057 | +1.4256 / +0.005919 | +0.6948 / +0.002840 | -0.1134 / -0.002924 | -4.1759 / -0.021734 |
| NE006l all Conv2d except head g4 | 30 | 12.6993 / 12.8885 | 13.7326 / 13.9838 | 0.945303 / 0.960038 | +1.0334 / +0.003626 | +0.8177 / +0.005740 | +0.0095 / -0.000024 | -4.0529 / -0.018834 |

single-sample sanity 只用于确认运行，不作为策略判断：

| variant | init s | recon s | elapsed min | single pre SNR/SSIM | single final SNR/SSIM | single gain |
|---|---:|---:|---:|---:|---:|---:|
| NE006i head g4 | 25.0 | 1320.9 | 22.51 | 13.1970 / 0.901847 | 13.2355 / 0.897892 | +0.0384 / -0.003955 |
| NE006j stage_output+conv g4 | 25.7 | 1340.7 | 22.86 | 13.1508 / 0.906505 | 13.3789 / 0.906634 | +0.2281 / +0.000128 |
| NE006k all Conv2d except split/merge g4 | 25.7 | 1321.4 | 22.51 | 13.1808 / 0.904265 | 13.3432 / 0.904612 | +0.1624 / +0.000347 |
| NE006l all Conv2d except head g4 | 26.1 | 1325.5 | 22.61 | 13.1257 / 0.896068 | 13.4422 / 0.906453 | +0.3165 / +0.010384 |

by-source 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | Anisotropic | Kerry3D | Shots0001 |
|---|---:|---:|---:|
| NE000_2 baseline | 13.2802 / +0.00 | 9.0332 / +0.00 | 13.0047 / +0.00 |
| NE006i head g4 | 13.2608 / -0.02 | 9.0372 / +0.00 | 12.9936 / -0.01 |
| NE006j stage_output+conv g4 | 14.4953 / +1.22 | 9.1336 / +0.10 | 13.6341 / +0.63 |
| NE006k all Conv2d except split/merge g4 | 14.4754 / +1.20 | 9.1433 / +0.11 | 13.6266 / +0.62 |
| NE006l all Conv2d except head g4 | 14.6524 / +1.37 | 9.1988 / +0.17 | 13.7419 / +0.74 |

by-input-SNR 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | -2 dB | -1 dB | 1 dB | 5 dB | 10 dB |
|---|---:|---:|---:|---:|---:|
| NE006i head g4 | 11.6400 / -0.01 | 12.0429 / -0.01 | 12.7113 / -0.01 | 13.6870 / -0.01 | 14.4342 / -0.01 |
| NE006j stage_output+conv g4 | 12.2814 / +0.63 | 12.7096 / +0.65 | 13.4247 / +0.70 | 14.4523 / +0.76 | 15.2251 / +0.78 |
| NE006k all Conv2d except split/merge g4 | 12.2715 / +0.62 | 12.7029 / +0.65 | 13.4131 / +0.69 | 14.4424 / +0.75 | 15.2187 / +0.77 |
| NE006l all Conv2d except head g4 | 12.3480 / +0.70 | 12.7912 / +0.73 | 13.5235 / +0.80 | 14.5910 / +0.89 | 15.4095 / +0.96 |

by-missing-rate 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | 0.02 | 0.08 | 0.18 | 0.28 | 0.38 |
|---|---:|---:|---:|---:|---:|
| NE006i head g4 | 13.9038 / -0.02 | 13.6028 / -0.02 | 13.0506 / -0.01 | 12.3946 / -0.01 | 11.5636 / -0.00 |
| NE006j stage_output+conv g4 | 14.6629 / +0.74 | 14.3597 / +0.74 | 13.7830 / +0.72 | 13.0911 / +0.69 | 12.1965 / +0.63 |
| NE006k all Conv2d except split/merge g4 | 14.6463 / +0.72 | 14.3440 / +0.72 | 13.7752 / +0.72 | 13.0885 / +0.69 | 12.1945 / +0.63 |
| NE006l all Conv2d except head g4 | 14.8501 / +0.93 | 14.5173 / +0.90 | 13.8922 / +0.83 | 13.1633 / +0.76 | 12.2403 / +0.67 |

关键结论：

1. `head g4` 单独无效。NE006i final SNR mean `12.9031`，比 NE000_2 baseline 还低 `-0.0119 dB`，说明 `model.head` 不是独立收益来源。
2. `all Conv2d except head` 与 `all Conv2d g4` 基本等价，甚至 SNR mean 高 `+0.0095 dB`。因此 head 不但不是关键，作为 g4 细粒度对象还可以从主策略中排除。
3. `stage_output_conv + conv role` 解释了 all Conv2d 的大部分收益。NE006j 达到 `13.6186 dB`，只比 all Conv2d g4 低 `0.1045 dB`，且比 W4A4 baseline 高 `+0.7036 dB`。
4. `all Conv2d except split/merge` 与 NE006j 几乎一致：`13.6097 dB`，比 NE006j 低 `0.0089 dB`。这进一步证明 split/merge 不是新协议 W4A4 主收益来源。
5. `all Conv2d except head` 是当前最强 W4A4 g4 结果：`13.7326 / 13.9838`，相对 NE000_2 提升 `+0.8177 dB`，略高于 NE006c all Conv2d g4。
6. by-source 收益主要来自 Anisotropic 和 Shots0001，Kerry3D 提升始终较小。NE006l 在 Anisotropic 上 `+1.37 dB`，Shots0001 上 `+0.74 dB`，Kerry3D 只有 `+0.17 dB`。
7. by-input-SNR 趋势显示收益在高输入 SNR 条件下更大；NE006l 从 `-2 dB` 条件的 `+0.70 dB` 增至 `10 dB` 条件的 `+0.96 dB`。
8. by-missing-rate 趋势显示收益在低 missing rate 更大，missing rate 越高收益越小；NE006l 从 `0.02` 的 `+0.93 dB` 降至 `0.38` 的 `+0.67 dB`。

对后续实验的影响：

- NE006 granularity 仍有价值，但当前最佳只到 `+0.8177 dB`，没有达到 `>= +1.0 dB` 强候选阈值，更没有达到可以替代 W4A8 的程度。
- 后续不应再优先研究 `split_proj + merge_proj`，也不应把 `head` 纳入 W4A4 g4 主策略。
- 当前最合理的 W4A4 g4 候选是 `all Conv2d except head`，但部署前还需要更强证据；它可以作为后续 mixed precision / selective A8 的基础组合。
- 若继续 granularity，应转向 stage-level 拆分：优先验证 `stage_output_conv + conv role` 是否只需要某些 stage，或者 `all Conv2d except head` 是否可以进一步减少 selected count。
- 若目标是把 W4A4 推近 W4A32/W4A8，单纯 g4 granularity 可能不够，下一阶段应准备 `W4A4 + selective A8` 或 mixed precision，优先把残余误差最大的 Conv2d 子组升到 A8。

下一步建议：

1. 可选继续 NE006m-q：做 `stage1/2/3/4/5 Conv2d except head g4` 或 `stage_output_conv + conv role` 的 stage-level 拆分，目标是减少 selected count 并定位 residual gain。
2. 更直接的下一阶段是 NE007：以 `all Conv2d except head g4` / `stage_output_conv + conv role g4` 为基础，做 selective A8 或 mixed precision，对 residual gap 最高的 Conv2d 子组升 A8。
3. 暂不做 packed export。当前 W4A4 g4 最强结果仍比 E007 W4A32 低 `4.0529 dB`，packed equivalence 不会改变模型质量判断。

## 2026-05-12 NE006m-q W4A4 g4 stage-level granularity 实验结果

本批继续 NE006，只做 stage-level W4A4 g4 拆分，目标是解释 `NE006l all Conv2d except head g4` 的 `+0.8177 dB` 收益是否由单个 stage 主导，或来自跨 stage 叠加。

固定口径：

- 起点：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- calibration：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- eval：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- eval grid：`478 x 25 = 11950` rows，seed `20260507`
- activation：W4A4，`group_wise`，`group_size=4`，`activation_range_method=mse_grid`
- reconstruction：`iters_a=5000`，`activation_lr=0.0004`，`lp_norm=2.4`

执行备注：

- NE006m-p 使用 GPU `1,2,3,0` 并行完成；NE006q 最初在 GPU3 启动的 run `20260512_163124_ne006q_w4a4_g4_stage5_conv2d_a5000_1024cali_gpu3` 只产出 pre-act checkpoint，没有产出 final/config/metrics，因此不纳入结果。
- NE006q 使用完全相同参数在 GPU3 重跑，实际有效 run 为 `20260512_172203_ne006q_w4a4_g4_stage5_conv2d_a5000_1024cali_gpu3_rerun`。
- eval 阶段 GPU0 出现外部进程占用，正式 grid eval 使用 GPU `1,2,3` 轮转完成，未修改 batch、seed 或实验变量。

run 目录：

| id | quant run dir | eval run dir | selected |
|---|---|---|---:|
| NE006m | `.../quant/20260512_155253_ne006m_w4a4_g4_stage1_conv2d_a5000_1024cali_gpu1` | `.../eval/20260512_175147_ne006m_w4a4_g4_stage1_conv2d_grid478_seed20260507` | 6 |
| NE006n | `.../quant/20260512_155252_ne006n_w4a4_g4_stage2_conv2d_a5000_1024cali_gpu2` | `.../eval/20260512_175147_ne006n_w4a4_g4_stage2_conv2d_grid478_seed20260507` | 6 |
| NE006o | `.../quant/20260512_155252_ne006o_w4a4_g4_stage3_conv2d_a5000_1024cali_gpu3` | `.../eval/20260512_175146_ne006o_w4a4_g4_stage3_conv2d_grid478_seed20260507` | 6 |
| NE006p | `.../quant/20260512_155252_ne006p_w4a4_g4_stage4_conv2d_a5000_1024cali_gpu0` | `.../eval/20260512_181016_ne006p_w4a4_g4_stage4_conv2d_grid478_seed20260507` | 6 |
| NE006q | `.../quant/20260512_172203_ne006q_w4a4_g4_stage5_conv2d_a5000_1024cali_gpu3_rerun` | `.../eval/20260512_181016_ne006q_w4a4_g4_stage5_conv2d_grid478_seed20260507` | 6 |

验证结果：5 个 final checkpoint 均 `passed=true`，`weight_quant=true`，`act_quant=true`，`n_bits_a=4`，`activation_delta_count=52`，`initialized_activation_quantizers=52`，`level_offender_count=0`。activation-only metrics 中 `non_positive_delta_count=0`。每个 eval 的 `per_sample_metrics.jsonl` 都是 `11950` 行。

overall full-grid 结果：

| variant | selected | pre SNR mean/median | final SNR mean/median | final SSIM mean/median | recon gain SNR/SSIM | vs NE000_2 W4A4 | gap to NE006l | gap to E007 W4A32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NE000_2 baseline | 51 | 11.1727 / 11.1577 | 12.9150 / 13.1180 | 0.939563 / 0.954078 | +1.7422 / -0.001929 | +0.0000 / +0.000000 | -0.8177 / -0.005740 | -4.8706 / -0.024574 |
| NE006l all Conv2d except head | 30 | 12.6993 / 12.8885 | 13.7326 / 13.9838 | 0.945303 / 0.960038 | +1.0334 / +0.003626 | +0.8177 / +0.005740 | +0.0000 / +0.000000 | -4.0529 / -0.018834 |
| NE006m stage1 Conv2d g4 | 6 | 11.5111 / 11.5033 | 13.1610 / 13.3724 | 0.940107 / 0.954443 | +1.6499 / +0.000399 | +0.2460 / +0.000544 | -0.5716 / -0.005195 | -4.6246 / -0.024030 |
| NE006n stage2 Conv2d g4 | 6 | 11.0960 / 11.0894 | 12.8185 / 13.0006 | 0.939369 / 0.953903 | +1.7225 / -0.000123 | -0.0965 / -0.000194 | -0.9141 / -0.005934 | -4.9671 / -0.024768 |
| NE006o stage3 Conv2d g4 | 6 | 11.1805 / 11.1818 | 12.8957 / 13.0939 | 0.939752 / 0.954255 | +1.7152 / -0.001827 | -0.0193 / +0.000189 | -0.8369 / -0.005551 | -4.8899 / -0.024385 |
| NE006p stage4 Conv2d g4 | 6 | 11.6581 / 11.7158 | 13.1833 / 13.4054 | 0.940243 / 0.954534 | +1.5252 / +0.000364 | +0.2683 / +0.000680 | -0.5493 / -0.005060 | -4.6023 / -0.023895 |
| NE006q stage5 Conv2d g4 | 6 | 11.6677 / 11.7584 | 13.2033 / 13.4520 | 0.943476 / 0.958300 | +1.5356 / +0.007208 | +0.2883 / +0.003913 | -0.5294 / -0.001827 | -4.5823 / -0.020661 |

single-sample sanity 只用于确认运行，不作为策略判断：

| variant | single pre SNR | single final SNR |
|---|---:|---:|
| NE006m stage1 | 13.2654 | 13.3391 |
| NE006n stage2 | 13.1999 | 13.2849 |
| NE006o stage3 | 13.2277 | 13.2529 |
| NE006p stage4 | 13.1922 | 13.2534 |
| NE006q stage5 | 13.0585 | 13.2928 |

by-source 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | Anisotropic | Kerry3D | Shots0001 |
|---|---:|---:|---:|
| NE006m stage1 | 13.7274 / +0.4472 | 9.0688 / +0.0356 | 13.2204 / +0.2158 |
| NE006n stage2 | 13.1214 / -0.1589 | 9.0472 / +0.0140 | 12.9157 / -0.0889 |
| NE006o stage3 | 13.2403 / -0.0399 | 9.0576 / +0.0244 | 12.9876 / -0.0171 |
| NE006p stage4 | 13.7863 / +0.5061 | 9.0643 / +0.0311 | 13.2367 / +0.2321 |
| NE006q stage5 | 13.6798 / +0.3996 | 9.0883 / +0.0552 | 13.2811 / +0.2764 |

by-input-SNR 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | -2 dB | -1 dB | 1 dB | 5 dB | 10 dB |
|---|---:|---:|---:|---:|---:|
| NE006m stage1 | 11.8556 / +0.2052 | 12.2781 / +0.2206 | 12.9693 / +0.2441 | 13.9681 / +0.2711 | 14.7339 / +0.2893 |
| NE006n stage2 | 11.5924 / -0.0580 | 11.9858 / -0.0716 | 12.6333 / -0.0918 | 13.5742 / -0.1229 | 14.3067 / -0.1380 |
| NE006o stage3 | 11.6430 / -0.0075 | 12.0441 / -0.0134 | 12.7083 / -0.0169 | 13.6710 / -0.0261 | 14.4122 / -0.0324 |
| NE006p stage4 | 11.8864 / +0.2359 | 12.3045 / +0.2470 | 12.9879 / +0.2628 | 13.9860 / +0.2889 | 14.7518 / +0.3071 |
| NE006q stage5 | 11.8788 / +0.2283 | 12.3030 / +0.2456 | 13.0011 / +0.2759 | 14.0238 / +0.3267 | 14.8096 / +0.3650 |

by-missing-rate 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | 0.02 | 0.08 | 0.18 | 0.28 | 0.38 |
|---|---:|---:|---:|---:|---:|
| NE006m stage1 | 14.2050 / +0.2805 | 13.8991 / +0.2785 | 13.3185 / +0.2584 | 12.6297 / +0.2269 | 11.7528 / +0.1860 |
| NE006n stage2 | 13.8302 / -0.0944 | 13.5176 / -0.1030 | 12.9522 / -0.1079 | 12.3042 / -0.0986 | 11.4883 / -0.0784 |
| NE006o stage3 | 13.9245 / -0.0000 | 13.6138 / -0.0068 | 13.0433 / -0.0168 | 12.3758 / -0.0270 | 11.5211 / -0.0456 |
| NE006p stage4 | 14.1806 / +0.2561 | 13.8856 / +0.2650 | 13.3393 / +0.2791 | 12.6788 / +0.2760 | 11.8323 / +0.2655 |
| NE006q stage5 | 14.2333 / +0.3088 | 13.9186 / +0.2980 | 13.3463 / +0.2862 | 12.6797 / +0.2769 | 11.8385 / +0.2717 |

人类可读结论：

1. 单个 stage 不能解释 `all Conv2d except head` 的收益。最强单 stage 是 NE006q stage5，仅 `+0.2883 dB`；NE006l 是 `+0.8177 dB`，说明收益来自跨 stage 叠加，而不是单 stage 主导。
2. stage5、stage4、stage1 是正收益 stage。排序为 stage5 `+0.2883 dB`，stage4 `+0.2683 dB`，stage1 `+0.2460 dB`。
3. stage2 和 stage3 不应作为 g4 主策略对象。stage2 相对 baseline 为 `-0.0965 dB`，stage3 为 `-0.0193 dB`，都没有正向 full-grid 价值。
4. by-source 显示 stage1/4/5 的主要收益来自 Anisotropic 与 Shots0001，Kerry3D 始终只有很小提升。stage4 在 Anisotropic 上最高 `+0.5061 dB`，stage5 在 Shots0001 上最高 `+0.2764 dB`。
5. by-input-SNR 显示 stage1/4/5 在高输入 SNR 条件收益更大，尤其 stage5 从 `-2 dB` 的 `+0.2283 dB` 增至 `10 dB` 的 `+0.3650 dB`。
6. by-missing-rate 显示 stage5/stage4 的收益相对稳定；stage1 随 missing rate 增大收益下降更明显。

对后续实验的影响：

- 继续做纯 g4 granularity 的边际价值下降。stage-level 已证明单 stage 不够强，组合收益来自多 stage 叠加。
- 若仍在 NE006 内收尾，唯一值得补的最小组合是 `stage1 + stage4 + stage5 Conv2d g4`，因为这三个 stage 是正收益来源，selected count 18，可能接近 NE006l 的 30 个 selected count 效果。
- 更建议进入 NE007：以 W4A4 为主对象做 selective A8 / mixed precision。优先候选为 stage5、stage4、stage1 的 Conv2d，或者以 NE006l `all Conv2d except head g4` 为基础，将 residual 最大的 stage/role 升 A8。
- 不建议继续研究 stage2/stage3 g4，也不建议回到 split/merge、head 或 range/clipping。
