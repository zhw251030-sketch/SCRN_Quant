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

## 实验索引

| 实验编号 | 日期 | 目的 | 主要改动或配置 | 关键结果 | 结论 |
| --- | --- | --- | --- | --- | --- |
| E000 | 2026-05-04 | 建立激活量化研究日志 | 新增本文档 | 不涉及实验 | 后续实验从 E001 开始记录 |

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
