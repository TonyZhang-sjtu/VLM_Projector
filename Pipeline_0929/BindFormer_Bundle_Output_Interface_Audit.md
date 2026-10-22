# BindFormer 四角色输出接口：当前结构核查与修改建议

## 核查结论

当前最终 LN/MLP 没有跨 token 混合。slot 内四角色的连续排列一直保持到 Llama 输入；没有发现 flatten、拼接或输入替换打乱分组。问题是四角色内容趋同，以及 Decoder 未接收显式 slot/role 元数据。不能将这两点解释为“MLP 把所有 token 混到一起”。

本次检查实际 checkpoint 的 head 权重与已有真实病例输出，未修改生产模型、未训练新方案。GPU 仅使用物理 3 号卡。

## 1. 最终 token 的实际布局

`src/models/projectors/bindformer/projector.py` 的路径：

```python
roles = self.interactions(roles, global_tokens)  # [K, 4, 768]
tokens = torch.cat((global_tokens, roles.flatten(0, 1)), dim=0)
output = self.head(self.final_norm(tokens))     # [G + 4*K, 4096]
```

当前 16×：G=137、K=103、local=412、总视觉 token=549。顺序为：

```text
[g0 ... g136,
 F0 L0 S0 M0,
 F1 L1 S1 M1,
 ...,
 F102 L102 S102 M102]
```

role 顺序为 finding、location_laterality、signal、morphology_extent。视觉序列内索引为 `G + 4*k + r`。flatten 只把两条索引轴合并成一条，不计算均值或混合特征，随时可以 reshape 回 `[103,4,4096]`。

`mapping_head` 是 `Linear(768,4096) → GELU → Linear(4096,4096)`。Linear 仅作用于最后一维；final LayerNorm 也仅归一化最后的 768 个通道。所有 token 共享参数，但不读取其他 token 的数值。因此 `z[i] = f(x[i])`，不存在 head 内跨 token 项 `x[j] → z[i] (j≠i)`。

共享映射可能把不同输入映射到接近的输出方向，也可能不利于不同角色的专门化；这与跨 token 聚合是不同机制。

## 2. 实际检查

脚本：[audit_bundle_interface.py](../results/BindFormer_decoder_sensitivity_r16/audit_bundle_interface.py)

原始结果：[bundle_interface_audit.json](../results/BindFormer_decoder_sensitivity_r16/bundle_interface_audit.json)

| 检查 | 固定区域 VA | 固定区域 SA+CA |
| --- | --- | --- |
| `[K,4,d]` 调用相同 head 与 flatten 调用的最大误差 | 6.22e-15 | 7.11e-15 |
| 改变一个输入 token 后其他输出变化 | 0 | 0 |
| 单个输出对其他输入 token 的梯度 | 0 | 0 |
| 输入排列后输出对应排列的最大误差 | 0 | 0 |
| 18 个真实输出 × 训练/生成 prepare 的原值与顺序核查 | 全部通过 | 全部通过 |

head 检查使用 checkpoint 权重、合成输入与 FP64，验证算子性质，不是新的语义指标。接口检查使用之前保存的 native BF16 患者视觉输出、真实 LlamaDecoder 和 tokenizer；没有执行新 LLM 推理或训练。

## 3. 哪些结构保留，哪些没有显式传入

| 层面 | 当前状态 |
| --- | --- |
| 区域读取范围 | 固定区域 Reader 直接读取范围受约束 |
| 四角色连续排列 | 保留 |
| 不同区域输出差异 | 已较原版提高，见前次诊断 |
| 同一区域四角色内容区分 | 很弱：最终 FP32 role 离散度约 4.07e-6 / 1.65e-6 |
| slot ID、role ID、区域坐标作为独立 Decoder 输入 | 没有 |
| 每组四 token 的分组 attention mask | 没有 |
| 结构相关信息是否可能隐含在向量中 | 可能，需属性读出及干预评估，不能仅由无显式元数据否定 |

`MultimodalModel` 只把 `ProjectedVolume.embeddings` 交给 decoder.prepare；metadata/auxiliary 不进入 LLM。prepare 把这 549 个向量原样替换到 `<image>` 位置，使用普通连续 position IDs 和 padding mask。安装环境中的 LlamaModel 随后使用 RoPE 与普通因果 self-attention，没有每四个一组的特殊语义掩码。

这意味着后面的视觉 token 可读取此前 global/其他 slot；第一个 role 不能读取本组排在它后面的 role，而后面的报告 token 可以读取整个视觉前缀。这是当前模型的普通因果结构，不能单独据此认定它导致了临床退化。

## 4. 是否必须保持结构

需要区分“以生成准确报告为目标”和“验证四角色绑定设计”。报告生成不要求所有内部层都保持四角色互相隔离；跨区域上下文对跨区病灶和占位关系可能有用。最终向量也需要适应 LLM，不能要求和上游原始视觉空间保持同样的余弦几何。

如果要证明 BindFormer 的四角色设计有效，那么在 projector 输出接口应有可检验的区域归属与角色任务作用。顺序正确是必要的工程属性，却不足以证明 finding/location/signal/morphology 的语义已经成立。当前证据更接近“保留了四 token 外形，角色内容趋同”。

四向量相同也不等于所有临床属性都已经丢失：共同向量可能同时承载多种属性，不同读出器可能提取不同内容。需要用训练/验证分离的属性探针，分别检查 Reader 输出、最终输出和 LLM 读出，才能定位实际的信息损失。

## 5. 四个一组 MLP：三种不同实现

| 方案 | 实际变化 | 判断 |
| --- | --- | --- |
| reshape 为 `[K,4,d]` 后用相同 MLP | 无数学功能变化 | 不足以解决问题，已通过 checkpoint 算子检查 |
| 每个 role 使用自己的 head/adapter，跨 slot 共享 | `z[k,r]=f_r(b[k,r])` | 可保留角色专门化接口；仅靠不同参数仍可能只产生固定角色偏移 |
| 同槽四向量拼接后一起映射，再拆回四向量 | `Z[k]=reshape(F_bundle(vec(B[k])),[4,D])` | 可行；新增同槽交互且不跨 slot，但也可能把四角色进一步平均化，需要残差和任务验证 |

第三种方案应输出四个 token，保持 G+4K 的预算。global 分支单独调用原 head，不能把 global 拼入每个 slot 的 bundle 输入，否则实验含义又改变。

直接使用 `3072 → 4096 → 16384` 的 bundle MLP 会显著增加参数。更建议保留已对齐的共享 head，通过轻量角色 adapter 和可选同槽残差实现：

```text
B[k] = [b_F, b_L, b_S, b_M]                 # [4,768]
x[k,r] = final_LN(b[k,r])
u[k] = small_bundle_MLP(concat_r x[k,r])    # 例如 256 维
z[k,r] = shared_head(x[k,r])
         + role_adapter[r](x[k,r])
         + gate[r] * context_projection[r](u[k])
```

role adapter 可用低秩形式；bundle context 仅取同一 slot，不读其他 slot。初始化时可将新增残差输出设为零，以复用现有 checkpoint，并分别记录每条分支的幅度和梯度。不同 role head、bundle context 都应独立开关，不能把“多参数”“角色头”“同槽交互”三个变化合并归因。

上述属于待验证方案；它没有在本次实现或训练，也不保证语义分工自动出现。

## 6. 优先顺序与验证方式

1. **先处理上游 query。** 实测共同 slot 项范数约 79/121，role 项约 1，概念项约 .43。对三分支分别归一化后再组合，观察 Reader 前后的 role 差异。它与既有 R1 的 router value normalization 是不同位置的开关。不能仅修改 final MLP，期望它恢复已被 Reader 抹去的区别。
2. **比较共享 head、角色 adapter、再加同槽 context。** 先固定其他设置、相同 token 预算与训练预算，分别衡量结构作用。不要只凭 head 输出间距离增加判定有效。
3. **监督角色可读出的视觉事实。** 最终 location、signal、morphology、finding 输出应分别能支持相应任务；有区域/病灶对应标签时在对应 slot 上监督，只有报告级标签时需合理的多实例对齐。不能给所有正常 slot 强行贴上同一个病灶标签。
4. **区域内容重建与角色任务分开。** 本区域 final bundle 重建本区域特征能约束内容保留，但四角色都复制同一信息也能完成重建，因此它不是角色专门化的充分条件。
5. **验证实际作用。** 在患者独立验证集上比较属性读出、对应 role/区域替换后的事实变化及完整报告准确性。删除固定 role bias 后仍应有患者相关属性；交换某角色不应只是随机改变措辞。真实属性存在关联，不要求 role 绝对排他或彼此正交。

只有确认最终输出含有可读出的临床属性后，再比较冻结 LLM 与 LLM LoRA 的读出收益，才能把 projector 的问题与 Decoder 的适配问题区分开。
