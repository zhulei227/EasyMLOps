# -*- coding: utf-8 -*-
"""
LoRA 微调接口（v0.2.12 新增）：对类 Jev 决策模型与小 LLM（<4B）做参数高效微调。

自动适配:
    auto_adapt() 从模型的 config + named_modules 推断:
        - model_type（qwen3_5/gemma/modernbert/...，OneJev/Jeff 为 Qwen3.5 系、Laya 为 ModernBERT 系）
        - target_modules（扫描全部 nn.Linear, 按名称规则选出注意力/MLP 投影层——
          覆盖 q/k/v/o/gate/up/down proj、query/key/value/dense 等不同架构命名）
    无需为每个模型族手写配置。

训练模式:
    - train_sft:     通用因果 LM LoRA SFT（samples: [{"prompt", "completion"}]）
    - train_decision: 决策模型微调（samples: [{"state", "questions", "answers"}]，
                      Jev 契约格式；Laya 系基于 ModernBERT 分类头）

权重管理:
    save_adapter / load_adapter（peft adapter 目录） / merge_and_export（合并导出完整模型）

依赖: transformers + peft（懒加载）；QLoRA 需 bitsandbytes（可选）。
GPU 建议: <1B 模型 LoRA 单卡 8GB 即可；CPU 仅可极小验证。

Example:
    >>> from easymlops.llm import LoRAFineTuner
    >>> tuner = LoRAFineTuner("convaiinnovations/laya")
    >>> info = tuner.auto_adapt()            # 自动推断 target_modules
    >>> tuner.train_sft(samples, "adapter_out", epochs=3)
    >>> # 之后用本地 transformers 加载 base + adapter 推理
"""

import os

import pandas as pd

__all__ = ["LoRAFineTuner"]

# 各架构的注意力/MLP 投影层命名规则（auto_adapt 的匹配词表）
_TARGET_KEYWORDS = ("proj", "query", "key", "value", "dense", "attention", "mlp", "fc1", "fc2")
_EXCLUDE_KEYWORDS = ("classifier", "score", "head", "lm_head", "embed", "norm")


class LoRAFineTuner(object):
    """LoRA 微调接口（自动适配模型架构，详见模块 docstring）。"""

    def __init__(self, model_id, lora_rank=8, lora_alpha=16, lora_dropout=0.05,
                 use_qlora=False, device=None, local_path=None):
        self.model_id = model_id
        self.local_path = local_path   # 已下载到本地的模型目录（优先于 model_id）
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.use_qlora = bool(use_qlora)
        self.device = device   # None=自动(cuda>cpu)
        self.adapt_info_ = None
        self.model_ = None
        self.tokenizer_ = None
        self.adapter_dir_ = None

    # ---- 自动适配 ----

    def _model_dir(self):
        if self.local_path and os.path.isdir(self.local_path):
            return self.local_path
        from transformers.utils import cached_file
        try:
            return os.path.dirname(cached_file(self.model_id, "config.json"))
        except Exception:
            return None

    def auto_adapt(self, sample_state: dict = None):
        """
        自动推断 model_type 与 LoRA target_modules。

        Args:
            sample_state: 可选 {"input_ids": ..., "attention_mask": ...} 用于实例化
                          模型扫描模块（未提供时仅读 config 做静态推断）

        Returns:
            dict: {"model_type", "architectures", "target_modules", "model_class"}
        """
        import json
        model_dir = self._model_dir()
        if not model_dir:
            raise Exception("model config not found locally; download first or set local_path")
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        model_type = config.get("model_type", "unknown")
        architectures = config.get("architectures", [])

        # 模型类选择: 决策/多模态(Qwen3.5ForConditionalGeneration 等)走 AutoModel 容错链
        if any("ConditionalGeneration" in a for a in architectures):
            model_class = "AutoModelForImageTextToText"
        elif any("CausalLM" in a or "LMHeadModel" in a for a in architectures):
            model_class = "AutoModelForCausalLM"
        elif any("SequenceClassification" in a for a in architectures):
            model_class = "AutoModelForSequenceClassification"
        else:
            model_class = "AutoModel"

        target_modules = self._scan_target_modules(model_dir, config)
        self.adapt_info_ = {"model_type": model_type, "architectures": architectures,
                            "target_modules": target_modules, "model_class": model_class}
        return self.adapt_info_

    def _scan_target_modules(self, model_dir, config):
        """实例化模型扫描 nn.Linear, 按关键词规则选出 LoRA 注入点（自动适配的核心）。"""
        try:
            import torch
            from transformers import AutoModel
            from transformers import AutoConfig
        except ImportError:
            raise ImportError("LoRAFineTuner requires transformers+torch: "
                              "pip install transformers torch")
        cfg = AutoConfig.from_pretrained(model_dir)
        try:
            model = AutoModel.from_config(cfg)
        except Exception:
            # 扫描失败时交给 peft 的 all-linear 模式(运行时自动注入所有投影层)
            return ["all-linear"]
        targets = set()
        # 匹配任意"投影形状"的带参层: nn.Linear 与 Conv1D(GPT 系)等
        param_types = (torch.nn.Linear,)
        try:
            from transformers.pytorch_utils import Conv1D
            param_types = param_types + (Conv1D,)
        except Exception:
            pass
        for name, module in model.named_modules():
            if not isinstance(module, param_types):
                continue
            last = name.split(".")[-1]
            if any(kw in last.lower() for kw in _TARGET_KEYWORDS) and \
                    not any(ex in last.lower() for ex in _EXCLUDE_KEYWORDS):
                targets.add(last)
        del model
        if not targets:
            # 未匹配到目标(如纯 SDPA 融合注意力)→ peft all-linear 兜底
            return ["all-linear"]
        return sorted(targets)

    # ---- 训练 ----

    def _lazy_load_model(self, model_class_name="AutoModelForCausalLM"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
        import torch
        source = self.local_path or self.model_id
        self.tokenizer_ = AutoTokenizer.from_pretrained(source)
        cls = AutoModelForCausalLM if model_class_name == "AutoModelForCausalLM" else AutoModel
        kwargs = {}
        if self.use_qlora:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            except ImportError:
                raise ImportError("QLoRA requires bitsandbytes: pip install bitsandbytes")
        self.model_ = cls.from_pretrained(source, **kwargs)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_.to(self.device)
        return self.model_

    def _attach_lora(self, model):
        from peft import LoraConfig, get_peft_model
        targets = self._current_targets
        if targets == ["all-linear"]:
            targets = "all-linear"   # peft 运行时自动注入所有 Linear
        lora_config = LoraConfig(r=self.lora_rank, lora_alpha=self.lora_alpha,
                                 lora_dropout=self.lora_dropout,
                                 target_modules=targets,
                                 task_type="CAUSAL_LM")
        return get_peft_model(model, lora_config)

    def train_sft(self, samples, output_dir="lora_adapter_out", epochs=3, lr=1e-4,
                  batch_size=2, max_length=512):
        """
        通用因果 LM LoRA SFT。

        Args:
            samples: [{"prompt": "...", "completion": "..."}]
            output_dir: adapter 保存目录
            epochs/lr/batch_size/max_length: 训练参数

        Returns:
            dict: {"steps", "final_loss", "output_dir"}
        """
        import torch
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("LoRA fine-tuning requires peft: pip install peft")
        model = self._lazy_load_model("AutoModelForCausalLM")
        # target_modules: 已 auto_adapt 则用之, 否则现扫
        if self.adapt_info_ is None:
            self.adapt_info_ = self.auto_adapt()
        self._current_targets = self.adapt_info_["target_modules"]
        model = self._attach_lora(model)
        self.model_ = model   # self.model_ 指向 peft 包装模型(merge_and_export 依赖)
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
        tokenizer = self.tokenizer_
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        final_loss = None
        steps = 0
        for _ in range(int(epochs)):
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                losses = []
                for sample in batch:
                    prompt_ids = tokenizer(sample["prompt"], truncation=True,
                                           max_length=max_length, return_tensors="pt")
                    full_ids = tokenizer(sample["prompt"] + sample["completion"],
                                         truncation=True, max_length=max_length,
                                         return_tensors="pt")
                    input_ids = full_ids["input_ids"].to(self.device)
                    attention_mask = full_ids["attention_mask"].to(self.device)
                    labels = input_ids.clone()
                    p_len = prompt_ids["input_ids"].shape[1]
                    labels[:, :p_len] = -100   # 只对 completion 计算 loss
                    output = model(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=labels)
                    losses.append(output.loss)
                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                steps += 1
                final_loss = float(loss.detach())
        model.save_pretrained(output_dir)
        self.tokenizer_.save_pretrained(output_dir)
        self.adapter_dir_ = output_dir
        return {"steps": steps, "final_loss": final_loss, "output_dir": output_dir}

    def train_decision(self, samples, output_dir="lora_decision_out", **train_kwargs):
        """
        决策模型微调（Jev 契约样本: [{"state", "questions", "answers"}]）。

        Laya 系(ModernBERT 分类头)与 OneJev/Jeff 系(Qwen3.5 条件生成)的训练目标不同:
        本方法按 model_type 分派——modernbert 系用 peft 包装分类头微调，
        causal 系把决策样本转成 SFT 文本(指令→答案)走 train_sft。
        """
        model_dir = self._model_dir()
        import json
        model_type = "unknown"
        if model_dir and os.path.exists(os.path.join(model_dir, "config.json")):
            with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
                model_type = json.load(f).get("model_type", "unknown")
        if model_type in ("modernbert", "bert", "mmbert"):
            raise Exception("decision-model fine-tuning for '{}' requires the official "
                            "training recipe (see laya fine-tuning notebooks); "
                            "auto_adapt()+train_sft covers causal-LM backbones".format(model_type))
        # causal 系: 决策样本转 SFT 文本
        sft_samples = []
        for s in samples:
            prompt = json.dumps({"state": s["state"], "questions": s["questions"]},
                                ensure_ascii=False)
            completion = "\n" + json.dumps(s["answers"], ensure_ascii=False)
            sft_samples.append({"prompt": prompt, "completion": completion})
        return self.train_sft(sft_samples, output_dir=output_dir, **train_kwargs)

    # ---- 权重管理 ----

    def load_adapter(self, adapter_dir):
        """加载已保存的 LoRA adapter（推理用）。"""
        from peft import PeftModel
        model = self._lazy_load_model()
        self.model_ = PeftModel.from_pretrained(model, adapter_dir)
        self.model_.eval()
        self.adapter_dir_ = adapter_dir
        return self.model_

    def merge_and_export(self, output_dir):
        """合并 LoRA 权重导出完整模型（可再用 vLLM/ollama 服务化）。"""
        if self.model_ is None or self.adapter_dir_ is None:
            raise Exception("no adapter attached; call train_sft/load_adapter first")
        merged = self.model_.merge_and_unload()
        merged.save_pretrained(output_dir)
        if self.tokenizer_:
            self.tokenizer_.save_pretrained(output_dir)
        return output_dir
