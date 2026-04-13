#!/usr/bin/env python3
"""
YOLO Model Analyzer - 模型分析工具 V4
分析每一层的:
- 所有输入的 shape 和数据大小
- 输出的 shape 和数据大小
- 权重参数大小
- 当前内存中真正缓存的激活数据（残差等结构造成）

核心改进：实现真正的数据流追踪，当激活被 concat 消费后从缓存移除
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from collections import OrderedDict

# 项目根目录
ULTRALYTICS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LayerAnalyzerV4:
    """层分析器 V4 - 实现真正的数据流追踪"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_info = []
        self.handles = []
        self.activation_cache = OrderedDict()  # 当前内存中的激活
        self.consumed_activations = set()  # 已被消费的激活 ID
        self.tensor_to_cache_key = {}  # 输入tensor ID -> 缓存key的反向映射，用于快速查找待移除的缓存项

    def _get_tensor_id(self, tensor: torch.Tensor) -> int:
        """获取 tensor 的唯一 ID"""
        return id(tensor)

    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        if tensor is None:
            return 0
        return tensor.element_size() * tensor.nelement()

    def _format_size(self, size_bytes: int) -> str:
        if size_bytes == 0:
            return "0 B"
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    def _register_hooks(self):
        self.handles = []

        def get_forward_pre_hook(name):
            def forward_pre_hook(module, input):
                # 获取当前缓存中真正存活的激活（未被消费的）
                live_cache = []
                total_cache_size = 0
                for cache_name, (cache_tensor, tensor_id) in self.activation_cache.items():
                    if tensor_id not in self.consumed_activations:
                        if cache_tensor is not None and isinstance(cache_tensor, torch.Tensor):
                            size = self._get_tensor_size(cache_tensor)
                            live_cache.append({
                                'name': cache_name,
                                'shape': list(cache_tensor.shape),
                                'size': size,
                                'tensor_id': tensor_id
                            })
                            total_cache_size += size

                # 收集所有输入信息
                input_list = []
                input_tensor_ids = []
                total_input_size = 0
                if input and isinstance(input, (tuple, list)):
                    for idx, inp in enumerate(input):
                        if isinstance(inp, torch.Tensor):
                            size = self._get_tensor_size(inp)
                            tensor_id = self._get_tensor_id(inp)
                            input_list.append({
                                'index': idx,
                                'shape': list(inp.shape),
                                'size': size,
                                'tensor_id': tensor_id
                            })
                            input_tensor_ids.append(tensor_id)
                            total_input_size += size

                # 构建反向映射：输入tensor ID -> 对应的缓存key
                # 这样在forward_hook中可以快速找到并移除已消费的缓存项
                for cache_name, (cache_tensor, tensor_id) in self.activation_cache.items():
                    if tensor_id not in self.tensor_to_cache_key:
                        self.tensor_to_cache_key[tensor_id] = cache_name

                self.layer_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'pre_forward': True,
                    'input_list': input_list,
                    'input_tensor_ids': input_tensor_ids,
                    'total_input_size': total_input_size,
                    'cache_info': live_cache,
                    'total_cache_size': total_cache_size,
                })
            return forward_pre_hook

        def get_forward_hook(layer_name):
            def forward_hook(module, input, output):
                module_type = type(module).__name__

                # 收集输出信息
                output_list = []
                total_output_size = 0
                if output is not None:
                    if isinstance(output, torch.Tensor):
                        output_list.append({
                            'shape': list(output.shape),
                            'size': self._get_tensor_size(output),
                            'tensor': output
                        })
                        total_output_size += self._get_tensor_size(output)
                    elif isinstance(output, (tuple, list)):
                        for idx, out in enumerate(output):
                            if isinstance(out, torch.Tensor):
                                output_list.append({
                                    'index': idx,
                                    'shape': list(out.shape),
                                    'size': self._get_tensor_size(out),
                                    'tensor': out
                                })
                                total_output_size += self._get_tensor_size(out)

                # 收集参数信息（展开所有子模块的权重）
                param_memory = 0
                param_count = 0
                param_shapes = []
                weight_details = []
                for child_name, child in module.named_children():
                    child_params = sum(p.numel() for p in child.parameters())
                    if child_params > 0:
                        for p in child.parameters():
                            param_memory += self._get_tensor_size(p)
                            param_count += p.numel()
                            weight_details.append(f"{child_name}:{list(p.shape)}")
                if not weight_details:
                    for param in module.parameters():
                        param_memory += self._get_tensor_size(param)
                        param_count += param.numel()
                        param_shapes.append(list(param.shape))

                # 收集当前层的输入 tensor_id（用于判断是否被消费）
                input_tensor_ids = []
                if input and isinstance(input, (tuple, list)):
                    for inp in input:
                        if isinstance(inp, torch.Tensor):
                            input_tensor_ids.append(self._get_tensor_id(inp))

                # 判断是否有残差连接（同一个输入被多次使用）
                has_residual = len(input_tensor_ids) > 0 and len(set(input_tensor_ids)) < len(input_tensor_ids)

                # 判断是否是 concat 或多输入操作
                is_concat = len(input_tensor_ids) >= 2

                # 将输出添加到缓存（只有残差连接时才保留输入的引用）
                if output_list:
                    for idx, out_info in enumerate(output_list):
                        out_tensor = out_info['tensor']
                        tensor_id = self._get_tensor_id(out_tensor)
                        cache_name = f"{layer_name}_out" if len(output_list) == 1 else f"{layer_name}_out_{idx}"
                        self.activation_cache[cache_name] = (out_tensor, tensor_id)

                # 如果是 concat 或多输入操作，消费所有输入
                # 如果没有残差连接，输入应该被消费
                if is_concat or (len(input_tensor_ids) > 0 and not has_residual):
                    for tid in input_tensor_ids:
                        # 不立即标记为消费，而是等待确认
                        pass

                # 关键改进：只有当前层的输入被后续层使用时，才标记为消费
                # 这里简单处理：如果没有残差连接，则输入会被"消费"
                # 同时从activation_cache中移除被消费的缓存项，避免历史缓存累积
                if len(input_tensor_ids) > 0 and not has_residual:
                    for tid in input_tensor_ids:
                        self.consumed_activations.add(tid)
                        # 通过反向映射找到该输入对应的缓存项并移除
                        if tid in self.tensor_to_cache_key:
                            cache_key = self.tensor_to_cache_key.pop(tid)
                            if cache_key in self.activation_cache:
                                del self.activation_cache[cache_key]

                # 更新层信息
                for i in range(len(self.layer_info) - 1, -1, -1):
                    if self.layer_info[i]['name'] == layer_name and self.layer_info[i].get('pre_forward'):
                        self.layer_info[i].update({
                            'output_list': [{k: v for k, v in o.items() if k != 'tensor'} for o in output_list],
                            'total_output_size': total_output_size,
                            'param_memory': param_memory,
                            'param_count': param_count,
                            'param_shapes': param_shapes,
                            'weight_details': weight_details,
                            'is_concat': is_concat,
                            'has_residual': has_residual,
                            'consumed_inputs': input_tensor_ids if not has_residual else [],
                            'pre_forward': False,
                        })
                        break

            return forward_hook

        for name, module in self.model.named_modules():
            handle = module.register_forward_pre_hook(get_forward_pre_hook(name))
            self.handles.append(handle)
            handle = module.register_forward_hook(get_forward_hook(name))
            self.handles.append(handle)

    def analyze(self, input_size: Tuple = (1, 3, 640, 640)) -> List[Dict]:
        self.layer_info = []
        self.activation_cache = OrderedDict()
        self.consumed_activations = set()
        self.tensor_to_cache_key = {}

        dummy_input = torch.randn(input_size)
        self._register_hooks()

        self.model.eval()
        with torch.no_grad():
            self.model(dummy_input)

        for handle in self.handles:
            handle.remove()

        results = []
        for info in self.layer_info:
            if not info.get('pre_forward'):
                results.append(info)

        return results

    def print_summary(self, results: List[Dict]):
        """打印分析结果"""
        print("\n" + "=" * 280)
        print(f"{'Layer Name':<45} {'Type':<18} {'Inputs (所有输入)':<80} {'Output':<45} {'Weight Shapes':<60} {'Params':<12}")
        print("=" * 280)

        total_param_memory = 0
        total_activation_memory = 0

        for info in results:
            name = info['name'][:44]
            layer_type = info['type'][:17]

            # Inputs
            if info.get('input_list'):
                input_parts = []
                for inp in info['input_list']:
                    input_parts.append(f"in{inp['index']}:{inp['shape']}({self._format_size(inp['size'])})")
                input_str = " | ".join(input_parts)
                if len(input_str) > 78:
                    input_str = input_str[:75] + "..."
            else:
                input_str = "-"

            # Output
            if info.get('output_list'):
                if len(info['output_list']) == 1:
                    out = info['output_list'][0]
                    output_str = f"{out['shape']} ({self._format_size(out['size'])})"
                else:
                    output_str = f"tuple[{len(info['output_list'])}]"
                if len(output_str) > 43:
                    output_str = output_str[:40] + "..."
            else:
                output_str = "-"

            # Weight shapes
            if info.get('weight_details'):
                weight_str = "; ".join(info['weight_details'])
            elif info.get('param_shapes'):
                weight_str = "; ".join(str(s) for s in info['param_shapes'])
            else:
                weight_str = '-'
            if len(weight_str) > 55:
                weight_str = weight_str[:52] + "..."

            # Params
            param_count = f"{info['param_count']:,}" if info['param_count'] > 0 else '-'

            # 标记 concat 或残差
            marker = ""
            if info.get('is_concat'):
                marker = "[CONCAT]"
            elif info.get('has_residual'):
                marker = "[RESIDUAL]"

            print(f"{name:<45} {layer_type:<18} {input_str:<80} {output_str:<45} {weight_str:<60} {param_count:<12} {marker}")

            total_param_memory += info['param_memory']
            total_activation_memory += info['total_output_size']

        print("=" * 280)
        print(f"{'Total Params Memory:':<100} {self._format_size(total_param_memory)}")
        print(f"{'Total Activation Memory (outputs):':<100} {self._format_size(total_activation_memory)}")
        print(f"{'Estimated Total Memory (params + activations):':<100} {self._format_size(total_param_memory + total_activation_memory)}")
        print("=" * 280)

    def print_cache_analysis(self, results: List[Dict]):
        """打印真正存活的激活缓存分析"""
        print("\n\n" + "=" * 300)
        print("Activation Cache Analysis - 当前层执行时内存中真正存活的激活数据（残差等结构造成）")
        print("=" * 300)

        for idx, info in enumerate(results[:60]):
            name = info['name']
            layer_type = info['type']
            is_concat = info.get('is_concat', False)
            has_residual = info.get('has_residual', False)

            # 标题行
            marker = ""
            if is_concat:
                marker = "[CONCAT - 消费输入]"
            elif has_residual:
                marker = "[RESIDUAL - 保留输入]"
            print(f"\n[{idx+1}] {name} ({layer_type}) {marker}")

            # 当前存活的缓存
            if info.get('cache_info'):
                total_cache = 0
                for cache in info['cache_info']:
                    size = cache['size']
                    total_cache += size
                    cache_name = cache['name']
                    cache_shape = str(cache['shape'])
                    print(f"    {cache_name}: {cache_shape} ({self._format_size(size)})")
                print(f"    --- Live Cache Total: {self._format_size(total_cache)}")
            else:
                print(f"    (无存活缓存)")

            # 如果被消费了，显示消费了哪些
            if info.get('consumed_inputs'):
                consumed_str = ", ".join(str(tid)[:8] for tid in info['consumed_inputs'][:5])
                print(f"    [Consumed inputs: {consumed_str}...]")

        print("\n" + "-" * 300)


class ConcatDetector:
    """专门检测和分析 Concat 算子"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.concat_ops = []
        self.handles = []

    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        if tensor is None:
            return 0
        return tensor.element_size() * tensor.nelement()

    def _format_size(self, size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    def _register_hooks(self):
        self.handles = []

        def hook_fn(name):
            def hook(module, input, output):
                if not input:
                    return
                try:
                    tensor_inputs = []
                    for inp in input:
                        if isinstance(inp, torch.Tensor):
                            tensor_inputs.append(inp)
                        elif isinstance(inp, list):
                            for item in inp:
                                if isinstance(item, torch.Tensor):
                                    tensor_inputs.append(item)

                    if len(tensor_inputs) < 2:
                        return

                    module_type = type(module).__name__
                    input_details = []
                    total_in_size = 0
                    for idx, inp in enumerate(tensor_inputs):
                        size = self._get_tensor_size(inp)
                        input_details.append({
                            'idx': idx,
                            'shape': list(inp.shape),
                            'size': size
                        })
                        total_in_size += size

                    out_size = 0
                    out_shape = None
                    if output is not None:
                        if isinstance(output, torch.Tensor):
                            out_size = self._get_tensor_size(output)
                            out_shape = list(output.shape)
                        elif isinstance(output, (tuple, list)):
                            for out in output:
                                if isinstance(out, torch.Tensor):
                                    out_size = self._get_tensor_size(out)
                                    out_shape = list(out.shape)
                                    break

                    self.concat_ops.append({
                        'name': name,
                        'type': module_type,
                        'inputs': input_details,
                        'total_input_size': total_in_size,
                        'output_shape': out_shape,
                        'output_size': out_size,
                    })
                except:
                    pass
            return hook

        for name, module in self.model.named_modules():
            try:
                handle = module.register_forward_hook(hook_fn(name))
                self.handles.append(handle)
            except:
                pass

    def analyze(self, input_size: Tuple = (1, 3, 640, 640)):
        self.concat_ops = []
        dummy_input = torch.randn(input_size)
        self._register_hooks()

        self.model.eval()
        with torch.no_grad():
            self.model(dummy_input)

        for handle in self.handles:
            handle.remove()

        return self.concat_ops

    def print_summary(self, results):
        print("\n\n" + "=" * 220)
        print("Concat/Merge Operations - 多输入算子详细分析")
        print("=" * 220)
        print(f"{'Layer Name':<50} {'Type':<20} {'输入1':<40} {'输入2':<40} {'输出':<40} {'总输入':<15}")
        print("-" * 220)

        for info in results:
            name = info['name'][:49]
            layer_type = info['type'][:19]

            in1_str = f"#{info['inputs'][0]['idx']}:{info['inputs'][0]['shape']}({self._format_size(info['inputs'][0]['size'])})" if len(info['inputs']) >= 1 else "-"
            in2_str = f"#{info['inputs'][1]['idx']}:{info['inputs'][1]['shape']}({self._format_size(info['inputs'][1]['size'])})" if len(info['inputs']) >= 2 else "-"
            out_str = f"{info['output_shape']}({self._format_size(info['output_size'])})" if info['output_shape'] else "-"

            print(f"{name:<50} {layer_type:<20} {in1_str:<40} {in2_str:<40} {out_str:<40} {self._format_size(info['total_input_size']):<15}")

        print("-" * 220)
        print(f"共发现 {len(results)} 个多输入算子")

        # 显示更多输入
        multi_input = [r for r in results if len(r['inputs']) > 2]
        if multi_input:
            print("\n\n" + "=" * 220)
            print("多输入算子 (3个及以上输入)")
            print("=" * 220)
            for info in multi_input:
                name = info['name'][:49]
                print(f"\n{name}:")
                for inp in info['inputs']:
                    print(f"  输入{inp['idx']}: {inp['shape']} ({self._format_size(inp['size'])})")
                if info['output_shape']:
                    print(f"  输出: {info['output_shape']} ({self._format_size(info['output_size'])})")


def analyze_yolo_model(model_path: str = None, input_size: Tuple = (1, 3, 640, 640), output_file: str = None):
    """分析 YOLO 模型"""
    import sys
    from ultralytics import YOLO

    if model_path is None:
        model_path = '/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset/cat_face/runs/detect/cat_face_yolo26n2/weights/best.pt'

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    pt_model = model.model

    class OutputCapture:
        def __init__(self):
            self.outputs = []
        def write(self, text):
            self.outputs.append(text)
            sys.__stdout__.write(text)
        def flush(self):
            pass

    if output_file:
        output_capture = OutputCapture()
        sys.stdout = output_capture

    print("\n" + "=" * 80)
    print("YOLO Model Layer Analysis V4 - 数据流追踪版")
    print("=" * 80)

    analyzer = LayerAnalyzerV4(pt_model)
    results = analyzer.analyze(input_size=input_size)
    analyzer.print_summary(results)
    analyzer.print_cache_analysis(results)

    print("\n\n")
    detector = ConcatDetector(pt_model)
    concat_results = detector.analyze(input_size=input_size)
    detector.print_summary(concat_results)

    if output_file:
        sys.stdout = sys.__stdout__
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(''.join(output_capture.outputs))
        print(f"\n\nResults saved to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze YOLO Model V4')
    parser.add_argument('--model', type=str, default=None,
                        help='Model path (default: best.pt)')
    parser.add_argument('--input-size', type=int, nargs=4, default=[1, 3, 640, 640],
                        help='Input size (default: 1 3 640 640)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: print to console)')

    args = parser.parse_args()

    analyze_yolo_model(
        model_path=args.model,
        input_size=tuple(args.input_size),
        output_file=args.output
    )


if __name__ == '__main__':
    main()
