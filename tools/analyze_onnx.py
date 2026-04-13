#!/usr/bin/env python3
"""ONNX Model Analysis Tool

Analyzes ONNX models to extract:
- Weight parameters and memory usage per layer
- Activation shapes and memory usage
- Data flow tracking for cached activations
- Execution time via onnxruntime profiling
- Peak memory estimation
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
import onnxruntime as ort


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f}TB"


def format_params(num_params: int) -> str:
    """Format parameter count to human readable string."""
    if num_params >= 1_000_000:
        return f"{num_params/1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params/1_000:.2f}K"
    return str(num_params)


def get_dtype_size(dtype: int) -> int:
    """Get size in bytes for ONNX data type."""
    dtype_sizes = {
        TensorProto.FLOAT: 4,
        TensorProto.DOUBLE: 8,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.BOOL: 1,
        TensorProto.FLOAT16: 2,
        TensorProto.BFLOAT16: 2,
    }
    return dtype_sizes.get(dtype, 4)


def get_tensor_size(shape: List[int], dtype: int) -> int:
    """Calculate tensor size in bytes."""
    if not shape or any(d <= 0 for d in shape):
        return 0
    num_elements = 1
    for d in shape:
        num_elements *= d
    return num_elements * get_dtype_size(dtype)


def infer_shape_from_value_info(value_info) -> Optional[List[int]]:
    """Extract shape from ValueInfoProto."""
    shape = []
    tensor_type = value_info.type.tensor_type
    if tensor_type.HasField('shape'):
        for dim in tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            else:
                shape.append(-1)  # dynamic dimension
    return shape if shape else None


class ONNXAnalyzer:
    def __init__(self, onnx_path: str, tmp_dir: str = "tmp", output_dir: str = "output"):
        self.onnx_path = Path(onnx_path)
        self.model = onnx.load(str(self.onnx_path))
        self.graph = self.model.graph

        # Setup directories
        self.tmp_dir = Path(tmp_dir)
        self.output_dir = Path(output_dir)
        self.tmp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Build initializers map (weights)
        self.initializers: Dict[str, TensorProto] = {}
        for init in self.graph.initializer:
            self.initializers[init.name] = init

        # Build value_info map for shapes
        self.value_info: Dict[str, Tuple[List[int], int]] = {}
        for vi in self.graph.value_info:
            shape = infer_shape_from_value_info(vi)
            if shape:
                self.value_info[vi.name] = (shape, vi.type.tensor_type.elem_type)

        # Add inputs
        for inp in self.graph.input:
            if inp.name not in self.initializers:
                shape = infer_shape_from_value_info(inp)
                if shape:
                    self.value_info[inp.name] = (shape, inp.type.tensor_type.elem_type)

        # Add outputs
        for out in self.graph.output:
            shape = infer_shape_from_value_info(out)
            if shape:
                self.value_info[out.name] = (shape, out.type.tensor_type.elem_type)

        # Node analysis results
        self.node_weights: Dict[str, Dict] = {}  # node_name -> weight info
        self.node_activations: Dict[str, Dict] = {}  # node_name -> activation info
        self.node_times: Dict[str, float] = {}  # node_name -> execution time (ms)

        # Cache analysis
        self.cached_tensors: Set[str] = set()  # tensors that need to be cached

    def analyze_weights(self):
        """Analyze weight parameters for each node."""
        print("\n" + "="*80)
        print("WEIGHT ANALYSIS")
        print("="*80)

        total_params = 0
        total_memory = 0

        # Track which weights are used by which nodes
        weight_usage: Dict[str, List[str]] = {}  # weight_name -> [node_names]

        for node in self.graph.node:
            node_name = node.name or f"{node.op_type}_{len(self.node_weights)}"

            weight_info = {
                'name': node_name,
                'op_type': node.op_type,
                'weights': [],
                'total_params': 0,
                'total_memory': 0
            }

            for inp in node.input:
                if inp in self.initializers:
                    init = self.initializers[inp]
                    arr = numpy_helper.to_array(init)
                    params = arr.size
                    memory = arr.nbytes

                    weight_info['weights'].append({
                        'name': inp,
                        'shape': list(arr.shape),
                        'dtype': str(arr.dtype),
                        'params': params,
                        'memory': memory
                    })
                    weight_info['total_params'] += params
                    weight_info['total_memory'] += memory

                    if inp not in weight_usage:
                        weight_usage[inp] = []
                    weight_usage[inp].append(node_name)

            if weight_info['total_params'] > 0:
                self.node_weights[node_name] = weight_info
                total_params += weight_info['total_params']
                total_memory += weight_info['total_memory']

        # Print weight summary by node type
        print(f"\n{'Op Type':<20} {'Params':>15} {'Memory':>15}")
        print("-"*52)

        op_type_stats: Dict[str, Tuple[int, int]] = {}
        for node_name, info in self.node_weights.items():
            op_type = info['op_type']
            if op_type not in op_type_stats:
                op_type_stats[op_type] = (0, 0)
            params, mem = op_type_stats[op_type]
            op_type_stats[op_type] = (params + info['total_params'], mem + info['total_memory'])

        for op_type in sorted(op_type_stats.keys(), key=lambda x: op_type_stats[x][0], reverse=True):
            params, mem = op_type_stats[op_type]
            print(f"{op_type:<20} {format_params(params):>15} {format_size(mem):>15}")

        print("-"*52)
        print(f"{'TOTAL':<20} {format_params(total_params):>15} {format_size(total_memory):>15}")

        # Print detailed weight info
        print(f"\n{'Node Name':<40} {'Shape':<30} {'Params':>12} {'Memory':>12}")
        print("-"*100)
        for node_name, info in sorted(self.node_weights.items(), key=lambda x: -x[1]['total_params']):
            for w in info['weights']:
                print(f"{node_name:<40} {str(w['shape']):<30} {format_params(w['params']):>12} {format_size(w['memory']):>12}")

        return total_params, total_memory

    def analyze_activations(self):
        """Analyze activation shapes and memory for each node."""
        print("\n" + "="*80)
        print("ACTIVATION ANALYSIS")
        print("="*80)

        total_activation_memory = 0

        for node in self.graph.node:
            node_name = node.name or f"{node.op_type}_{len(self.node_activations)}"

            act_info = {
                'name': node_name,
                'op_type': node.op_type,
                'inputs': [],
                'outputs': [],
                'input_memory': 0,
                'output_memory': 0
            }

            # Analyze inputs
            for inp in node.input:
                if inp in self.value_info:
                    shape, dtype = self.value_info[inp]
                    memory = get_tensor_size(shape, dtype)
                    act_info['inputs'].append({
                        'name': inp,
                        'shape': shape,
                        'dtype': dtype,
                        'memory': memory
                    })
                    act_info['input_memory'] += memory

            # Analyze outputs
            for out in node.output:
                if out in self.value_info:
                    shape, dtype = self.value_info[out]
                    memory = get_tensor_size(shape, dtype)
                    act_info['outputs'].append({
                        'name': out,
                        'shape': shape,
                        'dtype': dtype,
                        'memory': memory
                    })
                    act_info['output_memory'] += memory

            self.node_activations[node_name] = act_info
            total_activation_memory += act_info['output_memory']

        # Print activation summary by node type
        print(f"\n{'Op Type':<20} {'Input Mem':>15} {'Output Mem':>15} {'Nodes':>10}")
        print("-"*62)

        op_type_stats: Dict[str, Tuple[int, int, int]] = {}
        for node_name, info in self.node_activations.items():
            op_type = info['op_type']
            if op_type not in op_type_stats:
                op_type_stats[op_type] = (0, 0, 0)
            in_mem, out_mem, count = op_type_stats[op_type]
            op_type_stats[op_type] = (in_mem + info['input_memory'],
                                      out_mem + info['output_memory'],
                                      count + 1)

        for op_type in sorted(op_type_stats.keys(), key=lambda x: op_type_stats[x][1], reverse=True):
            in_mem, out_mem, count = op_type_stats[op_type]
            print(f"{op_type:<20} {format_size(in_mem):>15} {format_size(out_mem):>15} {count:>10}")

        print("-"*62)
        print(f"{'TOTAL OUTPUT':<20} {'':<15} {format_size(total_activation_memory):>15}")

        # Print detailed activation info
        print(f"\n{'Node Name':<40} {'Input Shape':<25} {'Output Shape':<25} {'Out Mem':>10}")
        print("-"*105)
        for node_name, info in sorted(self.node_activations.items(),
                                      key=lambda x: -x[1]['output_memory'])[:30]:
            in_shape = str(info['inputs'][0]['shape']) if info['inputs'] else 'N/A'
            out_shape = str(info['outputs'][0]['shape']) if info['outputs'] else 'N/A'
            print(f"{node_name[:40]:<40} {in_shape:<25} {out_shape:<25} {format_size(info['output_memory']):>10}")

        return total_activation_memory

    def analyze_cache(self):
        """Identify tensors that need to be cached (multi-input nodes)."""
        print("\n" + "="*80)
        print("CACHE ANALYSIS")
        print("="*80)

        # Track all outputs and how many times they're used
        tensor_usage: Dict[str, List[str]] = {}  # tensor_name -> [consumer_node_names]

        for node in self.graph.node:
            node_name = node.name or f"{node.op_type}"
            for inp in node.input:
                if inp not in tensor_usage:
                    tensor_usage[inp] = []
                tensor_usage[inp].append(node_name)

        # Find tensors used by multiple nodes or specific ops that require caching
        cache_candidates = []

        for tensor_name, consumers in tensor_usage.items():
            # Skip if it's a weight
            if tensor_name in self.initializers:
                continue

            # Multi-consumer tensors need caching
            if len(consumers) > 1:
                self.cached_tensors.add(tensor_name)

                # Get tensor info
                shape, dtype = self.value_info.get(tensor_name, ([], 1))
                memory = get_tensor_size(shape, dtype)

                cache_candidates.append({
                    'name': tensor_name,
                    'shape': shape,
                    'memory': memory,
                    'consumers': consumers,
                    'reason': f"Used by {len(consumers)} nodes"
                })

        # Also check for Concat and Add nodes that need all inputs simultaneously
        for node in self.graph.node:
            if node.op_type in ['Concat', 'Add', 'Sum', 'Max', 'Min', 'Mean']:
                non_weight_inputs = [inp for inp in node.input if inp not in self.initializers]
                if len(non_weight_inputs) > 1:
                    for inp in non_weight_inputs:
                        if inp not in self.cached_tensors:
                            self.cached_tensors.add(inp)

                            shape, dtype = self.value_info.get(inp, ([], 1))
                            memory = get_tensor_size(shape, dtype)

                            cache_candidates.append({
                                'name': inp,
                                'shape': shape,
                                'memory': memory,
                                'consumers': [node.name or node.op_type],
                                'reason': f"Required for {node.op_type}"
                            })

        # Print cache analysis
        if cache_candidates:
            total_cache_memory = sum(c['memory'] for c in cache_candidates)

            print(f"\nTensors requiring cache:")
            print(f"{'Tensor Name':<50} {'Shape':<25} {'Memory':>12} {'Reason':<20}")
            print("-"*110)

            for c in sorted(cache_candidates, key=lambda x: -x['memory']):
                shape_str = str(c['shape']) if c['shape'] else 'N/A'
                print(f"{c['name'][:50]:<50} {shape_str:<25} {format_size(c['memory']):>12} {c['reason'][:20]:<20}")

            print("-"*110)
            print(f"{'TOTAL CACHE MEMORY':<50} {'':<25} {format_size(total_cache_memory):>12}")
        else:
            print("\nNo tensors require caching.")
            total_cache_memory = 0

        return total_cache_memory

    def profile_execution(self, warmup_runs: int = 3, profile_runs: int = 10, optimized: bool = False):
        """Profile model execution with onnxruntime.

        Args:
            warmup_runs: Number of warmup iterations
            profile_runs: Number of profiling iterations
            optimized: If True, enable graph optimization for better per-op profiling
        """
        print("\n" + "="*80)
        print("EXECUTION TIME PROFILING")
        print("="*80)

        # Create inference session with profiling
        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = True
        sess_options.profile_file_prefix = str(self.tmp_dir / "onnx_profile")

        if optimized:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            print("(Using graph optimization for per-operator profiling)")
        else:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            print("(Graph optimization disabled - shows executor overhead only)")

        # Get input info
        image_size = 320 
        input_info = []
        for inp in self.graph.input:
            if inp.name not in self.initializers:
                shape, dtype = self.value_info.get(inp.name, ([1, 3, image_size, image_size], 1))
                input_info.append((inp.name, shape, dtype))

        if not input_info:
            print("No inputs found, using default shape.")
            input_info = [('images', [1, 3, image_size, image_size], 1)]

        # Create dummy input
        dummy_inputs = {}
        for name, shape, dtype in input_info:
            # Replace dynamic dimensions with 1
            fixed_shape = [d if d > 0 else 1 for d in shape]
            dummy_inputs[name] = np.random.randn(*fixed_shape).astype(np.float32)

        # Create session
        session = ort.InferenceSession(str(self.onnx_path), sess_options)

        # Warmup runs
        print(f"\nRunning {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            session.run(None, dummy_inputs)

        # Profile runs
        print(f"Running {profile_runs} profile iterations...")
        for _ in range(profile_runs):
            session.run(None, dummy_inputs)

        # Get profiling results
        prof_file = session.end_profiling()

        # Move profiling file to tmp directory if it's not already there
        prof_path = Path(prof_file)
        if prof_path.parent != self.tmp_dir:
            target_path = self.tmp_dir / prof_path.name
            prof_path.rename(target_path)
            prof_file = str(target_path)

        print(f"Profiling data saved to: {prof_file}")

        # Parse profiling file
        with open(prof_file, 'r') as f:
            prof_data = json.load(f)

        # onnxruntime profiling can have various formats
        # Common keys: 'name', 'node_name', 'dur' (duration in nanoseconds)
        node_times: Dict[str, List[float]] = {}
        total_duration = 0.0

        for entry in prof_data:
            # Try various key names for node name and duration
            node_name = entry.get('node_name') or entry.get('name', '')
            duration_ns = entry.get('dur', 0) or entry.get('duration', 0)

            if node_name and duration_ns:
                duration_ms = duration_ns / 1_000_000  # Convert to ms

                if node_name not in node_times:
                    node_times[node_name] = []
                node_times[node_name].append(duration_ms)

            # Track total session time if available
            if entry.get('name') == 'tot' or 'total' in str(entry.get('name', '')).lower():
                total_duration = entry.get('dur', 0) / 1_000_000

        # Calculate average times
        print(f"\n{'Node Name':<60} {'Avg Time (ms)':>15} {'% Total':>10}")
        print("-"*90)

        # Sum up all node times for percentage calculation
        all_times = []
        for times in node_times.values():
            all_times.extend(times)
        total_time = sum(all_times)

        node_avg_times = []

        for node_name, times in sorted(node_times.items(), key=lambda x: -sum(x[1])):
            avg_time = sum(times) / len(times)
            self.node_times[node_name] = avg_time
            node_avg_times.append((node_name, avg_time))
            pct = (sum(times) / total_time * 100) if total_time > 0 else 0

            if avg_time > 0.01:  # Only show nodes with significant time
                print(f"{node_name[:60]:<60} {avg_time:>15.3f} {pct:>10.1f}%")

        print("-"*90)

        # Calculate overall session time
        if total_duration > 0:
            print(f"{'SESSION TOTAL':<60} {total_duration:>15.3f}")
        else:
            print(f"{'TOTAL (sum of nodes)':<60} {total_time / profile_runs if profile_runs > 0 else 0:>15.3f}")

        if not node_avg_times:
            print("\nNote: Profiling data may require running with graph optimization enabled.")
            print("      Try running without --profile to see static analysis results.")

        return node_avg_times

    def analyze_execution_memory_timeline(self):
        """Analyze memory usage at each node execution step.

        Tracks:
        - Node weights memory (only this node's weights)
        - Cached activations memory (tensors needed by future nodes)
        - Current working memory (current node's outputs)
        - Total memory at each step
        """
        print("\n" + "="*80)
        print("EXECUTION MEMORY TIMELINE")
        print("="*80)
        print("\nTracking memory usage at each node execution step...")
        print("(Weights show per-node size, activations cached until no longer needed)\n")

        # Track which weights have been loaded
        loaded_weights: Set[str] = set()
        loaded_weights_memory = 0

        # Track which tensors are still needed (will be used by future nodes)
        tensor_needed_by: Dict[str, Set[int]] = {}  # tensor_name -> set of node indices

        # Build tensor usage map: which future nodes need this tensor?
        for idx, node in enumerate(self.graph.node):
            for inp in node.input:
                if inp not in self.initializers:
                    if inp not in tensor_needed_by:
                        tensor_needed_by[inp] = set()
                    tensor_needed_by[inp].add(idx)

        # Track live cached tensors
        live_cached_tensors: Set[str] = set()
        cached_memory = 0

        # Track memory timeline
        timeline = []

        print(f"{'Idx':<5} {'Node Name':<45} {'Weights':>12} {'Cache':>12} {'Output':>12} {'Total':>12}")
        print("-" * 105)

        peak_total = 0
        peak_node = ""

        for idx, node in enumerate(self.graph.node):
            node_name = node.name or f"{node.op_type}_{idx}"

            # 1. Load weights for this node (track cumulative loaded weights separately)
            node_weights_mem = 0
            for inp in node.input:
                if inp in self.initializers and inp not in loaded_weights:
                    init = self.initializers[inp]
                    arr = numpy_helper.to_array(init)
                    loaded_weights.add(inp)
                    loaded_weights_memory += arr.nbytes
                if inp in self.initializers:
                    init = self.initializers[inp]
                    arr = numpy_helper.to_array(init)
                    node_weights_mem += arr.nbytes

            # 2. Update cache: remove tensors no longer needed
            tensors_to_remove = []
            for tensor_name in live_cached_tensors:
                # Check if any future node still needs this tensor
                future_users = tensor_needed_by.get(tensor_name, set())
                future_users = {i for i in future_users if i > idx}

                if not future_users:
                    tensors_to_remove.append(tensor_name)

            for tensor_name in tensors_to_remove:
                if tensor_name in self.value_info:
                    shape, dtype = self.value_info[tensor_name]
                    memory = get_tensor_size(shape, dtype)
                    cached_memory -= memory
                live_cached_tensors.remove(tensor_name)

            # 3. Add current outputs to cache if needed by future nodes
            current_output_mem = 0
            for out in node.output:
                if out in self.value_info:
                    shape, dtype = self.value_info[out]
                    memory = get_tensor_size(shape, dtype)
                    current_output_mem += memory

                    # Check if this output is needed by future nodes
                    future_users = tensor_needed_by.get(out, set())
                    future_users = {i for i in future_users if i > idx}

                    if future_users and out not in live_cached_tensors:
                        live_cached_tensors.add(out)
                        cached_memory += memory

            # 4. Calculate total memory at this step
            total_memory = loaded_weights_memory + cached_memory

            # Track peak
            if total_memory > peak_total:
                peak_total = total_memory
                peak_node = node_name

            timeline.append({
                'idx': idx,
                'node_name': node_name,
                'weights_mem': node_weights_mem,
                'cache_mem': cached_memory,
                'output_mem': current_output_mem,
                'total_mem': total_memory
            })

            # Print every 20th node or peak nodes
            if idx % 20 == 0 or total_memory == peak_total:
                print(f"{idx:<5} {node_name[:45]:<45} {format_size(node_weights_mem):>12} "
                      f"{format_size(cached_memory):>12} {format_size(current_output_mem):>12} "
                      f"{format_size(total_memory):>12}{'  <- PEAK' if total_memory == peak_total else ''}")

        print("-" * 105)
        print(f"\nPeak Memory: {format_size(peak_total)} at node: {peak_node}")

        # Print top memory-consuming nodes
        print(f"\n{'Top 10 Memory-Critical Nodes:':<50}")
        print(f"{'Idx':<5} {'Node Name':<45} {'Weights':>12} {'Cache':>12} {'Output':>12} {'Total':>12}")
        print("-" * 105)

        sorted_timeline = sorted(timeline, key=lambda x: -x['total_mem'])[:10]
        for entry in sorted_timeline:
            print(f"{entry['idx']:<5} {entry['node_name'][:45]:<45} "
                  f"{format_size(entry['weights_mem']):>12} "
                  f"{format_size(entry['cache_mem']):>12} "
                  f"{format_size(entry['output_mem']):>12} "
                  f"{format_size(entry['total_mem']):>12}")

        return timeline, peak_total

    def estimate_peak_memory(self):
        """Estimate peak memory during inference."""
        print("\n" + "="*80)
        print("PEAK MEMORY ESTIMATION")
        print("="*80)

        # Calculate weights memory
        weights_memory = sum(info['total_memory'] for info in self.node_weights.values())

        # Calculate activation memory at each step
        # This is a simplified estimation - assumes sequential execution
        peak_activation = 0
        current_activation = 0

        # Track live tensors
        live_tensors: Set[str] = set()

        for node in self.graph.node:
            # Add outputs
            for out in node.output:
                if out in self.value_info:
                    shape, dtype = self.value_info[out]
                    memory = get_tensor_size(shape, dtype)
                    current_activation += memory
                    live_tensors.add(out)

            # Update peak
            if current_activation > peak_activation:
                peak_activation = current_activation

            # Remove inputs that are no longer needed
            for inp in node.input:
                if inp in self.initializers:
                    continue
                # Check if this tensor is used elsewhere
                still_used = False
                for other_node in self.graph.node:
                    if other_node != node and inp in other_node.input:
                        still_used = True
                        break

                if not still_used and inp in live_tensors:
                    if inp in self.value_info:
                        shape, dtype = self.value_info[inp]
                        memory = get_tensor_size(shape, dtype)
                        current_activation -= memory
                    live_tensors.remove(inp)

        total_peak = weights_memory + peak_activation

        print(f"\nWeights Memory:       {format_size(weights_memory)}")
        print(f"Peak Activation:      {format_size(peak_activation)}")
        print(f"Cache Memory:         {format_size(sum(get_tensor_size(*self.value_info.get(t, ([], 1))) for t in self.cached_tensors))}")
        print("-" * 40)
        print(f"Estimated Peak:       {format_size(total_peak)}")

        return total_peak

    def generate_summary(self):
        """Generate overall summary."""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        total_params = sum(info['total_params'] for info in self.node_weights.values())
        total_weights_mem = sum(info['total_memory'] for info in self.node_weights.values())
        total_activation_mem = sum(info['output_memory'] for info in self.node_activations.values())
        cache_mem = sum(get_tensor_size(*self.value_info.get(t, ([], 1))) for t in self.cached_tensors)

        print(f"\n{'Metric':<30} {'Value':>20}")
        print("-"*52)
        print(f"{'Total Parameters':<30} {format_params(total_params):>20}")
        print(f"{'Weights Memory':<30} {format_size(total_weights_mem):>20}")
        print(f"{'Activation Memory':<30} {format_size(total_activation_mem):>20}")
        print(f"{'Cache Memory':<30} {format_size(cache_mem):>20}")
        print(f"{'Number of Nodes':<30} {len(list(self.graph.node)):>20}")
        print(f"{'Number of Cached Tensors':<30} {len(self.cached_tensors):>20}")


def main():
    parser = argparse.ArgumentParser(description='Analyze ONNX model')
    parser.add_argument('model', type=str, help='Path to ONNX model')
    parser.add_argument('--profile', action='store_true', help='Enable execution profiling')
    parser.add_argument('--optimized', action='store_true', help='Enable graph optimization for per-op profiling')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup runs for profiling')
    parser.add_argument('--runs', type=int, default=10, help='Profile runs')
    parser.add_argument('--tmp-dir', type=str, default='tmp', help='Directory for temporary files')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory for output files')

    args = parser.parse_args()

    # Create analyzer with directories
    analyzer = ONNXAnalyzer(args.model, tmp_dir=args.tmp_dir, output_dir=args.output_dir)

    # Prepare output file
    from datetime import datetime
    model_name = Path(args.model).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = analyzer.output_dir / f"{model_name}_analysis_{timestamp}.txt"

    # Run analysis and save to file
    import sys
    from io import StringIO

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    print("="*80)
    print(f"ONNX Model Analysis: {args.model}")
    print("="*80)

    # Run all analyses
    analyzer.analyze_weights()
    analyzer.analyze_activations()
    analyzer.analyze_cache()

    if args.profile:
        analyzer.profile_execution(warmup_runs=args.warmup, profile_runs=args.runs, optimized=args.optimized)

    analyzer.analyze_execution_memory_timeline()
    analyzer.estimate_peak_memory()
    analyzer.generate_summary()

    # Get captured output
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Print to console and save to file
    print(output)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"\nReport saved to: {output_file}")
    print(f"Temporary files in: {analyzer.tmp_dir}")


if __name__ == '__main__':
    main()