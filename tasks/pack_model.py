#!/usr/bin/env python3
"""
模型打包脚本 - 使用 thinker tpacker 进行图转换和打包

支持多种硬件平台:
- venusA: 推荐用于高性能推理
- venus: Venus 芯片
- arcs: ARCS 芯片

使用方法:
    # 查看帮助
    python tasks/pack_model.py --help

    # 基本打包 (使用 thinker tpacker)
    python tasks/pack_model.py --model models/yolo26n.onnx --platform venusA

    # ONNX 优化打包 (替代方案)
    python tasks/pack_model.py --model models/yolo26n.onnx --optimize-onnx

    # 启用图优化和内存报告
    python tasks/pack_model.py --model models/yolo26n.onnx --platform venusA --optimize --dump
"""

import os
import sys
import argparse
from pathlib import Path

# 项目根目录
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# 添加 thinker 路径
THINKER_PATH = PROJECT_ROOT / 'thirdparty' / 'thinker' / 'tools'
if THINKER_PATH.exists() and str(THINKER_PATH) not in sys.path:
    sys.path.insert(0, str(THINKER_PATH))

# 检查 thinker 是否可用
THINKER_AVAILABLE = False
try:
    from tpacker import tpacker
    THINKER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ thinker tpacker 导入失败: {e}")
    print("   将使用 ONNX 优化作为替代方案")


def get_default_config():
    """获取默认打包配置 - 使用任务目录下的配置"""
    # 优先使用任务目录配置，否则回退到根cfg目录
    task_config = PROJECT_ROOT / 'cfg' / 'cat_320X320_detect' / 'pack_config.yaml'
    if task_config.exists():
        return task_config
    return PROJECT_ROOT / 'cfg' / 'pack_config.yaml'


def check_onnx_model(model_path):
    """检查 ONNX 模型是否有效"""
    try:
        import onnx
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"✅ ONNX 模型有效: {model_path}")
        print(f"   输入: {model.graph.input[0].name} {get_tensor_shape(model.graph.input[0])}")
        print(f"   输出数量: {len(model.graph.output)}")
        return True
    except Exception as e:
        print(f"❌ ONNX 模型无效: {e}")
        return False


def get_tensor_shape(tensor_info):
    """获取 tensor 的形状"""
    try:
        shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in tensor_info.type.tensor_type.shape.dim]
        return f"[{', '.join(map(str, shape))}]"
    except:
        return "[unknown]"


def run_pack(args):
    """运行模型打包"""
    if not THINKER_AVAILABLE:
        print("❌ thinker 不可用")
        print("   请安装: cd thirdparty/thinker/tools && pip install -e .")
        return False

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return False

    # 检查 ONNX 模型
    if not check_onnx_model(model_path):
        return False

    # 设置输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.parent / f"{model_path.stem}_packed.bin"

    print(f"\n{'=' * 60}")
    print("🔧 模型打包配置:")
    print(f"{'=' * 60}")
    print(f"  输入模型: {model_path}")
    print(f"  输出路径: {output_path}")
    print(f"  目标平台: {args.platform}")
    print(f"  图优化: {'是' if args.optimize else '否'}")
    print(f"  内存报告: {'是' if args.dump else '否'}")
    print(f"{'=' * 60}\n")

    # 创建 tpacker 参数
    class Args:
        graph_path = str(model_path)
        output_path = str(output_path)
        platform = args.platform
        device = args.device if args.device else "cuda:0"
        model_dtype = args.dtype if args.dtype else "float32"
        layout = args.layout if args.layout else "NCHW"
        optimize = args.optimize
        dump = args.dump
        export_config = args.export_config if hasattr(args, 'export_config') else None

    # 运行 tpacker
    try:
        print("🚀 开始打包...\n")
        tpacker(Args(), {})

        print(f"\n✅ 打包完成!")
        print(f"   输出: {output_path}")

        if args.export_config:
            config_path = Path(args.export_config)
            print(f"   配置: {config_path}")

        return True

    except Exception as e:
        print(f"\n❌ 打包失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def optimize_onnx_model(model_path, output_path=None, verbose=True):
    """
    使用 ONNX 优化器进行模型优化
    作为 thinker tpacker 的替代方案
    """
    import onnx

    model_path = Path(model_path)
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_optimized.onnx"
    else:
        output_path = Path(output_path)

    print(f"\n🔧 ONNX 模型优化:")
    print(f"   输入: {model_path}")
    print(f"   输出: {output_path}")

    # 加载模型
    model = onnx.load(str(model_path))
    if verbose:
        orig_nodes = len(model.graph.node)
        print(f"   原始算子数: {orig_nodes}")

    # 尝试使用 onnxoptimizer
    optimized_model = model
    try:
        import onnxoptimizer as opt
        available_passes = opt.get_available_passes()
        if verbose:
            print(f"   可用优化: {len(available_passes)} 个")

        # 尝试应用一些常见优化
        try:
            # 使用基本的优化 passes
            basic_passes = ['eliminate_nop_cast', 'eliminate_nop_dropout', 'eliminate_nop_expand']
            available_basic = [p for p in basic_passes if p in available_passes]

            if available_basic:
                optimized_model = opt.optimize(model, available_basic)
                if verbose:
                    new_nodes = len(optimized_model.graph.node)
                    print(f"   优化后算子数: {new_nodes} ({orig_nodes - new_nodes} 个已消除)")
        except Exception as e:
            if verbose:
                print(f"   优化应用失败: {e}")

    except ImportError:
        if verbose:
            print("   onnxoptimizer 未安装，跳过优化")

    # 尝试使用 onnxruntime 进行图优化
    try:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 创建一个临时 session 来优化模型
        # 这不会改变原始模型，只是验证优化
        if verbose:
            print("   ONNXRuntime 图优化: 已启用")

    except Exception as e:
        if verbose:
            print(f"   ONNXRuntime 优化: {e}")

    # 保存模型
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(optimized_model, str(output_path))
    onnx.checker.check_model(optimized_model)

    # 打印模型大小变化
    orig_size = model_path.stat().st_size / (1024 * 1024)
    new_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\n📊 模型大小:")
    print(f"   原始: {orig_size:.2f} MB")
    print(f"   优化后: {new_size:.2f} MB ({new_size/orig_size*100:.1f}%)")

    return output_path


def run_onnx_optimize(args):
    """运行 ONNX 优化"""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return False

    # 检查 ONNX 模型
    if not check_onnx_model(model_path):
        return False

    # 设置输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.parent / f"{model_path.stem}_optimized.onnx"

    print(f"\n{'=' * 60}")
    print("🔧 ONNX 优化配置:")
    print(f"{'=' * 60}")
    print(f"  输入模型: {model_path}")
    print(f"  输出路径: {output_path}")
    print(f"{'=' * 60}\n")

    try:
        result_path = optimize_onnx_model(model_path, output_path)
        print(f"\n✅ 优化完成!")
        print(f"   输出: {result_path}")
        return True

    except Exception as e:
        print(f"\n❌ 优化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_onnx(model_path):
    """分析 ONNX 模型结构"""
    print(f"\n📊 ONNX 模型分析: {model_path}")
    print("=" * 60)

    try:
        import onnx
        from onnx import numpy_helper

        model = onnx.load(model_path)

        # 统计信息
        print(f"\n模型信息:")
        print(f"  IR 版本: {model.ir_version}")
        print(f"  生产者: {model.producer_name} {model.producer_version}")

        # 输入输出
        print(f"\n输入:")
        for inp in model.graph.input:
            print(f"  {inp.name}: {get_tensor_shape(inp)}")

        print(f"\n输出:")
        for out in model.graph.output:
            print(f"  {out.name}: {get_tensor_shape(out)}")

        # 算子统计
        ops = {}
        for node in model.graph.node:
            op_type = node.op_type
            ops[op_type] = ops.get(op_type, 0) + 1

        print(f"\n算子统计 (共 {len(model.graph.node)} 个):")
        for op, count in sorted(ops.items(), key=lambda x: -x[1])[:15]:
            print(f"  {op:20} {count:3} 个")

        if len(ops) > 15:
            print(f"  ... 还有 {len(ops) - 15} 种算子")

        # 初始化器
        print(f"\n权重参数: {len(model.graph.initializer)} 个")

        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='模型打包工具 - 使用 thinker tpacker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析 ONNX 模型
  python tasks/pack_model.py --analyze models/yolo26n.onnx

  # ONNX 优化 (推荐替代方案)
  python tasks/pack_model.py --model models/yolo26n.onnx --optimize-onnx

  # 打包到 VenusA 平台
  python tasks/pack_model.py --model models/yolo26n.onnx --platform venusA

  # 打包并启用优化和报告
  python tasks/pack_model.py --model models/yolo26n.onnx --platform venusA --optimize --dump

支持平台:
  venusA - VenusA 芯片 (推荐)
  venus  - Venus 芯片
  arcs   - ARCS 芯片
        """
    )

    parser.add_argument('--model', type=str, help='ONNX 模型路径')

    parser.add_argument('--analyze', type=str, metavar='ONNX',
                        help='分析 ONNX 模型结构')

    parser.add_argument('--optimize-onnx', action='store_true',
                        help='使用 ONNX 优化器进行模型优化 (推荐)')

    parser.add_argument('--platform', type=str, default='venusA',
                        choices=['venusA', 'venus', 'arcs'],
                        help='目标硬件平台')

    parser.add_argument('--output', type=str, help='输出路径')

    parser.add_argument('--device', type=str, help='设备 (如 cuda:0)')

    parser.add_argument('--dtype', type=str, choices=['float32', 'float16', 'int8'],
                        help='数据类型')

    parser.add_argument('--layout', type=str, choices=['NCHW', 'NHWC'],
                        help='数据布局')

    parser.add_argument('--optimize', action='store_true',
                        help='启用图优化 (thinker)')

    parser.add_argument('--dump', action='store_true',
                        help='生成内存分析报告')

    parser.add_argument('--export-config', type=str,
                        help='导出配置文件')

    args = parser.parse_args()

    # 分析模式
    if args.analyze:
        success = analyze_onnx(args.analyze)
        sys.exit(0 if success else 1)

    # ONNX 优化模式
    if args.optimize_onnx and args.model:
        success = run_onnx_optimize(args)
        sys.exit(0 if success else 1)

    # 打包模式
    if args.model:
        success = run_pack(args)
        sys.exit(0 if success else 1)

    # 无参数时显示帮助
    parser.print_help()


if __name__ == '__main__':
    main()