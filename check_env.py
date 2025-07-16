import importlib.util
import sys


MAIN_DEPENDENCIES = [
    "torch",  # PyTorch框架
    "numpy",  # 数值计算库
    "pandas",  # 数据处理库
    "matplotlib",  # 可视化库
    "jupyter",  # Jupyter核心
    "jupyterlab",  # JupyterLab界面
    "ipykernel",  # Jupyter内核
    "notebook",  # Jupyter Notebook
    "scipy",  # 科学计算库
    "requests",  # HTTP请求库
    "sklearn",  # 机器学习库（scikit-learn）
    "tensorboard",  # 训练可视化工具
]


def verify_import(package_name: str) -> bool:
    """验证指定的包是否可以成功导入"""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False
    else:
        # 尝试实际导入包（某些包需要完整导入才能验证）
        try:
            __import__(package_name)
            return True
        except Exception as e:
            print(f"导入 {package_name} 时出错: {e}", file=sys.stderr)
            return False


def main():
    print("开始验证主要依赖...")
    all_passed = True

    for package in MAIN_DEPENDENCIES:
        result = verify_import(package)
        status = "成功" if result else "失败"
        print(f"{status} - {package}")

        if not result:
            all_passed = False

    if all_passed:
        print("\n所有主要依赖均已成功导入")
    else:
        print("\n部分依赖导入失败，请检查环境配置。")


if __name__ == "__main__":
    main()