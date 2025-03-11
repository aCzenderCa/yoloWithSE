import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
import torch
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def measure_latency(model, input_size, num_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 生成归一化的随机输入
    dummy_input = torch.rand(1, 3, input_size, input_size).to(device)

    # 禁用输出
    model.verbose = False

    # 预热
    for _ in range(10):
        _ = model(dummy_input)

    # 测量延迟
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    end_time = time.time()

    avg_latency = (end_time - start_time) / num_iterations
    return avg_latency * 1000  # 转换为毫秒


def main():
    yolo_versions = [
        r'yolo11n-obb.yaml',
        r'yolo11n-obb-withTransform.yaml',
    ]
    input_size = 640  # YOLO的标准输入大小

    print(f"设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"输入大小: {input_size}x{input_size}")
    print("模型\t\t延迟 (ms)")
    print("-" * 30)

    for version in yolo_versions:
        model = YOLO(version)
        latency = measure_latency(model, input_size)
        model_name = os.path.basename(version).split('.')[0]  # 提取文件名，去掉扩展名
        print(f"{model_name}\t\t{latency:.2f}")


if __name__ == "__main__":
    main()