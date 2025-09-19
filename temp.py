# import pickle
# import numpy as np
#
# # 定义文件路径
# pkl_file_path = "WESAD/S2/S2.pkl"
#
# # 加载 .pkl 文件，使用二进制模式和正确的编码参数
# with open(pkl_file_path, 'rb') as f:
#     data = pickle.load(f, encoding='latin1')
#
# # 打印整个数据结构的键
# print("主数据结构中的键:")
# for key in data.keys():
#     print(f"  {key}")
#
# # 检查 'signal' 键下的内容
# if 'signal' in data:
#     signal_data = data['signal']
#     print("\n'signal' 键下的内容:")
#     for key in signal_data.keys():
#         print(f"  {key}")
#
#     # 检查 'chest' 键下的内容
#     if 'chest' in signal_data:
#         chest_data = signal_data['chest']
#         print("\n'chest' 键下的内容:")
#         for key in chest_data.keys():
#             print(f"  {key}")
#
#         # 详细检查 chest 部分的所有传感器通道
#         print("\n胸部传感器数据详情:")
#         for channel_name, channel_data in chest_data.items():
#             if isinstance(channel_data, np.ndarray):
#                 print(f"\n{channel_name} 通道:")
#                 print(f"  数据形状: {channel_data.shape}")
#                 print(f"  数据类型: {channel_data.dtype}")
#                 print(f"  数据范围: [{np.min(channel_data):.6f}, {np.max(channel_data):.6f}]")
#                 print("  前5行数据:")
#                 if channel_data.ndim == 1:
#                     print(f"    {channel_data[:5]}")
#                 elif channel_data.ndim == 2:
#                     for i in range(min(5, channel_data.shape[0])):
#                         print(f"    {channel_data[i]}")
#
#     # 检查 'wrist' 键下的内容（如果存在）
#     if 'wrist' in signal_data:
#         wrist_data = signal_data['wrist']
#         print("\n'wrist' 键下的内容:")
#         for key in wrist_data.keys():
#             print(f"  {key}")
#
#         # 详细检查 wrist 部分的所有传感器通道
#         print("\n腕部传感器数据详情:")
#         for channel_name, channel_data in wrist_data.items():
#             if isinstance(channel_data, np.ndarray):
#                 print(f"\n{channel_name} 通道:")
#                 print(f"  数据形状: {channel_data.shape}")
#                 print(f"  数据类型: {channel_data.dtype}")
#                 print(f"  数据范围: [{np.min(channel_data):.6f}, {np.max(channel_data):.6f}]")
#                 print("  前5行数据:")
#                 if channel_data.ndim == 1:
#                     print(f"    {channel_data[:5]}")
#                 elif channel_data.ndim == 2:
#                     for i in range(min(5, channel_data.shape[0])):
#                         print(f"    {channel_data[i]}")
#
# # 检查标签数据
# if 'label' in data:
#     label_data = data['label']
#     print(f"\n标签数据:")
#     print(f"  数据形状: {label_data.shape}")
#     print(f"  数据类型: {label_data.dtype}")
#     print(f"  唯一标签值: {np.unique(label_data)}")
#     print(f"  标签分布:")
#     unique, counts = np.unique(label_data, return_counts=True)
#     for u, c in zip(unique, counts):
#         print(f"    标签 {u}: {c} 个样本")
#
# # 检查受试者信息
# if 'subject' in data:
#     subject_data = data['subject']
#     print(f"\n受试者信息: {subject_data}")
from pathlib import Path

WESAD_ROOT = Path('./WESAD')
OUTPUT_PATH = Path('./data')
print(WESAD_ROOT)
print(OUTPUT_PATH)