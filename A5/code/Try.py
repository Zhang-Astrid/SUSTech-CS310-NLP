import numpy as np

# 加载两个npy文件
array1 = np.load('input_train_wordpos.npy')
array2 = np.load('input_train_test.npy')

# 比较两个数组是否完全相同
if array1.shape != array2.shape:
    print("两个npy文件的形状不同")
else:
    # 检查元素是否相同
    diff_indices = np.where(array1 != array2)
    diff_count = len(diff_indices[0])

    if diff_count == 0:
        print("两个npy文件的内容完全相同")
    else:
        for idx in zip(*diff_indices):
            print(f"位置 {idx}: file1.npy 的值为 {array1[idx]}, file2.npy 的值为 {array2[idx]}")
        print(f"两个npy文件有 {diff_count} 处不同")

# print(array1.shape)
# print(array1[:5])
# print('#####')
# print(array2[:5])
