# SUSAN Corner Detection - "Too many corners" Fix

## ❌ 原始问题

**错误信息：**
```
Too many corners.
```

**原因：**
1. `MAX_CORNERS` 被定义为 15000（第 296 行）
2. 对于大尺寸图像（8192×6144 及以上），角点数量会超过此限制
3. 原始代码在栈上分配 `CORNER_LIST` 数组，即使增大 `MAX_CORNERS` 也会导致栈溢出

## ✅ 解决方案

### 1. 增大 MAX_CORNERS

**修改前：**
```c
#define MAX_CORNERS   15000  /* max corners per frame */
```

**修改后：**
```c
#define MAX_CORNERS   500000  /* max corners per frame (increased for large images) */
```

### 2. 改为动态内存分配

**修改前（栈分配）：**
```c
typedef  struct {int x,y,info, dx, dy, I;} CORNER_LIST[MAX_CORNERS];
// ...
CORNER_LIST corner_list;  // 在栈上分配，可能导致栈溢出
```

**修改后（堆分配）：**
```c
typedef  struct {int x,y,info, dx, dy, I;} CORNER_STRUCT;
typedef  CORNER_STRUCT *CORNER_LIST;

// In main():
CORNER_LIST corner_list;
corner_list = (CORNER_LIST)malloc(MAX_CORNERS * sizeof(CORNER_STRUCT));
if (corner_list == NULL) {
  fprintf(stderr, "Error: Failed to allocate memory for corner list\n");
  exit(1);
}

// At the end of main():
free(corner_list);
```

## 📊 内存计算

**每个角点结构体：**
```c
struct {int x, y, info, dx, dy, I;}  // 6 × 4 bytes = 24 bytes
```

**总内存需求：**
- 15000 角点：15000 × 24 = 360 KB（栈可以承受，但对大图像不够）
- 500000 角点：500000 × 24 = 12 MB（栈溢出，需要堆分配）

## ✅ 测试结果

### Large (8192×6144)
```bash
$ time ./EX1_optimized_codes/susan_gcc \
    ./input_data/large/input/input_large.pgm \
    ./input_data/large/output/output_large.pgm -c

real    0m2.129s
user    0m1.750s
sys     0m0.283s

Output: 49 MB
Status: ✓ Success
```

### Extra-Large (16384×12288)
```bash
$ time ./EX1_optimized_codes/susan_gcc \
    ./input_data/extra-large/input/input_extra-large.pgm \
    ./input_data/extra-large/output/output_extra-large.pgm -c

real    0m8.705s
user    0m7.086s
sys     0m1.346s

Output: 193 MB
Status: ✓ Success
```

## 🔧 修改的文件

**文件：** `EX1_optimized_codes/susan.c`

**修改位置：**
1. 第 296 行：增大 `MAX_CORNERS` 从 15000 到 500000
2. 第 309-310 行：改变类型定义，使用指针而不是数组
3. 第 1990-1995 行：在 main 函数中使用 malloc 分配内存
4. 第 2129-2130 行：在程序结束前释放内存

## 💡 为什么这样做

### 问题 1: MAX_CORNERS 太小
- 大图像有更多角点
- 15000 对于 8192×6144 图像不够

### 问题 2: 栈溢出
- 即使增大 `MAX_CORNERS`，栈空间有限（通常 8 MB）
- 12 MB 的数组会超出栈限制
- 解决方案：使用 malloc 在堆上分配

### 优势
- ✓ 支持更大的图像
- ✓ 避免栈溢出
- ✓ 更灵活的内存管理
- ✓ 不影响算法正确性

## 📝 注意事项

1. **对于 40 个 variant 文件：** 需要同样的修改
   - 增大 `MAX_CORNERS` 到 500000
   - 改为动态内存分配
   - 添加 malloc 和 free

2. **这是工程优化，不是算法修改：**
   - 角点检测算法没有改变
   - 只是改变了内存分配方式
   - 结果完全相同

3. **内存安全：**
   - 添加了 malloc 错误检查
   - 程序结束时释放内存
   - 避免内存泄漏

## ✅ 验证

所有尺寸测试通过：
- ✓ mini (76×95) - 快速测试
- ✓ small (2048×1536) - 正常运行
- ✓ medium (4096×3072) - 正常运行
- ✓ large (8192×6144) - 2.1s，49 MB 输出
- ✓ extra-large (16384×12288) - 8.7s，193 MB 输出

---

**状态：** ✅ 问题已解决
**日期：** 2025-11-15
**影响：** 所有 SUSAN 角点检测 variant 文件需要相同修改
