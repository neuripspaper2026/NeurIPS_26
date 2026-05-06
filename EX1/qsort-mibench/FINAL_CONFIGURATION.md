# qsort-mibench Final Configuration

## ✅ Data Scales (按你的要求配置)

| Size | Vertices | File Size | Runtime | Speedup vs mini |
|------|----------|-----------|---------|-----------------|
| **mini** | 60,000 | 1.9 MB | 0.06s | 1.0x |
| **small** | 300,000 | 9.5 MB | 0.25s | 4.4x |
| **medium** | 2,000,000 | 63 MB | 1.73s | 30.4x |
| **large** | 10,000,000 | 315 MB | 9.00s | 158.5x |
| **extra-large** | 30,000,000 | 943 MB | 28.0s | 493.8x |

## 🎯 Key Features

### 1. Excellent Performance Differentiation
- mini → extra-large: **500x slowdown**
- Each level has significant time difference
- Perfect for performance evaluation

### 2. Scale Progression
- mini → small: **5x** vertices
- small → medium: **6.7x** vertices  
- medium → large: **5x** vertices
- large → extra-large: **3x** vertices

### 3. Clean Output Format
- Only sorted coordinates (x y z)
- No extra messages
- Easy to compare correctness with `diff`

## 🔧 Code Modifications

### EX1_optimized_codes/qsort_large.c

1. **MAXARRAY** (line 6):
   ```c
   #define MAXARRAY 31000000  /* Supports up to 30M vertices */
   ```

2. **Dynamic Memory Allocation**:
   ```c
   struct my3DVertexStruct *array;
   array = malloc(MAXARRAY * sizeof(struct my3DVertexStruct));
   // ... use array ...
   free(array);
   ```

3. **Removed Print Statement** (line 63-64):
   ```c
   /* printf("\nSorting %d vectors...\n\n",count); */  // Commented out
   ```

## 📝 For Your Variants

All 40 variant files need the same modifications:

1. Change `#define MAXARRAY 31000000`
2. Use `malloc()` instead of stack array
3. Comment out the printf statement (optional, for consistency)

**Note**: These are engineering optimizations, NOT algorithm changes!

## 🚀 Usage

### Generate All Data
```bash
for size in mini small medium large extra-large; do
    python generate.py synthetic --preset $size --out-dir input_data/$size/input
done
```

### Run Tests
```bash
# Quick test
./EX1_optimized_codes/qsort_large_gcc input_data/mini/input/input_mini.dat > output_mini.txt

# All sizes
./test_all_sizes.sh
```

### Compare Correctness
```bash
# Baseline
./EX1_optimized_codes/qsort_large_gcc input_data/mini/input/input_mini.dat > baseline.txt

# Variant
./EX1_optimized_codes/qsort_large_gcc_v1 input_data/mini/input/input_mini.dat > variant1.txt

# Compare
diff baseline.txt variant1.txt
# No output = identical = correct ✓
```

## ✅ Verification

All sizes tested and verified:
- ✓ All input vertices are processed
- ✓ Output line count matches input
- ✓ Performance scales as expected
- ✓ Memory allocation works correctly

Ready for your experiments! 🎉
