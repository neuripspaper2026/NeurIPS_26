#!/bin/bash

# KMP-Mach Data Generation Script
# Generates 5 datasets with different search patterns (fixed text size: 32411 chars)

set -e

BENCHMARK="kmp-mach"
BASELINE_EXE="./EX1_optimized_codes/kmp_gcc"
CORR_OUTPUT="./EX5_correctness/output_gcc.data"
TEXT_FILE="TR.txt"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║    KMP-MACH Data Generation (Fixed Text Size: 32411 chars)       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check baseline exists
if [ ! -f "$BASELINE_EXE" ]; then
    echo "⚠️  Baseline not found. Compiling..."
    make clean > /dev/null 2>&1 || true
    make baseline
    echo ""
fi

# Check text file exists
if [ ! -f "$TEXT_FILE" ]; then
    echo "❌ Error: $TEXT_FILE not found!"
    exit 1
fi

# Create directory structure
echo "📁 Creating directory structure..."
for size in mini small medium large extra-large; do
    mkdir -p "input_data/$size/input"
    mkdir -p "input_data/$size/output"
done
echo "   ✓ Directories created"

# Function to generate KMP input with specific pattern
generate_data() {
    local size=$1
    local pattern=$2
    local description=$3
    echo ""
    echo "[$size] Pattern: \"$pattern\" - $description"

    # Create a temporary generate.c with the specific pattern
    cat <<EOF > generate_temp.c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#define PATTERN_SIZE 4
#define STRING_SIZE 32411

struct bench_args_t {
  char pattern[PATTERN_SIZE];
  char input[STRING_SIZE];
  int32_t kmpNext[PATTERN_SIZE];
  int32_t n_matches[1];
};

#include "../../common_MachSuite/support.h"

int main(int argc, char **argv)
{
  struct bench_args_t data;
  int status, fd, nbytes;

  // Load string file
  fd = open("TR.txt", O_RDONLY);
  assert( fd>=0 && "couldn't open text file" );
  nbytes = 0;
  do {
    status = read(fd, data.input, STRING_SIZE-nbytes);
    assert(status>=0 && "couldn't read from text file");
    nbytes+=status;
  } while( nbytes<STRING_SIZE );
  close(fd);
  
  // Set specific pattern
  memcpy(data.pattern, "$pattern", PATTERN_SIZE);

  // Write to file
  fd = open("temp_input.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  assert( fd>0 && "Couldn't open output data file" );
  
  // Write section 1: pattern
  write_section_header(fd);
  write_string(fd, data.pattern, PATTERN_SIZE);
  
  // Write section 2: text
  write_section_header(fd);
  write_string(fd, data.input, STRING_SIZE);
  
  close(fd);

  return 0;
}
EOF

    # Compile
    cc -O3 -Wall -Wno-unused-label -I../../common_MachSuite \
        -o generate_temp \
        generate_temp.c ../../common_MachSuite/support.c 2>/dev/null
    
    # Generate data
    ./generate_temp
    mv temp_input.data "input_data/$size/input/input.data"
    
    echo "  ✓ Generated input_data/$size/input/input.data"
}

# Generate all datasets with different search patterns
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating Input Data (Different Search Patterns)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

generate_data "mini"        "bull" "Common word (original pattern)"
generate_data "small"       "tion" "Common suffix pattern"
generate_data "medium"      "John" "Proper name pattern"
generate_data "large"       "that" "Very common conjunction word"
generate_data "extra-large" "atte" "Substring pattern"

# Clean up
rm -f generate_temp generate_temp.c temp_input.data

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating Golden Output (check.data)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for size in mini small medium large extra-large; do
    echo "[$size] Running baseline to generate check.data..."
    
    # Run baseline (it will output to EX5_correctness/)
    $BASELINE_EXE \
        "input_data/$size/input/input.data" \
        "input_data/$size/input/input.data" 2>/dev/null || true
    
    # Copy output to check.data
    if [ -f "$CORR_OUTPUT" ]; then
        cp "$CORR_OUTPUT" "input_data/$size/output/check.data"
        
        # Extract match count for display
        matches=$(grep -v "^%%" "$CORR_OUTPUT" | head -1 || echo "?")
        echo "  ✓ Generated check.data (matches: $matches)"
    else
        echo "  ✗ Failed to generate check.data"
        exit 1
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Verifying All Datasets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

all_passed=true
for size in mini small medium large extra-large; do
    echo -n "[$size] Testing... "
    
    output=$($BASELINE_EXE \
        "input_data/$size/input/input.data" \
        "input_data/$size/output/check.data" 2>&1 || true)
    
    if echo "$output" | grep -q "Success"; then
        echo "✓ Success"
    else
        echo "✗ Failed"
        all_passed=false
    fi
done

echo ""
if [ "$all_passed" = true ]; then
    echo "🎉 All datasets generated and verified successfully!"
else
    echo "⚠️  Some datasets failed verification."
    exit 1
fi

# Generate README
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating Documentation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cat > input_data/README.md <<'EOFREADME'
# KMP-Mach Dataset Documentation

This directory contains test datasets for the **KMP-Mach** benchmark, which implements the Knuth-Morris-Pratt string matching algorithm.

---

## ⚠️ Important: About "Size"

Due to the algorithm's design and the use of fixed-size arrays defined by macros (e.g., `PATTERN_SIZE 4`, `STRING_SIZE 32411`), all test sets for `kmp-mach` have **exactly the same input size** (4-character pattern + 32,411-character text).

The different "sizes" (mini, small, medium, large, extra-large) represent different **search patterns** for correctness validation, NOT different computational scales. This approach allows for testing the algorithm's behavior with various patterns that may result in different numbers of matches.

---

## Dataset Structure

```
input_data/
├── mini/
│   ├── input/input.data     (6 lines: pattern + text)
│   └── output/check.data    (3 lines: match count)
├── small/
├── medium/
├── large/
├── extra-large/
└── README.md (this file)
```

---

## Dataset Details

| Size        | Pattern | Description                        | Expected Matches |
|-------------|---------|------------------------------------|--------------------|
| mini        | "bull"  | Common word (original pattern)     | ~1-5 matches       |
| small       | "tion"  | Common suffix pattern              | Many matches       |
| medium      | "John"  | Proper name pattern                | Few matches        |
| large       | "zzzz"  | Rare/absent pattern                | 0 or very few      |
| extra-large | "atte"  | Substring pattern                  | Multiple matches   |

**Key Points:**
- All datasets: Same text (32,411 characters from Theodore Roosevelt biography)
- Pattern size: 4 characters (fixed)
- Text source: `TR.txt` (Theodore Roosevelt text)
- Output: Single integer (number of pattern matches found)
- Algorithm: Knuth-Morris-Pratt (KMP) string matching

---

## Algorithm Information

### KMP (Knuth-Morris-Pratt) String Matching
- **Type**: Efficient string searching algorithm
- **Pattern**: 4 characters (fixed)
- **Text**: 32,411 characters (fixed)
- **Complexity**: O(n + m) where n = text length, m = pattern length
- **Optimization**: Preprocesses pattern to avoid redundant comparisons
- **Output**: Count of pattern occurrences in text

### Algorithm Features
The KMP algorithm:
- Preprocesses the pattern to build a "failure function" (kmpNext array)
- Uses the failure function to skip unnecessary character comparisons
- Achieves linear time complexity O(n + m)
- More efficient than naive string matching for certain patterns

### Text Content
The text (`TR.txt`) contains a biography snippet about Theodore Roosevelt (TR), discussing events like assassination attempts and his political career. This real-world text provides realistic testing for the string matching algorithm.

---

## Usage

### Test Single Size
```bash
./EX1_optimized_codes/kmp_gcc \
    input_data/mini/input/input.data \
    input_data/mini/output/check.data
```

Expected output: `Success.`

### Test All Sizes
```bash
for size in mini small medium large extra-large; do
    echo "Testing $size..."
    ./EX1_optimized_codes/kmp_gcc \
        input_data/$size/input/input.data \
        input_data/$size/output/check.data
done
```

### Run with Custom Executable
```bash
# Test an optimized variant
./EX1_optimized_codes/kmp_gcc_optimized \
    input_data/medium/input/input.data \
    input_data/medium/output/check.data
```

---

## Data Format

### Input Data (`input.data`)
```
%% Section 1
<4 characters: search pattern>

%% Section 2
<32411 characters: text to search in>
```

### Output Data (`check.data`)
```
%% Section 1
<1 integer: number of matches found>
```

### Example
Pattern: "bull"
Text: "...TRwasshotina**bull**et...wasshotinanassassination**bull**y..."
Output: 2 (if "bull" appears twice)

### Validation
- Exact integer match required
- No tolerance (exact count comparison)

---

## Regeneration

To regenerate all datasets from scratch:

```bash
# 1. Ensure baseline is compiled and TR.txt exists
make clean
make baseline

# 2. Run generation script
bash generate_all_data.sh
```

This will:
1. Create directory structure
2. Generate 5 different search patterns
3. Use baseline executable to compute match counts
4. Verify all datasets pass correctness tests

---

## Technical Notes

### Why Fixed Size?
The KMP algorithm implementation uses:
- Fixed 4-character pattern size (defined by `PATTERN_SIZE`)
- Fixed 32,411-character text size (defined by `STRING_SIZE`)
- Hardcoded array sizes: `char pattern[4]`, `char input[32411]`

Changing sizes would require modifying the algorithm parameters, not just input data.

### Pattern Variation
Different patterns test:
- **Common patterns** (e.g., "tion"): Test performance with many matches
- **Rare patterns** (e.g., "zzzz"): Test behavior with few/no matches
- **Real words** (e.g., "John", "bull"): Test with actual English text patterns
- **Substrings** (e.g., "atte"): Test partial word matching

### KMP Preprocessing
Each pattern requires preprocessing to build the failure function:
- The `kmpNext` array stores the longest proper prefix which is also a suffix
- This preprocessing takes O(m) time where m = pattern length
- The preprocessing result depends on the specific pattern

---

## File Sizes

Each dataset:
- `input.data`: ~32 KB (6 lines: 2 sections with pattern + text)
- `check.data`: ~6 bytes (3 lines: 1 section with single integer)

Total: ~160 KB for all 5 datasets

---

## Validation Results

All datasets have been verified to produce correct outputs:
- ✓ mini
- ✓ small
- ✓ medium
- ✓ large
- ✓ extra-large

---

## Algorithm Complexity

- **Preprocessing**: O(m) where m = PATTERN_SIZE = 4
- **Searching**: O(n) where n = STRING_SIZE = 32,411
- **Total**: O(m + n) = O(4 + 32,411) ≈ O(32,415) operations
- **Space**: O(m) for the kmpNext array

This is significantly more efficient than naive string matching which has O(n×m) complexity.

---

## References

- Implementation based on: http://www-igm.univ-mlv.fr/~lecroq/string/node8.html
- Knuth-Morris-Pratt algorithm: D. E. Knuth, J. H. Morris, and V. R. Pratt. "Fast pattern matching in strings." SIAM Journal on Computing, 1977.
EOFREADME

echo "  ✓ Generated input_data/README.md"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                     Generation Complete!                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Generated files:"
echo "  • 5 input.data files (different search patterns)"
echo "  • 5 check.data files (golden match counts)"
echo "  • input_data/README.md (documentation)"
echo ""
echo "All datasets verified successfully! ✓"

