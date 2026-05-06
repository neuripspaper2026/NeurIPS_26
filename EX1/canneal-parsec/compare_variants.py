#!/usr/bin/env python3
"""
Compare Multiple Canneal Variants for Correctness
Runs all variants and compares their outputs
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json


class VariantComparator:
    """Compare multiple canneal variants"""
    
    def __init__(self, size="mini", output_dir="correctness_test"):
        self.size = size
        self.output_dir = Path(output_dir) / size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_bin = None
        self.variant_bins = []
        self.params = {}
        self.results = {}
        
    def load_parameters(self):
        """Load test parameters from input_data"""
        params_file = Path(f"input_data/{self.size}/input/params.txt")
        if not params_file.exists():
            print(f"❌ Error: Parameters file not found: {params_file}")
            return False
        
        try:
            with open(params_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        self.params[key.strip()] = value.strip()
        except Exception as e:
            print(f"❌ Error reading parameters: {e}")
            return False
        
        return True
    
    def find_binaries(self):
        """Find baseline and variant binaries"""
        opt_dir = Path("EX1_optimized_codes")
        if not opt_dir.exists():
            print(f"❌ Error: Directory not found: {opt_dir}")
            return False
        
        # Find baseline
        baseline_candidates = list(opt_dir.glob("canneal_gcc"))
        if baseline_candidates:
            self.baseline_bin = baseline_candidates[0]
        else:
            print("❌ Error: Baseline binary not found (canneal_gcc)")
            return False
        
        # Find variants
        self.variant_bins = sorted(opt_dir.glob("canneal_gcc_*"))
        
        return True
    
    def run_binary(self, binary: Path, output_file: Path) -> bool:
        """Run a single binary and save output"""
        cmd = [
            str(binary),
            self.params.get('NTHREADS', '1'),
            self.params.get('NSWAPS', '1000'),
            self.params.get('TEMP', '2000'),
            self.params.get('NETLIST', f'input_data/{self.size}/input/input_{self.size}.nets'),
            self.params.get('NSTEPS', '50')
        ]
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            with open(output_file, 'w') as f:
                f.write(result.stdout)
            
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  Timeout (> 5 min)")
            return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def verify_output(self, output_file: Path) -> Tuple[bool, Dict]:
        """Verify output and extract metrics"""
        # Import the checker from verify_correctness
        sys.path.insert(0, str(Path(__file__).parent))
        from verify_correctness import CorrectnessChecker
        
        checker = CorrectnessChecker(output_file)
        success = checker.verify()
        
        metrics = {
            'success': success,
            'initial_cost': checker.initial_cost,
            'final_cost': checker.final_cost,
            'improvement': checker.improvement,
            'errors': checker.errors
        }
        
        return success, metrics
    
    def compare_with_baseline(self, variant_final: float, baseline_final: float) -> str:
        """Compare variant result with baseline"""
        if baseline_final is None or variant_final is None:
            return "N/A"
        
        diff_pct = abs(variant_final - baseline_final) / baseline_final * 100
        
        if diff_pct < 5:
            return f"{diff_pct:.2f}% (✅ Normal)"
        elif diff_pct < 10:
            return f"{diff_pct:.2f}% (⚠️  Acceptable)"
        else:
            return f"{diff_pct:.2f}% (❌ Large diff)"
    
    def run_comparison(self):
        """Run comparison on all variants"""
        print("=" * 70)
        print(f"Canneal Variant Comparison - Size: {self.size}")
        print("=" * 70)
        print()
        
        print("Test Parameters:")
        print(f"  Size:     {self.size}")
        print(f"  Netlist:  {self.params.get('NETLIST', 'N/A')}")
        print(f"  Swaps:    {self.params.get('NSWAPS', 'N/A')}")
        print(f"  Steps:    {self.params.get('NSTEPS', 'N/A')}")
        print()
        
        all_binaries = [self.baseline_bin] + self.variant_bins
        
        print("=" * 70)
        print("Running Tests...")
        print("=" * 70)
        print()
        
        for binary in all_binaries:
            binary_name = binary.name
            is_baseline = binary == self.baseline_bin
            
            print(f"{'[Baseline]' if is_baseline else '[Variant] '} {binary_name}")
            
            output_file = self.output_dir / f"{binary_name}.txt"
            
            # Run binary
            success = self.run_binary(binary, output_file)
            if not success:
                print("  ❌ Failed to run")
                self.results[binary_name] = {'success': False}
                print()
                continue
            
            # Verify output
            verified, metrics = self.verify_output(output_file)
            self.results[binary_name] = metrics
            
            if verified:
                print(f"  ✅ PASS - Final cost: {metrics['final_cost']:,.0f}")
            else:
                print(f"  ❌ FAIL")
                for error in metrics['errors']:
                    print(f"     • {error}")
            
            print()
        
        # Print comparison summary
        self.print_summary()
        
        # Save results to JSON
        self.save_results()
    
    def print_summary(self):
        """Print comparison summary"""
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print()
        
        baseline_name = self.baseline_bin.name
        baseline_result = self.results.get(baseline_name, {})
        baseline_final = baseline_result.get('final_cost')
        
        # Count successes/failures
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r.get('success', False))
        failed = total - passed
        
        print(f"Total binaries tested: {total}")
        print(f"✅ Passed: {passed}")
        if failed > 0:
            print(f"❌ Failed: {failed}")
        else:
            print(f"Failed: 0")
        print()
        
        # Print comparison table
        if baseline_final is not None:
            print("Comparison with Baseline:")
            print(f"{'Binary':<30} {'Final Cost':>15} {'Diff from Baseline':>20}")
            print("-" * 70)
            
            for binary_name, result in self.results.items():
                if not result.get('success'):
                    print(f"{binary_name:<30} {'FAILED':>15} {'-':>20}")
                else:
                    final_cost = result['final_cost']
                    diff_str = self.compare_with_baseline(final_cost, baseline_final)
                    print(f"{binary_name:<30} {final_cost:>15,.0f} {diff_str:>20}")
        
        print()
        print(f"Output files saved in: {self.output_dir}/")
        print()
        
        # Print important notes
        self.print_notes()
    
    def print_notes(self):
        """Print important notes about Canneal"""
        print("=" * 70)
        print("⚠️  Important Notes for Canneal Correctness")
        print("=" * 70)
        print()
        print("1. Canneal uses simulated annealing (随机优化算法)")
        print("   - 每次运行的Final cost会略有不同")
        print("   - 这是正常的，不是错误")
        print()
        print("2. Correctness标准：")
        print("   ✅ Final cost < Initial cost (优化有效)")
        print("   ✅ 程序正常完成，输出格式正确")
        print("   ✅ 没有segfault或其他错误")
        print()
        print("3. 不同variant的Final cost可能不同：")
        print("   - 如果差异<5%:  正常（随机性导致）")
        print("   - 如果差异5-10%: 可接受（编译优化/并行化）")
        print("   - 如果差异>10%:  需要检查")
        print()
    
    def save_results(self):
        """Save results to JSON file"""
        json_file = self.output_dir / "results.json"
        
        try:
            with open(json_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Results saved to: {json_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save JSON results: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Canneal variants for correctness"
    )
    parser.add_argument(
        "size",
        nargs="?",
        default="mini",
        help="Size to test (mini, small, medium, large, extra-large)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="correctness_test",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    comparator = VariantComparator(size=args.size, output_dir=args.output_dir)
    
    # Load parameters
    if not comparator.load_parameters():
        sys.exit(1)
    
    # Find binaries
    if not comparator.find_binaries():
        sys.exit(1)
    
    # Run comparison
    comparator.run_comparison()
    
    # Exit with success if all tests passed
    all_passed = all(r.get('success', False) for r in comparator.results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

