#!/usr/bin/env python3
"""
Example: How to use Canneal correctness verification tools in Python code

This example demonstrates:
1. Running canneal binary
2. Verifying output correctness
3. Comparing multiple variants
4. Integrating into test frameworks
"""

import subprocess
from pathlib import Path
from verify_correctness import CorrectnessChecker


def example1_verify_single_output():
    """Example 1: Verify a single output file"""
    print("=" * 70)
    print("Example 1: Verify Single Output")
    print("=" * 70)
    
    # Create checker
    checker = CorrectnessChecker("test.txt")
    
    # Run verification
    success = checker.verify()
    
    # Get results
    if success:
        print(f"\n✅ Verification passed!")
        print(f"   Initial cost: {checker.initial_cost:,.0f}")
        print(f"   Final cost:   {checker.final_cost:,.0f}")
        print(f"   Improvement:  {checker.improvement:,.0f} ({checker.improvement/checker.initial_cost*100:.2f}%)")
    else:
        print(f"\n❌ Verification failed!")
        print("Errors:")
        for error in checker.errors:
            print(f"  - {error}")
    
    return success


def example2_run_and_verify():
    """Example 2: Run binary and verify output"""
    print("\n" + "=" * 70)
    print("Example 2: Run Binary and Verify")
    print("=" * 70)
    
    # Define test parameters
    binary = "EX1_optimized_codes/canneal_gcc"
    params = ["1", "1000", "2000", "input_data/mini/input/input_mini.nets", "50"]
    output_file = "example_output.txt"
    
    # Check if binary exists
    if not Path(binary).exists():
        print(f"❌ Binary not found: {binary}")
        return False
    
    # Run binary
    print(f"\n🚀 Running: {binary} {' '.join(params)}")
    try:
        result = subprocess.run(
            [binary] + params,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60
        )
        
        # Save output
        with open(output_file, 'w') as f:
            f.write(result.stdout)
        
        print(f"✅ Output saved to: {output_file}")
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout!")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Verify output
    print("\n🔍 Verifying output...")
    checker = CorrectnessChecker(output_file)
    success = checker.verify()
    
    if success:
        print(f"✅ Verification passed - Final cost: {checker.final_cost:,.0f}")
    else:
        print("❌ Verification failed")
        for error in checker.errors:
            print(f"  - {error}")
    
    return success


def example3_compare_multiple_outputs():
    """Example 3: Compare multiple output files"""
    print("\n" + "=" * 70)
    print("Example 3: Compare Multiple Outputs")
    print("=" * 70)
    
    # List of output files to compare (assuming they exist)
    output_files = [
        "test.txt",
        "example_output.txt"
    ]
    
    results = {}
    
    # Verify each file
    for output_file in output_files:
        if not Path(output_file).exists():
            print(f"⚠️  Skipping {output_file} (not found)")
            continue
        
        checker = CorrectnessChecker(output_file)
        success = checker.verify()
        
        results[output_file] = {
            'success': success,
            'initial_cost': checker.initial_cost,
            'final_cost': checker.final_cost,
            'improvement': checker.improvement
        }
    
    # Print comparison
    print("\nComparison Results:")
    print(f"{'File':<30} {'Status':<10} {'Final Cost':>15} {'Improvement %':>15}")
    print("-" * 70)
    
    for filename, result in results.items():
        if result['success']:
            status = "✅ PASS"
            final_cost = f"{result['final_cost']:,.0f}"
            improvement_pct = f"{result['improvement']/result['initial_cost']*100:.2f}%"
        else:
            status = "❌ FAIL"
            final_cost = "-"
            improvement_pct = "-"
        
        print(f"{filename:<30} {status:<10} {final_cost:>15} {improvement_pct:>15}")
    
    # Calculate differences
    if len(results) >= 2:
        baseline_file = list(results.keys())[0]
        baseline_cost = results[baseline_file]['final_cost']
        
        print("\nDifference from baseline:")
        for filename, result in results.items():
            if filename == baseline_file:
                continue
            
            if result['success'] and result['final_cost'] is not None:
                diff_pct = abs(result['final_cost'] - baseline_cost) / baseline_cost * 100
                
                if diff_pct < 5:
                    status = "✅ Normal"
                elif diff_pct < 10:
                    status = "⚠️  Acceptable"
                else:
                    status = "❌ Large diff"
                
                print(f"  {filename}: {diff_pct:.2f}% {status}")
    
    return all(r['success'] for r in results.values())


def example4_integrate_with_test_framework():
    """Example 4: Integration with test framework"""
    print("\n" + "=" * 70)
    print("Example 4: Test Framework Integration")
    print("=" * 70)
    
    def test_canneal_binary(binary_name, params):
        """Test a single canneal binary"""
        binary_path = f"EX1_optimized_codes/{binary_name}"
        
        if not Path(binary_path).exists():
            return {
                'binary': binary_name,
                'success': False,
                'error': 'Binary not found'
            }
        
        # Run binary
        try:
            result = subprocess.run(
                [binary_path] + params,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60
            )
            
            # Save output
            output_file = f"test_{binary_name}.txt"
            with open(output_file, 'w') as f:
                f.write(result.stdout)
            
            # Verify
            checker = CorrectnessChecker(output_file)
            success = checker.verify()
            
            return {
                'binary': binary_name,
                'success': success,
                'final_cost': checker.final_cost,
                'improvement': checker.improvement,
                'errors': checker.errors
            }
            
        except Exception as e:
            return {
                'binary': binary_name,
                'success': False,
                'error': str(e)
            }
    
    # Test parameters
    test_params = ["1", "1000", "2000", "input_data/mini/input/input_mini.nets", "50"]
    
    # Find binaries to test
    binaries_to_test = ["canneal_gcc"]
    
    # Add variants if they exist
    opt_dir = Path("EX1_optimized_codes")
    if opt_dir.exists():
        for variant in opt_dir.glob("canneal_gcc_*"):
            binaries_to_test.append(variant.name)
    
    print(f"\nTesting {len(binaries_to_test)} binaries...")
    
    # Run tests
    test_results = []
    for binary in binaries_to_test:
        print(f"  Testing {binary}...", end=" ")
        result = test_canneal_binary(binary, test_params)
        test_results.append(result)
        
        if result['success']:
            print(f"✅ PASS (cost: {result['final_cost']:,.0f})")
        else:
            error_msg = result.get('error', 'Verification failed')
            print(f"❌ FAIL ({error_msg})")
    
    # Summary
    passed = sum(1 for r in test_results if r['success'])
    failed = len(test_results) - passed
    
    print(f"\nTest Summary:")
    print(f"  Total:  {len(test_results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    return failed == 0


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("Canneal Correctness Verification - Usage Examples")
    print("=" * 70)
    
    # Run examples
    results = []
    
    try:
        results.append(("Example 1", example1_verify_single_output()))
    except Exception as e:
        print(f"❌ Example 1 failed: {e}")
        results.append(("Example 1", False))
    
    try:
        results.append(("Example 2", example2_run_and_verify()))
    except Exception as e:
        print(f"❌ Example 2 failed: {e}")
        results.append(("Example 2", False))
    
    try:
        results.append(("Example 3", example3_compare_multiple_outputs()))
    except Exception as e:
        print(f"❌ Example 3 failed: {e}")
        results.append(("Example 3", False))
    
    try:
        results.append(("Example 4", example4_integrate_with_test_framework()))
    except Exception as e:
        print(f"❌ Example 4 failed: {e}")
        results.append(("Example 4", False))
    
    # Final summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{name:<20} {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()

