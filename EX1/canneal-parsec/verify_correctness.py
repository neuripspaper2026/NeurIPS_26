#!/usr/bin/env python3
"""
Canneal Correctness Verification Tool
Verifies that canneal output is correct and valid
"""

import sys
import re
import argparse
from pathlib import Path


class CorrectnessChecker:
    """Checker for Canneal output correctness"""
    
    def __init__(self, output_file):
        self.output_file = Path(output_file)
        self.errors = []
        self.warnings = []
        self.initial_cost = None
        self.final_cost = None
        self.improvement = None
        
    def check_file_exists(self):
        """Check if output file exists and is not empty"""
        if not self.output_file.exists():
            self.errors.append(f"File not found: {self.output_file}")
            return False
        
        if self.output_file.stat().st_size == 0:
            self.errors.append("Output file is empty")
            return False
        
        return True
    
    def check_header(self, content):
        """Check for PARSEC header"""
        if "PARSEC Benchmark Suite" not in content:
            self.errors.append("Missing PARSEC Benchmark Suite header")
            return False
        return True
    
    def check_netlist_created(self, content):
        """Check netlist creation confirmation"""
        pattern = r"netlist created\.\s+(\d+)\s+elements"
        match = re.search(pattern, content)
        if not match:
            self.errors.append("Netlist creation not confirmed")
            return False
        
        num_elements = int(match.group(1))
        if num_elements == 0:
            self.errors.append("Zero elements in netlist")
            return False
        
        return True
    
    def extract_costs(self, content):
        """Extract initial and final costs"""
        # Extract initial cost
        initial_pattern = r"Initial cost:\s+([\d.e+]+)"
        initial_match = re.search(initial_pattern, content)
        if not initial_match:
            self.errors.append("Initial cost not found")
            return False
        
        try:
            self.initial_cost = float(initial_match.group(1))
        except ValueError:
            self.errors.append(f"Invalid initial cost format: {initial_match.group(1)}")
            return False
        
        # Extract final cost
        final_pattern = r"Final cost:\s+([\d.e+]+)"
        final_match = re.search(final_pattern, content)
        if not final_match:
            self.errors.append("Final cost not found")
            return False
        
        try:
            self.final_cost = float(final_match.group(1))
        except ValueError:
            self.errors.append(f"Invalid final cost format: {final_match.group(1)}")
            return False
        
        # Extract improvement
        improvement_pattern = r"Improvement:\s+([\d.e+-]+)"
        improvement_match = re.search(improvement_pattern, content)
        if not improvement_match:
            self.errors.append("Improvement not found")
            return False
        
        try:
            self.improvement = float(improvement_match.group(1))
        except ValueError:
            self.errors.append(f"Invalid improvement format: {improvement_match.group(1)}")
            return False
        
        return True
    
    def check_optimization_worked(self):
        """Check if optimization actually improved the cost"""
        if self.initial_cost is None or self.final_cost is None:
            return False
        
        if self.final_cost >= self.initial_cost:
            self.errors.append(
                f"Optimization failed: Final cost ({self.final_cost:.2f}) >= "
                f"Initial cost ({self.initial_cost:.2f})"
            )
            return False
        
        if self.improvement <= 0:
            self.errors.append(f"Improvement is not positive: {self.improvement}")
            return False
        
        return True
    
    def verify(self):
        """Run all verification checks"""
        if not self.check_file_exists():
            return False
        
        # Read file content
        try:
            with open(self.output_file, 'r') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Error reading file: {e}")
            return False
        
        # Run all checks
        checks = [
            self.check_header(content),
            self.check_netlist_created(content),
            self.extract_costs(content),
            self.check_optimization_worked()
        ]
        
        return all(checks)
    
    def print_report(self, verbose=True):
        """Print verification report"""
        print("=" * 70)
        print("Canneal Correctness Verification")
        print("=" * 70)
        print(f"File: {self.output_file}")
        print()
        
        if not self.errors:
            print("✅ ALL CHECKS PASSED - Output is CORRECT")
            print()
            if self.initial_cost is not None and self.final_cost is not None:
                print("Summary:")
                print(f"  Initial Cost: {self.initial_cost:,.0f}")
                print(f"  Final Cost:   {self.final_cost:,.0f}")
                print(f"  Improvement:  {self.improvement:,.0f} "
                      f"({(self.improvement/self.initial_cost*100):.2f}%)")
        else:
            print("❌ VERIFICATION FAILED")
            print()
            print("Errors:")
            for error in self.errors:
                print(f"  ❌ {error}")
        
        if self.warnings and verbose:
            print()
            print("Warnings:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
        
        print("=" * 70)
        print()
        
        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify Canneal output correctness"
    )
    parser.add_argument(
        "output_file",
        help="Path to canneal output file"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (less verbose)"
    )
    
    args = parser.parse_args()
    
    checker = CorrectnessChecker(args.output_file)
    success = checker.verify()
    checker.print_report(verbose=not args.quiet)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

