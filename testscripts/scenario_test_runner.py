#!/usr/bin/env python3
# scenario_test_runner.py - Comprehensive cross-scenario testing

import asyncio
from benchmark_config import BENCHMARK_SCENARIOS
from benchmark_runner import BenchmarkRunner
from test_metrics import NavigationEvaluator
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

class ScenarioTester:
    def __init__(self):
        self.runner = BenchmarkRunner()
        self.results = []
        self.metrics = {
            'success': [],
            'steps': [],
            'time': [],
            'confidence': [],
            'recoveries': [],
            'memory': []
        }
    
    async def run_all_scenarios(self):
        """Run all benchmark scenarios with metrics collection"""
        for scenario in BENCHMARK_SCENARIOS:
            result = await self.runner._run_saam_navigation(scenario)
            self._record_results(scenario, result)
            self._generate_scenario_report(scenario, result)
        
        self._generate_comparative_analysis()
    
    def _record_results(self, scenario, result):
        """Store results for comparative analysis"""
        self.results.append({
            'scenario': scenario.name,
            'complexity': scenario.complexity,
            'result': result
        })
        
        # Update metrics
        self.metrics['success'].append(int(result['success']))
        self.metrics['steps'].append(result['steps'])
        self.metrics['time'].append(result['time_elapsed'])
        self.metrics['confidence'].append(result.get('avg_confidence', 0))
        self.metrics['recoveries'].append(result.get('recoveries', 0))
        self.metrics['memory'].append(result.get('memory_used', 0)/1024)  # KB
    
    def _generate_scenario_report(self, scenario, result):
        """Generate detailed report for a single scenario"""
        evaluator = NavigationEvaluator(
            scenario.grid_size,
            {
                'expected_grid': "",  # Would come from actual test data
                'optimal_steps': scenario.optimal_steps
            }
        )
        
        # Create test case (would be populated with actual data)
        test_case = LLMTestCase(
            input="", 
            actual_output="",
            expected_output="",
            context={}
        )
        
        evaluation = evaluator.evaluate_run(test_case)
        
        print(f"\n=== SCENARIO REPORT: {scenario.name} [{scenario.complexity}] ===")
        print(f"Status: {'SUCCESS' if result['success'] else 'FAIL'}")
        print(f"Steps: {result['steps']} (Optimal: {scenario.optimal_steps})")
        print(f"Time: {result['time_elapsed']:.2f}s")
        print(f"Recoveries: {result.get('recoveries', 0)}")
        print(f"\nEvaluation Score: {evaluation.overall_score:.2f}/1.0")
        
        # Print recommendations if score is low
        if evaluation.overall_score < 0.7:
            print("\nRecommendations:")
            for rec in evaluation.recommendations:
                print(f"- {rec}")
    
    def _generate_comparative_analysis(self):
        """Generate comparative metrics across all scenarios"""
        df = pd.DataFrame({
            'Scenario': [r['scenario'] for r in self.results],
            'Complexity': [r['complexity'] for r in self.results],
            'Success': self.metrics['success'],
            'Steps': self.metrics['steps'],
            'Time': self.metrics['time'],
            'Confidence': self.metrics['confidence'],
            'Recoveries': self.metrics['recoveries'],
            'Memory': self.metrics['memory']
        })
        
        # Group by complexity
        complexity_groups = df.groupby('Complexity').mean()
        
        print("\n=== COMPARATIVE ANALYSIS ===")
        print("\nPerformance by Complexity Level:")
        print(complexity_groups)
        
        # Generate visualizations
        self._plot_metrics(df)
    
    def _plot_metrics(self, df):
        """Generate visualizations of key metrics"""
        plt.figure(figsize=(15, 10))
        
        # Success Rate by Complexity
        plt.subplot(2, 2, 1)
        success_rates = df.groupby('Complexity')['Success'].mean()
        success_rates.plot(kind='bar', color=['green', 'orange', 'red'])
        plt.title('Success Rate by Complexity')
        plt.ylabel('Success Rate')
        
        # Steps vs Optimal
        plt.subplot(2, 2, 2)
        plt.scatter(df['Complexity'], df['Steps'], alpha=0.6)
        plt.title('Steps Taken by Complexity')
        plt.ylabel('Steps')
        
        # Time vs Complexity
        plt.subplot(2, 2, 3)
        plt.scatter(df['Complexity'], df['Time'], alpha=0.6)
        plt.title('Execution Time by Complexity')
        plt.ylabel('Time (s)')
        
        # Recoveries vs Confidence
        plt.subplot(2, 2, 4)
        plt.scatter(df['Recoveries'], df['Confidence'], c=df['Complexity'].map({
            'simple': 'green',
            'moderate': 'orange',
            'complex': 'red'
        }))
        plt.title('Confidence vs Recovery Attempts')
        plt.xlabel('Recoveries')
        plt.ylabel('Confidence')
        
        plt.tight_layout()
        plt.savefig('scenario_metrics.png')
        print("\nSaved metrics visualization to scenario_metrics.png")

async def main():
    tester = ScenarioTester()
    await tester.run_all_scenarios()

if __name__ == "__main__":
    asyncio.run(main())
