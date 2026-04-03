import json
import numpy as np
from collections import defaultdict

def generate_report(file_path="audit_trail.jsonl"):
    e2e_latencies = []
    e2e_costs = []
    node_data = defaultdict(list)
    query_latencies = defaultdict(float) 
    query_costs = defaultdict(float)
    query_status_history = defaultdict(list)

    try:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                q_type= data.get("metadata","N/A")
                q_text = data.get("query","uknown_query")
                status = data.get("audit_status", "N/A")

                query_latencies[q_text] += data['run_stats']['total_latency']
                query_costs[q_text] += data['run_stats']['total_cost']
                query_status_history[q_text].append(status)

                # 2. Track Per-Node Latency from performance_breakdown
                # Use .items() to iterate through the dictionary
                benchmarks = data.get('performance_breakdown', {})
                for node_name, metrics in benchmarks.items():
                    node_data[node_name].append(metrics['latency'])

        e2e_latencies = list(query_latencies.values())
        e2e_costs = list(query_costs.values())

        if not e2e_latencies:
            print("No data found in log file.")
            return

        print(f"\n{'--- PERFORMANCE REPORT (BATCH RUN) ---':^60}")
        print(f"{'NODE NAME':<20} | {'AVG':<7} | {'P95':<7} | {'P99':<7}")
        print("-" * 60)

        # Calculate and Print for each node
        for node_name, times in node_data.items():
            avg = np.mean(times)
            p95 = np.percentile(times, 95)
            p99 = np.percentile(times, 99)
            print(f"{node_name:<20} | {avg:>6.2f}s | {p95:>6.2f}s | {p99:>6.2f}s")
        
        # Calculate and Print Overall E2E
        total_avg = np.mean(e2e_latencies)
        total_p95 = np.percentile(e2e_latencies, 95)
        total_p99 = np.percentile(e2e_latencies, 99)
        total_cost_sum = sum(e2e_costs)

        print("-" * 60)
        print(f"{'OVERALL E2E':<20} | {total_avg:>6.2f}s | {total_p95:>6.2f}s | {total_p99:>6.2f}s")
        print(f"{'TOTAL BATCH COST':<20} | ${total_cost_sum:>6.4f}")
        print("-" * 60)

    except FileNotFoundError:
        print(f"Error: {file_path} not found.")

if __name__ == "__main__":
    generate_report("audit_trail.jsonl")