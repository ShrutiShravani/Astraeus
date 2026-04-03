import mlflow

def test_retriever_hallucination():
    results=run_benchmarks(agent)
    avg_faithfulness= sum([r.faithfulness for r in results])/len(results)
    #log to mlflow
    mlflow.log_metric("benchmark_avg_faithfulnes",avg_faithfulness)
    assert avg_faithfulness>0.90,f"Hallucination risk too high: {avg_faithfulness}"

def tets_genertaed_reprot_hallucination():
    