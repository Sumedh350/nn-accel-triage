# Triage Engine Directory
- feature_extractor.py   → parses raw sim artifacts into feature vectors
- clusterer.py           → rule-based pre-bucketing + DBSCAN
- triage_agent.py        → calls Claude API, builds context, emits structured reports
- rag_store.py           → vector store of historical failures
- Input: regression_db.jsonl
- Output: triage_report.json per cluster