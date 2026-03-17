"""SyntheticDataGenerator: Anthropic-powered synthetic memory data creation.

Uses the Anthropic Claude API (Haiku for cost efficiency) to generate realistic
memory items and recall test queries in JSONL format.

Each memory item includes: id, type (fact/episode), content, entities, tick,
impact (0.1-1.0), associations (related memory ids), recall_query, recall_answer.

The association graph is designed so some memories act as "hubs" (frequently
referenced by many others) and some are "leaves" (isolated, few connections).

Key methods (to implement):
- generate_dataset(): produce a JSONL file of synthetic memory items
- generate_recall_tests(): produce held-out recall queries for evaluation
"""
