# Memory Retrieval Skill

You are answering a question about a user's past conversations using a memory-decay server.

## Server API

The memory-decay server is running at the URL provided in the prompt. Use `curl` via Bash to interact with it.

### Search memories
```bash
curl -s http://localhost:8100/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "top_k": 20}'
```

Response contains `results` array, each with:
- `text`: the memory content
- `score`: relevance score (higher = more relevant)
- `created_tick`: relative time (higher = more recent)
- `speaker`: who said it
- `date`: calendar date (if enriched)

### Server info
```bash
curl -s http://localhost:8100/health
curl -s http://localhost:8100/stats
```

## How to Answer

1. **Read the provided context first.** Initial search results are already included in your prompt.

2. **Re-search if needed.** If the initial results don't fully answer the question:
   - Try different keywords or phrasings
   - Search for specific entities mentioned in the question
   - For "how many" questions, search for each item separately to ensure completeness

3. **Temporal reasoning:**
   - Each memory has a date. Use dates to determine chronological order.
   - For "how many days/months ago" questions: compute `today - event_date`. Show your math.
   - For "which came first" questions: compare dates, not result order.
   - Convert relative time references within memories to absolute dates.

4. **Knowledge updates:**
   - When multiple memories mention the same fact with different values, the MOST RECENT memory (latest date) has the current value.

5. **Inference:**
   - Read between the lines. Use context clues to infer unstated facts.
   - If someone discusses adoption without mentioning a partner, they're likely single.

6. **Preference questions:**
   - Reference the user's specific past experiences and successes.
   - Tailor advice to what worked for them before.

## Output Format

Give a concise, direct answer. Include specific dates, names, and facts. Do NOT include your reasoning process — just the answer.

If the memories truly contain no relevant information after searching, say "I don't know."
