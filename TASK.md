# MemoryBench 실험 진행 상황 (2026-03-24)

## 완료

- [x] memorybench pull (4 new commits: agent mode, hybrid chunking, prompt v2, metrics)
- [x] memory-retrieval skill 복원 (`skills/memory-retrieval/skill.md`)
- [x] `cache/openai/embeddings.pkl` symlink 생성 (서버 시작 문제 해결)
- [x] exp_bench_0003 Stage A: **LongMemEval 80% (8/10)** — baseline(70%) 대비 +10%p 개선
- [x] exp_bench_0004 Stage A: **LongMemEval 0% (0/10)** — 실패 (empty answers), 폐기

## 다음 할 일

### 1. exp_bench_0003 Stage B 실행
baseline보다 개선된 exp_bench_0003을 20q × 3 벤치마크로 확인.

```bash
# 서버 시작
.venv/bin/python -m memory_decay.server --port 8100 \
  --cache-dir cache/openai \
  --embedding-provider openai --embedding-api-key "$OPENAI_API_KEY" \
  --experiment-dir experiments/exp_bench_0003 &

# 3개 벤치마크 순차 실행 (각 20q, agent mode, gpt-4o judge)
cd ~/memorybench  # 실제 경로: ~/.openclaw/workspace/memorybench
for BENCH in longmemeval locomo convomem; do
  OPENAI_API_KEY="$OPENAI_API_KEY" MEMORY_DECAY_AGENT_MODE=1 bun run src/index.ts run \
    -p memory-decay -b $BENCH -j gpt-4o -m sonnet-4 \
    -r exp_bench_0003-stageB-$BENCH -l 20 --force
done
```

### 2. 결과 판정
```
bench_score = 0.50 × longmemeval_acc + 0.30 × locomo_acc + 0.20 × convomem_acc
```
- bench_score > 0.85 (현재 best) → **accept**, best symlink 업데이트
- bench_score ≤ 0.85 → **reject**

### 3. bench_results.json 저장 & memory chain 기록
program.md Step 5~8 참고.

### 4. 다음 실험 설계
exp_bench_0003이 accept되면 그 위에서 추가 파라미터 탐색.
reject되면 다른 방향 모색.

## 중요 규칙 (실수 방지)
- **answer: Claude Code agent mode** (`MEMORY_DECAY_AGENT_MODE=1`)
- **judge: gpt-4o** (`-j gpt-4o`, OpenAI API key 필요)
- **Anthropic API key 불필요** — 절대 참조하지 말 것
- 서버 cache 경로: `cache/openai` (symlink 있음)
- program.md 전체 프로토콜 반드시 따를 것
