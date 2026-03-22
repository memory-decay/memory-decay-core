# MemoryBench 평가 가이드

## 1. 준비물

- Python 3.10+
- [Bun](https://bun.sh) (`curl -fsSL https://bun.sh/install | bash`)
- `GEMINI_API_KEY` (임베딩용)
- `OPENAI_API_KEY` (GPT-4o judge용)

## 2. 설치

```bash
# memory-decay 리포
git clone https://github.com/tmdgusya/memory-decay.git
cd memory-decay
pip install -e .

# memorybench 리포 (별도 디렉토리)
cd ..
git clone https://github.com/tmdgusya/memorybench.git
cd memorybench
bun install

# API 키 설정
cat > .env.local << 'EOF'
OPENAI_API_KEY=sk-여기에키입력
EOF
```

## 3. 실행

터미널 2개가 필요합니다.

**터미널 1 — memory-decay 서버:**
```bash
cd memory-decay
export GEMINI_API_KEY=여기에키입력
python -m memory_decay.server --port 8100 --experiment-dir experiments/exp_lme_0292
```
`Uvicorn running on http://127.0.0.1:8100` 뜨면 준비 완료.

**터미널 2 — 벤치마크 실행:**
```bash
cd memorybench

# 50문제 평가 (~3시간)
bun run src/index.ts run -p memory-decay -b longmemeval -j gpt-4o -r eval-50q --limit 50

# 또는 5문제 빠른 테스트 (~20분)
bun run src/index.ts run -p memory-decay -b longmemeval -j gpt-4o -r smoke-5 --limit 5
```

## 4. 결과 확인

```bash
# 요약
bun run src/index.ts status -r eval-50q

# 실패한 질문 분석
bun run src/index.ts show-failures -r eval-50q

# 웹 대시보드
bun run src/index.ts serve
# → http://localhost:3000 에서 확인
```

## 5. 중간에 끊겼을 때

체크포인트가 자동 저장됩니다. 같은 run ID로 다시 실행하면 이어서 진행:

```bash
# 서버 다시 시작하고
python -m memory_decay.server --port 8100 --experiment-dir experiments/exp_lme_0292

# 같은 run ID로 재실행 → 체크포인트에서 resume
bun run src/index.ts run -p memory-decay -b longmemeval -j gpt-4o -r eval-50q --limit 50
```

## 6. 다른 provider와 비교

```bash
# RAG provider (OpenAI 임베딩 기반)
bun run src/index.ts run -p rag -b longmemeval -j gpt-4o -r rag-50q --limit 50

# filesystem provider (CLAUDE.md 스타일)
bun run src/index.ts run -p filesystem -b longmemeval -j gpt-4o -r fs-50q --limit 50
```

## 참고

- 첫 실행은 Gemini 임베딩 API 호출이 많아서 느림 (~3.5분/질문)
- 서버의 임베딩 캐시가 메모리에 유지되므로, **서버를 안 끄면** 두 번째 실행부터 빨라짐 (~6초/질문)
- `report.json`에 retrieval metrics (Hit@K, MRR, NDCG) + QA accuracy 포함
