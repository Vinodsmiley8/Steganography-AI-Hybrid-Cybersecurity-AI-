
#!/usr/bin/env bash
python stego_ai_hybrid.py embed --cover examples/covers/cover1.png --payload examples/payloads/secret.txt --out examples/covers/stego_test.png
python stego_ai_hybrid.py extract --stego examples/covers/stego_test.png --out examples/payloads/extracted.txt
if cmp -s examples/payloads/secret.txt examples/payloads/extracted.txt; then
  echo "SMOKE TEST PASSED: extracted payload matches original"
else
  echo "SMOKE TEST FAILED: mismatch"
fi
