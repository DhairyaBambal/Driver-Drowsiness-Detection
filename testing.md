# Testing and Results

## Test Cases

| Test Case | Expected Result | Actual Result | Status |
|---|---|---|---|
| Eyes open | System shows ALERT | ALERT displayed | PASS |
| Eyes closed for prolonged period | System detects drowsiness | DROWSY displayed | PASS |
| Drowsiness detected | Emergency alarm activates | Alarm activated | PASS |
| Eyes reopen | System returns to ALERT | ALERT displayed | PASS |
| Drowsy event occurs | Event counter increases | Counter increased | PASS |
| Webcam monitoring | Live video displayed | Live video displayed | PASS |

## Result

The system successfully detected prolonged eye closure and generated a visual and audio warning in real time.