## H2 — Gold Data Preparation

### Dataset
- TermA processed: 5,000 sentences
- Development split: 3,000 TermA train sentences
- Final local test: 1,000 TermA validation sentences
- Official TermA test tidak digunakan untuk local gold metrics.

### Explicit Aspect–Opinion Gold
- Gold review: 1,314 sentences
- Gold aspect–opinion pairs: 3,202 pairs
- Duplicate pair: 0
- Missing core values: 0
- Span mismatch: 0
- Validation: PASS

### Polarity Gold
- Gold polarity: 1,000 aspect–opinion pairs
- Validation: PASS

### Implicit Aspect Gold
- Implicit candidates: 451
- Valid implicit gold rows: 407
- Validation: PASS

### Split Policy
- TermA train → development
- TermA validation → locked final evaluation
- Random seed: 42

### H2 Status
PASS