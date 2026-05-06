# Patricia-mibench Size Update (2× Expansion)

## New Dataset Sizes

| Size | Packets | Unique IPs | Ports | File Size | Runtime |
|------|---------|------------|-------|-----------|---------|
| mini | 62,721 | 1,200 | 1,024 | 1.6 MB | ~0.05s |
| small | 900,000 | 8,000 | 8,192 | 25.9 MB | ~0.78s |
| medium | 3,000,000 | 15,000 | 16,384 | 91.9 MB | ~2.72s |
| large | 12,000,000 | 40,000 | 32,768 | 390.1 MB | ~12.45s |
| extra-large | 30,000,000 | 80,000 | 65,535 | 1002.8 MB | ~54.0s |

## Performance Summary

- **mini**: 1.17M pkt/s
- **small**: 1.16M pkt/s  
- **medium**: 1.10M pkt/s
- **large**: 964K pkt/s
- **extra-large**: 556K pkt/s

## Total Changes
- mini: 保持不变
- small: 150K → 450K → 900K (6× 原始)
- medium: 500K → 1.5M → 3M (6× 原始)
- large: 2M → 6M → 12M (6× 原始)
- extra-large: 5M → 15M → 30M (6× 原始)
