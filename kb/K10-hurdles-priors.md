# K10 Hurdles 起点候选（零训练，8–39）

| 起点 | 结果 | 完赛 |
|---|---|---|
| `hurdles-fast-walker@1` | **2.84 s** | 62/64（8–71） |
| Steps 冠军策略（walk，head 0.72） | 3.05 s | 28/32 |
| `steps-climber@1` | 5.09 s | 31/32 |
| `wheel-uphill-runner-opus@1` | 15.13 s | 12/32 |
| Sprint 冠军/raw A1（raw host） | 27.8 / 26.6 s | 3–4/32 |

结论：HFW 是唯一能跨栏的库内起点；跨任务迁移（Steps 冠军、wheel）都不行。冠军 2.32 s 需要一个更快的跨障训练策略，而非换起点。
