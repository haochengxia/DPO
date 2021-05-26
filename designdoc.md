# Algo

1. 最为简单的epsilon调整方式，划分k轮，贪心法每次分配$\epsilon / k$
2. dprl，通过rl生成epsilon选取价值，提供离散化的选取或归一化分布选取
3. fdprl，通过分布式的方式在多轮次中分散epsilon，用相对更高的通信开销，降低对数据可用性的影响
