# Bayesian-b: Log-Odds 占据栅格地图

用 Go(仅标准库)实现的占据栅格地图(occupancy grid mapping),基于对数几率
(log-odds)累加更新,内置数值饱和保护、测距线自由空间/终点占据的正确区分,
以及传感器逆模型过度自信的自动检测。仿真结果导出为**单一静态 HTML 文件**
(原生 HTML/CSS/JavaScript,无任何框架与第三方库),浏览器打开即可逐帧回放。

## 运行

```sh
go run ./cmd/ogmap          # 生成 occupancy_map.html 并打印过度自信检测报告
go test ./...               # 运行全部测试
```

用浏览器打开 `occupancy_map.html`:可切换"合理模型 / 过度自信模型"两个场景,
播放/单步/拖动进度条逐帧查看每个栅格占据概率的变化(绿=空闲,灰=未知,红=占据),
并可叠加真实地图对照。右侧面板实时显示极端概率栅格数与过度自信告警。

## 对数几率模型

每个栅格维护占据概率的 log-odds:

```
l_t = l_{t-1} + log( p(m|z) / (1 - p(m|z)) )
```

贝叶斯更新在 log-odds 域退化为**加法**,因此多次观测只需累加,先验 l_0 = 0
对应 p = 0.5(未知)。概率由数值稳定的 sigmoid 还原(`ProbabilityOf`,
对 |l| 很大的输入不会溢出):

```
p = 1 / (1 + e^{-l})
```

## 传感器逆模型假设

每条测距线(beam)提供两类证据:

- **自由空间**:beam 穿过的栅格,按 `p_free < 0.5` 施加负增量
  `LFree = log(p_free/(1-p_free))`;
- **终点占据**:beam 命中的终点栅格,按 `p_occ > 0.5` 施加正增量
  `LOcc = log(p_occ/(1-p_occ))`;若该 beam 未命中(max-range / 无回波),
  终点栅格同样按空闲处理。

假设:各次观测条件独立(标准占据栅格假设)、传感器噪声为高斯测距噪声加少量
幻影回波(spurious return)、机器人在每次扫描期间静止。

**更新顺序**:`UpdateScan` 采用两阶段提交——先应用整帧扫描所有 beam 的自由
空间更新,再应用所有终点占据更新。因此终点栅格绝不会被自己 beam(或同一帧
中穿过它的其他 beam)的自由空间证据覆盖。

## 数值边界处理(饱和)

log-odds 累加器被钳制在 `[LMin, LMax]`(默认 ±6,见 `DefaultLMin/DefaultLMax`):

1. **不溢出**:任意次重复观测后累加器保持有限,sigmoid 输入有界;
2. **可恢复**:饱和不等于"锁死"。单次相反观测仍使累加器精确移动
   `|LFree|`(或 `|LOcc|`),从完全饱和翻回 p < 0.5 至多需要
   `ceil(LMax / |LFree|) + 1` 次相反观测——有明确上界,而非无限。

## 过度自信检测

若逆模型参数过于极端(如 `p_occ=0.99, p_free=0.01`),单次观测的增量就接近
饱和界,一两个噪声测量即可把本应不确定的栅格推到 p≈0 或 p≈1。`Grid.Analyze`
统计每帧:

- `Extreme`:概率超出 `[1-e, e]`(默认 e=0.95)的已观测栅格数;
- `Premature`:观测次数不足 `MinObs`(默认 6)就已达极端概率的栅格数——
  **异常收敛的直接证据**;
- `Saturated`:被钳制在饱和界上的栅格数。

合理模型下 Premature 占比接近 0(仅地图边缘掠射观测处有零星个例);过度自信
模型下噪声幻影回波会被一次观测"钉死"在极端概率,Premature 占比急剧上升
(仿真中约 0.3% vs 40%),占比超过 5% 即判定为异常收敛。

## 测试覆盖(`occupancy/grid_test.go`)

| 测试 | 覆盖需求 |
|---|---|
| `TestSaturationBoundsAndRecovery` / `TestProbabilityStability` | 饱和边界下 10 万次更新不溢出,单次相反观测仍生效,恢复步数有界 |
| `TestRayFreeSpaceAndEndpoint` / `TestScanEndpointNotOverwrittenByOtherRay` | 测距线经过栅格判空闲、终点判占据、终点不被自由空间更新覆盖 |
| `TestOverconfidenceDetection` / `TestOverconfidenceDetectionInSimulation` | 高置信度参数下少量噪声测量即被标记为异常收敛,合理参数不误报 |
| `TestConvergenceToStableValue` | 一致观测下概率单调收敛到饱和极限并保持稳定,混合证据下累加器永不出界 |

## 项目结构

```
occupancy/grid.go      栅格、log-odds 饱和更新、Bresenham 测距线、逆传感器模型
occupancy/analyze.go   过度自信/异常收敛检测
occupancy/sim.go       仿真世界、带噪声的 360° 扫描生成、逐帧记录
occupancy/html.go      单文件静态 HTML 回放页生成(数据全内嵌)
occupancy/grid_test.go 自动化测试
cmd/ogmap/main.go      主程序:跑两种模型参数的仿真、输出检测报告、生成 HTML
```
