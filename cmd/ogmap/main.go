// Command ogmap simulates a robot mapping a room with a noisy range
// sensor, integrates the scans into a log-odds occupancy grid under two
// inverse sensor models (balanced and deliberately overconfident), prints
// an overconfidence diagnostic, and writes a single self-contained HTML
// file for frame-by-frame replay.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/TommyMarsss/Bayesian/occupancy"
)

func main() {
	out := flag.String("o", "occupancy_map.html", "output HTML file")
	flag.Parse()

	const (
		gridW, gridH = 64, 48
		resolution   = 0.25 // meters per cell
	)
	world := occupancy.RoomWorld(gridW, gridH)

	// Robot path: a horizontal transect through the room, one scan per pose.
	var poses []occupancy.Pose
	for x := 2.0; x <= 13.5; x += 0.3 {
		poses = append(poses, occupancy.Pose{X: x, Y: 6.0})
	}

	base := occupancy.SimConfig{
		Resolution:   resolution,
		LMin:         occupancy.DefaultLMin,
		LMax:         occupancy.DefaultLMax,
		MaxRange:     6.0,
		Beams:        36,
		NoiseSigma:   0.15,
		SpuriousProb: 0.03,
		Seed:         42,
	}

	scenarios := []struct {
		name        string
		pOcc, pFree float64
	}{
		{"合理模型", 0.70, 0.40},
		{"过度自信模型", 0.99, 0.01},
	}

	var page occupancy.PageData
	page.Width, page.Height = gridW, gridH
	page.Truth = world.Occ

	for _, sc := range scenarios {
		cfg := base
		cfg.Model = occupancy.NewSensorModel(sc.pOcc, sc.pFree)
		frames, grid := occupancy.RunScenario(world, poses, cfg)
		page.Scenarios = append(page.Scenarios, occupancy.ScenarioData{
			Name: sc.name, POcc: sc.pOcc, PFree: sc.pFree, Frames: frames,
		})

		final := frames[len(frames)-1].Stats
		premature := grid.PrematureCells(occupancy.DefaultMinObs, occupancy.DefaultExtreme)
		ratio := 0.0
		if final.Observed > 0 {
			ratio = float64(final.Premature) / float64(final.Observed)
		}
		fmt.Printf("场景 %-10s (p_occ=%.2f, p_free=%.2f)\n", sc.name, sc.pOcc, sc.pFree)
		fmt.Printf("  已观测 %d 栅格, 极端概率 %d, 饱和 %d, 观测不足 %d 次即极端 %d (%.1f%%)\n",
			final.Observed, final.Extreme, final.Saturated, occupancy.DefaultMinObs,
			final.Premature, ratio*100)
		// 少量过早极端出现在地图边缘属正常(掠射角观测少但一致);
		// 占比超过 5% 则判定为传感器模型过度自信导致的异常收敛。
		if ratio > 0.05 {
			fmt.Printf("  ⚠ 检测到过度自信异常收敛: %.1f%% 的已观测栅格在噪声测量下过早达到极端概率, 示例: %v\n",
				ratio*100, previewCells(premature, 5))
		} else {
			fmt.Println("  ✓ 未检测到异常收敛")
		}
	}

	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintln(os.Stderr, "create output:", err)
		os.Exit(1)
	}
	defer f.Close()
	if err := occupancy.WriteHTML(f, page); err != nil {
		fmt.Fprintln(os.Stderr, "write html:", err)
		os.Exit(1)
	}
	fmt.Printf("\n已生成 %s — 用浏览器打开即可逐帧回放。\n", *out)
}

func previewCells(cells [][2]int, n int) [][2]int {
	if len(cells) > n {
		return cells[:n]
	}
	return cells
}
