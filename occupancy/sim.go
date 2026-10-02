package occupancy

import (
	"math"
	"math/rand"
)

// World is the ground-truth environment used to simulate sensor readings.
type World struct {
	W, H int
	Occ  []bool
}

// RoomWorld builds a rectangular room with boundary walls and a few
// interior obstacles, w×h cells.
func RoomWorld(w, h int) *World {
	wld := &World{W: w, H: h, Occ: make([]bool, w*h)}
	set := func(x, y int) {
		if x >= 0 && x < w && y >= 0 && y < h {
			wld.Occ[y*w+x] = true
		}
	}
	for x := 0; x < w; x++ {
		set(x, 0)
		set(x, h-1)
	}
	for y := 0; y < h; y++ {
		set(0, y)
		set(w-1, y)
	}
	// Interior obstacles: a block, a pillar and a partial wall.
	for x := w / 4; x < w/4+4; x++ {
		for y := h / 3; y < h/3+4; y++ {
			set(x, y)
		}
	}
	set(3*w/5, h/2)
	set(3*w/5+1, h/2)
	for y := 2 * h / 3; y < 2*h/3+8 && y < h-1; y++ {
		set(4*w/5, y)
	}
	return wld
}

// Occupied reports whether a cell is occupied; out-of-bounds counts as
// occupied so rays stop at the map edge.
func (w *World) Occupied(x, y int) bool {
	if x < 0 || x >= w.W || y < 0 || y >= w.H {
		return true
	}
	return w.Occ[y*w.W+x]
}

// RayCast marches along a ray from (ox, oy) meters at the given angle and
// returns the distance to the first occupied cell. hit is false when the
// ray reaches maxRange without hitting anything.
func (w *World) RayCast(ox, oy, angle, maxRange, cellSize float64) (dist float64, hit bool) {
	step := cellSize * 0.5
	ca, sa := math.Cos(angle), math.Sin(angle)
	for d := step; d < maxRange; d += step {
		cx := int((ox + d*ca) / cellSize)
		cy := int((oy + d*sa) / cellSize)
		if w.Occupied(cx, cy) {
			return d, true
		}
	}
	return maxRange, false
}

// Pose is a robot position in meters (the beams of a scan rotate around
// it, so no heading is needed).
type Pose struct{ X, Y float64 }

// SimConfig parameterizes one simulated mapping run.
type SimConfig struct {
	Resolution   float64 // meters per cell
	LMin, LMax   float64
	Model        SensorModel
	MaxRange     float64 // meters
	Beams        int     // beams per 360° scan
	NoiseSigma   float64 // Gaussian range noise, meters
	SpuriousProb float64 // probability a beam returns a phantom short reading
	Seed         int64
}

// Frame is one replayable snapshot: per-cell occupancy probability
// quantized to 0..255, plus the map-wide confidence statistics.
type Frame struct {
	Probs []uint8
	Stats FrameStats
}

// RunScenario simulates the robot visiting each pose, taking one noisy
// 360° scan per pose, integrating it into a fresh grid, and recording a
// frame per scan. It returns the frames and the final grid.
func RunScenario(world *World, poses []Pose, cfg SimConfig) ([]Frame, *Grid) {
	g := NewGrid(world.W, world.H, cfg.Resolution, cfg.LMin, cfg.LMax)
	rng := rand.New(rand.NewSource(cfg.Seed))
	frames := make([]Frame, 0, len(poses))

	for _, pose := range poses {
		ox := int(pose.X / cfg.Resolution)
		oy := int(pose.Y / cfg.Resolution)
		rays := make([]Ray, 0, cfg.Beams)
		for b := 0; b < cfg.Beams; b++ {
			angle := 2 * math.Pi * float64(b) / float64(cfg.Beams)
			dist, hit := world.RayCast(pose.X, pose.Y, angle, cfg.MaxRange, cfg.Resolution)
			if hit {
				// Gaussian range noise on genuine hits.
				dist += gauss(rng) * cfg.NoiseSigma
				if dist < cfg.Resolution {
					dist = cfg.Resolution
				}
				if dist >= cfg.MaxRange {
					dist, hit = cfg.MaxRange, false
				}
			}
			// Spurious phantom obstacle in free space.
			if rng.Float64() < cfg.SpuriousProb {
				dist = cfg.Resolution * (1 + rng.Float64()*(cfg.MaxRange/cfg.Resolution-1))
				hit = true
			}
			ex := int((pose.X + dist*math.Cos(angle)) / cfg.Resolution)
			ey := int((pose.Y + dist*math.Sin(angle)) / cfg.Resolution)
			rays = append(rays, Ray{EndX: ex, EndY: ey, Hit: hit})
		}
		g.UpdateScan(ox, oy, rays, cfg.Model)
		frames = append(frames, Frame{
			Probs: snapshotProbs(g),
			Stats: g.Analyze(DefaultMinObs, DefaultExtreme),
		})
	}
	return frames, g
}

// snapshotProbs quantizes every cell's probability to 0..255.
func snapshotProbs(g *Grid) []uint8 {
	out := make([]uint8, len(g.logOdds))
	for i, l := range g.logOdds {
		out[i] = uint8(math.Round(ProbabilityOf(l) * 255))
	}
	return out
}

// gauss draws a standard normal sample via Box-Muller.
func gauss(rng *rand.Rand) float64 {
	u1 := rng.Float64()
	for u1 == 0 {
		u1 = rng.Float64()
	}
	u2 := rng.Float64()
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}
