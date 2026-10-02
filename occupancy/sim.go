package occupancy

import (
	"math"
	"math/rand"
)

// World is the ground-truth map used by the simulator.
type World struct {
	W, H int
	Occ  []bool
}

// NewWorld builds a test world: border walls, a rectangular room and a
// pillar, so beams produce a mix of hits and max-range misses.
func NewWorld(w, h int) *World {
	wld := &World{W: w, H: h, Occ: make([]bool, w*h)}
	set := func(x, y int) {
		if x >= 0 && y >= 0 && x < w && y < h {
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
	// Rectangular block.
	for y := h/2 - 6; y <= h/2+2; y++ {
		for x := w/2 + 6; x <= w/2+12; x++ {
			set(x, y)
		}
	}
	// Pillar.
	for y := h/2 + 8; y <= h/2+10; y++ {
		for x := w/2 - 10; x <= w/2-8; x++ {
			set(x, y)
		}
	}
	return wld
}

func (w *World) occupied(x, y int) bool {
	if x < 0 || y < 0 || x >= w.W || y >= w.H {
		return true
	}
	return w.Occ[y*w.W+x]
}

// castRay marches from (x,y) along theta and returns the distance to the
// first occupied cell, or maxRange if nothing is hit.
func (w *World) castRay(x, y, theta, maxRange float64) (dist float64, hit bool) {
	const step = 0.05
	dx, dy := math.Cos(theta), math.Sin(theta)
	for d := step; d < maxRange; d += step {
		if w.occupied(int(x+dx*d), int(y+dy*d)) {
			return d, true
		}
	}
	return maxRange, false
}

// Frame holds one replay frame: quantized probabilities (p*255 rounded)
// for the sane and the overconfident map, row-major.
type Frame struct {
	Sane []byte
	Over []byte
}

// SimResult is everything the HTML replay needs.
type SimResult struct {
	Width, Height int
	World         []byte // 1 = truly occupied
	Frames        []Frame
	AnomSane      []Anomaly
	AnomOver      []Anomaly
}

// Simulate runs a robot with a noisy range finder around a circular path
// inside the world, feeding identical measurements into two maps: one with
// the sane sensor model and one with the overconfident model. A frame is
// recorded after every scan, and both maps are checked for overconfidence
// anomalies.
func Simulate(seed int64, scans int) *SimResult {
	const (
		size     = 64
		beams    = 24
		maxRange = 20.0
		sigma    = 0.25 // range noise, in cells
	)
	rng := rand.New(rand.NewSource(seed))
	world := NewWorld(size, size)

	sane := NewGrid(size, size, DefaultLMin, DefaultLMax)
	over := NewGrid(size, size, DefaultLMin, DefaultLMax)
	detSane := NewDetector()
	detOver := NewDetector()

	res := &SimResult{Width: size, Height: size}
	res.World = make([]byte, size*size)
	for i, o := range world.Occ {
		if o {
			res.World[i] = 1
		}
	}

	cx, cy, radius := size/2.0, size/2.0, 13.0
	for s := 0; s < scans; s++ {
		a := 2 * math.Pi * float64(s) / float64(scans)
		rx := cx + radius*math.Cos(a)
		ry := cy + radius*math.Sin(a)

		rays := make([]Ray, 0, beams)
		for b := 0; b < beams; b++ {
			theta := 2 * math.Pi * float64(b) / float64(beams)
			dist, hit := world.castRay(rx, ry, theta, maxRange)
			meas := maxRange
			if hit {
				meas = dist + rng.NormFloat64()*sigma
				if meas < 0.5 {
					meas = 0.5
				}
				if meas > maxRange {
					meas = maxRange
				}
			}
			rays = append(rays, Ray{
				X0: rx, Y0: ry,
				X1: rx + math.Cos(theta)*meas,
				Y1: ry + math.Sin(theta)*meas,
			})
		}

		sane.IntegrateScan(rays, DefaultSensorModel())
		over.IntegrateScan(rays, OverconfidentSensorModel())
		detSane.CheckScan(sane, s)
		detOver.CheckScan(over, s)

		res.Frames = append(res.Frames, Frame{
			Sane: probBytes(sane),
			Over: probBytes(over),
		})
	}
	res.AnomSane = detSane.Anomalies
	res.AnomOver = detOver.Anomalies
	return res
}

func probBytes(g *Grid) []byte {
	out := make([]byte, g.Width*g.Height)
	for i, l := range g.l {
		out[i] = byte(math.Round(Probability(l) * 255))
	}
	return out
}
