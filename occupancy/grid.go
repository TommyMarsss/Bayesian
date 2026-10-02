// Package occupancy implements a log-odds occupancy grid map with
// numerical saturation, a beam-based inverse sensor model, and
// overconfidence anomaly detection.
package occupancy

import "math"

// Default saturation bounds for the log-odds accumulator.
//
// With LMax = 12 the most extreme representable probability is
// 1/(1+e^-12) ≈ 0.9999938, so a cell can never reach exactly 0 or 1 and a
// single contradictory observation always shifts the belief by a full
// sensor-model step instead of being swallowed by an unbounded accumulator.
const (
	DefaultLMin = -12.0
	DefaultLMax = 12.0
)

// LogOdds converts a probability in (0,1) to log-odds form.
func LogOdds(p float64) float64 {
	return math.Log(p / (1 - p))
}

// Probability converts log-odds back to a probability using a
// numerically stable logistic (no overflow for large |l|).
func Probability(l float64) float64 {
	if l >= 0 {
		return 1 / (1 + math.Exp(-l))
	}
	e := math.Exp(l)
	return e / (1 + e)
}

// SensorModel is the inverse sensor model for a beam-based range finder.
//
// POcc is the probability that the endpoint cell of a beam is occupied
// given the beam terminated there; PFree is the probability that a cell
// the beam passed through is occupied (should be < 0.5).
type SensorModel struct {
	POcc  float64
	PFree float64
}

// DefaultSensorModel returns a conservative, well-behaved model.
func DefaultSensorModel() SensorModel {
	return SensorModel{POcc: 0.7, PFree: 0.4}
}

// OverconfidentSensorModel returns a deliberately extreme model used to
// demonstrate (and test detection of) overconfident convergence.
func OverconfidentSensorModel() SensorModel {
	return SensorModel{POcc: 0.97, PFree: 0.03}
}

// LOcc is the log-odds increment applied to a beam endpoint cell.
func (m SensorModel) LOcc() float64 { return LogOdds(m.POcc) }

// LFree is the (negative) log-odds increment applied to cells a beam
// passes through.
func (m SensorModel) LFree() float64 { return LogOdds(m.PFree) }

// Ray is one range measurement in grid (cell) coordinates. The endpoint
// (X1,Y1) is where the beam stopped.
type Ray struct {
	X0, Y0, X1, Y1 float64
}

// Grid is a log-odds occupancy grid with saturating updates.
type Grid struct {
	Width, Height int
	LMin, LMax    float64

	l   []float64 // log-odds accumulator per cell
	obs []int     // number of updates applied per cell
}

// NewGrid creates a grid whose cells start at the prior p=0.5 (l=0).
func NewGrid(w, h int, lMin, lMax float64) *Grid {
	if lMin >= lMax {
		panic("occupancy: LMin must be < LMax")
	}
	return &Grid{
		Width: w, Height: h,
		LMin: lMin, LMax: lMax,
		l:   make([]float64, w*h),
		obs: make([]int, w*h),
	}
}

// Index maps a cell coordinate to its slice index.
func (g *Grid) Index(x, y int) (int, bool) {
	if x < 0 || y < 0 || x >= g.Width || y >= g.Height {
		return 0, false
	}
	return y*g.Width + x, true
}

// LogOddsAt returns the accumulated log-odds of a cell.
func (g *Grid) LogOddsAt(x, y int) (float64, bool) {
	i, ok := g.Index(x, y)
	if !ok {
		return 0, false
	}
	return g.l[i], true
}

// ProbAt returns the occupancy probability of a cell.
func (g *Grid) ProbAt(x, y int) (float64, bool) {
	l, ok := g.LogOddsAt(x, y)
	if !ok {
		return 0, false
	}
	return Probability(l), true
}

// Observations returns how many updates a cell has received.
func (g *Grid) Observations(x, y int) int {
	i, ok := g.Index(x, y)
	if !ok {
		return 0
	}
	return g.obs[i]
}

// update applies one saturating log-odds increment.
func (g *Grid) update(idx int, delta float64) {
	v := g.l[idx] + delta
	if v > g.LMax {
		v = g.LMax
	} else if v < g.LMin {
		v = g.LMin
	}
	g.l[idx] = v
	g.obs[idx]++
}

// IntegrateScan fuses one scan (a set of rays) into the map.
//
// Ordering matters and is done deliberately:
//  1. Every cell a beam passes through (excluding its endpoint) is marked
//     free; the endpoint cell is marked occupied.
//  2. A cell that is the endpoint of one beam but lies on the free path
//     of another is removed from the free set — the hit evidence wins.
//  3. All free updates are applied first, all occupied updates last, so
//     an endpoint can never be overwritten by a free-space update from
//     its own or another beam in the same scan.
func (g *Grid) IntegrateScan(rays []Ray, m SensorModel) {
	free := make(map[int]struct{})
	occ := make(map[int]struct{})

	for _, r := range rays {
		cells := LineCells(int(r.X0), int(r.Y0), int(r.X1), int(r.Y1))
		for _, c := range cells[:len(cells)-1] {
			if idx, ok := g.Index(c[0], c[1]); ok {
				free[idx] = struct{}{}
			}
		}
		last := cells[len(cells)-1]
		if idx, ok := g.Index(last[0], last[1]); ok {
			occ[idx] = struct{}{}
		}
	}
	for idx := range occ {
		delete(free, idx)
	}
	for idx := range free {
		g.update(idx, m.LFree())
	}
	for idx := range occ {
		g.update(idx, m.LOcc())
	}
}

// LineCells returns the integer cells on the Bresenham line from
// (x0,y0) to (x1,y1), inclusive of both endpoints, in order.
func LineCells(x0, y0, x1, y1 int) [][2]int {
	cells := make([][2]int, 0, 8)
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx + dy
	for {
		cells = append(cells, [2]int{x0, y0})
		if x0 == x1 && y0 == y1 {
			return cells
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
