// Package occupancy implements a log-odds occupancy grid map with
// numerically saturated updates, correct free-space/endpoint handling
// along sensor rays, and diagnostics for overconfident sensor models.
package occupancy

import "math"

// Default saturation bounds for the log-odds accumulator.
//
// Clamping to [LMin, LMax] serves two purposes:
//  1. Repeated identical observations can never push the accumulator to
//     +/-Inf (or to a value so large that exp() overflows in Probability).
//  2. A saturated cell can still be "talked back" by a bounded number of
//     contradictory observations: at most ceil((LMax-LMin)/|l_free|) free
//     observations always suffice to flip a fully occupied-saturated cell.
const (
	DefaultLMin = -6.0
	DefaultLMax = 6.0
)

// SensorModel is the inverse sensor model p(m | z) expressed as the two
// log-odds increments a single beam applies.
//
// POcc  = probability that the endpoint cell is occupied, given a hit.
// PFree = probability that a traversed cell is occupied, given the beam
//
//	passed through it (must be < 0.5 for a sane model).
type SensorModel struct {
	POcc  float64
	PFree float64
	LOcc  float64 // log-odds increment applied to the endpoint cell
	LFree float64 // log-odds increment applied to traversed cells
}

// NewSensorModel builds a model from probabilities, precomputing the
// log-odds increments. Probabilities are clamped away from exactly 0 and 1
// so the increments stay finite.
func NewSensorModel(pOcc, pFree float64) SensorModel {
	const eps = 1e-9
	clamp := func(p float64) float64 {
		if p < eps {
			return eps
		}
		if p > 1-eps {
			return 1 - eps
		}
		return p
	}
	pOcc, pFree = clamp(pOcc), clamp(pFree)
	return SensorModel{
		POcc:  pOcc,
		PFree: pFree,
		LOcc:  math.Log(pOcc / (1 - pOcc)),
		LFree: math.Log(pFree / (1 - pFree)),
	}
}

// Grid is a 2-D occupancy grid storing the log-odds of occupancy per cell.
type Grid struct {
	Width      int
	Height     int
	Resolution float64 // meters per cell
	LMin, LMax float64 // saturation bounds for the accumulator

	logOdds []float64
	obs     []uint32 // number of updates applied to each cell
}

// NewGrid creates a grid with all cells at log-odds 0 (p = 0.5, unknown).
func NewGrid(w, h int, resolution, lMin, lMax float64) *Grid {
	if lMin >= lMax {
		panic("occupancy: LMin must be < LMax")
	}
	return &Grid{
		Width:      w,
		Height:     h,
		Resolution: resolution,
		LMin:       lMin,
		LMax:       lMax,
		logOdds:    make([]float64, w*h),
		obs:        make([]uint32, w*h),
	}
}

// InBounds reports whether (x, y) is a valid cell coordinate.
func (g *Grid) InBounds(x, y int) bool {
	return x >= 0 && x < g.Width && y >= 0 && y < g.Height
}

func (g *Grid) idx(x, y int) int { return y*g.Width + x }

// LogOdds returns the current accumulator value of a cell.
func (g *Grid) LogOdds(x, y int) float64 { return g.logOdds[g.idx(x, y)] }

// Observations returns how many updates a cell has received.
func (g *Grid) Observations(x, y int) int { return int(g.obs[g.idx(x, y)]) }

// ProbabilityOf converts a log-odds value to a probability without
// overflowing for very large magnitudes.
func ProbabilityOf(l float64) float64 {
	if l >= 0 {
		return 1 / (1 + math.Exp(-l))
	}
	e := math.Exp(l)
	return e / (1 + e)
}

// Probability returns the occupancy probability of a cell.
func (g *Grid) Probability(x, y int) float64 { return ProbabilityOf(g.LogOdds(x, y)) }

// addDelta applies one log-odds increment to a cell, saturating at the
// configured bounds. Out-of-bounds cells are silently ignored so rays
// clipped by the map edge do not corrupt the update.
func (g *Grid) addDelta(x, y int, delta float64) {
	if !g.InBounds(x, y) {
		return
	}
	i := g.idx(x, y)
	l := g.logOdds[i] + delta
	if l > g.LMax {
		l = g.LMax
	} else if l < g.LMin {
		l = g.LMin
	}
	g.logOdds[i] = l
	g.obs[i]++
}

// Bresenham returns every grid cell on the line from (x0,y0) to (x1,y1),
// inclusive of both endpoints, in traversal order.
func Bresenham(x0, y0, x1, y1 int) [][2]int {
	abs := func(v int) int {
		if v < 0 {
			return -v
		}
		return v
	}
	dx, dy := abs(x1-x0), -abs(y1-y0)
	sx, sy := -1, -1
	if x0 < x1 {
		sx = 1
	}
	if y0 < y1 {
		sy = 1
	}
	err := dx + dy
	x, y := x0, y0
	pts := make([][2]int, 0, abs(x1-x0)+abs(y1-y0)+1)
	for {
		pts = append(pts, [2]int{x, y})
		if x == x1 && y == y1 {
			return pts
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x += sx
		}
		if e2 <= dx {
			err += dx
			y += sy
		}
	}
}

// Ray is one sensor beam: the grid cell the beam ends in, and whether the
// beam actually hit something (Hit=false means max-range / no return, so
// even the final cell is free space).
type Ray struct {
	EndX, EndY int
	Hit        bool
}

// UpdateRay integrates a single beam: every traversed cell except the
// endpoint is updated as free space; the endpoint is updated as occupied
// when the beam hit something. The endpoint update is applied strictly
// after all free-space updates of this ray, so the endpoint cell can never
// be overwritten by its own ray's free-space evidence.
func (g *Grid) UpdateRay(x0, y0 int, r Ray, m SensorModel) {
	pts := Bresenham(x0, y0, r.EndX, r.EndY)
	free := pts
	if r.Hit {
		free = pts[:len(pts)-1]
	}
	for _, p := range free {
		g.addDelta(p[0], p[1], m.LFree)
	}
	if r.Hit {
		g.addDelta(r.EndX, r.EndY, m.LOcc)
	}
}

// UpdateScan integrates a whole scan (many beams from one pose). All
// free-space updates of all beams are applied first, and all endpoint
// occupied updates afterwards. This two-pass ordering guarantees that a
// cell which is the endpoint of one beam is never re-marked free because
// another beam of the same scan happened to pass through it.
func (g *Grid) UpdateScan(x0, y0 int, rays []Ray, m SensorModel) {
	for _, r := range rays {
		pts := Bresenham(x0, y0, r.EndX, r.EndY)
		free := pts
		if r.Hit {
			free = pts[:len(pts)-1]
		}
		for _, p := range free {
			g.addDelta(p[0], p[1], m.LFree)
		}
	}
	for _, r := range rays {
		if r.Hit {
			g.addDelta(r.EndX, r.EndY, m.LOcc)
		}
	}
}
