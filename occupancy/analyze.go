package occupancy

import "math"

// FrameStats summarizes the confidence state of the whole map after one
// scan, used to detect overconfident (abnormally fast) convergence.
type FrameStats struct {
	Observed  int // cells that received at least one update
	Extreme   int // observed cells with p >= Extreme or p <= 1-Extreme
	Premature int // extreme cells whose observation count is below MinObs
	Saturated int // cells pinned at a saturation bound
}

// Analysis thresholds.
const (
	// DefaultExtreme is the probability beyond which a cell is considered
	// to hold an "extreme" (near-certain) belief.
	DefaultExtreme = 0.95
	// DefaultMinObs is the minimum number of observations a cell should
	// need before an extreme belief is considered justified. Reaching an
	// extreme with fewer observations indicates an overconfident inverse
	// sensor model.
	DefaultMinObs = 6
)

// Analyze computes the FrameStats of the grid's current state.
//
// A cell is flagged "premature" when its belief is already extreme even
// though it has been observed fewer than minObs times. With a sane sensor
// model (|LOcc|, |LFree| well below the saturation bounds) several
// consistent observations are required to reach an extreme, so premature
// cells stay near zero. With an overconfident model a single noisy
// measurement can saturate a cell, and Premature grows quickly.
func (g *Grid) Analyze(minObs int, extreme float64) FrameStats {
	var s FrameStats
	lExt := logOddsOf(extreme)
	for i, l := range g.logOdds {
		if g.obs[i] == 0 {
			continue
		}
		s.Observed++
		if l >= g.LMax || l <= g.LMin {
			s.Saturated++
		}
		if l >= lExt || l <= -lExt {
			s.Extreme++
			if int(g.obs[i]) < minObs {
				s.Premature++
			}
		}
	}
	return s
}

// PrematureCells lists the coordinates of cells whose belief is extreme
// despite too few observations — the concrete evidence of overconfidence.
func (g *Grid) PrematureCells(minObs int, extreme float64) [][2]int {
	lExt := logOddsOf(extreme)
	var out [][2]int
	for y := 0; y < g.Height; y++ {
		for x := 0; x < g.Width; x++ {
			i := g.idx(x, y)
			if g.obs[i] == 0 || int(g.obs[i]) >= minObs {
				continue
			}
			if l := g.logOdds[i]; l >= lExt || l <= -lExt {
				out = append(out, [2]int{x, y})
			}
		}
	}
	return out
}

func logOddsOf(p float64) float64 {
	// p is a caller-supplied threshold in (0.5, 1); no clamping needed
	// beyond guarding the degenerate inputs.
	if p <= 0.5 {
		return 0
	}
	if p >= 1 {
		p = 1 - 1e-12
	}
	return math.Log(p / (1 - p))
}
