package occupancy

// Anomaly records a cell whose belief became extreme suspiciously fast —
// the signature of an overconfident inverse sensor model.
type Anomaly struct {
	Scan int     `json:"scan"`
	X    int     `json:"x"`
	Y    int     `json:"y"`
	Obs  int     `json:"obs"`
	P    float64 `json:"p"`
}

// Detector flags cells that reach near-0 or near-1 probability after only
// a handful of observations. A well-calibrated model needs many consistent
// measurements to become that certain; if a map gets there in <= MaxObs
// updates, the sensor model is almost certainly too confident.
type Detector struct {
	// Lo/Hi are the "extreme probability" thresholds.
	Lo, Hi float64
	// MaxObs is the largest observation count at which an extreme belief
	// is still considered anomalous.
	MaxObs int

	Anomalies []Anomaly
	reported  map[int]bool
}

// NewDetector returns a Detector with defaults tuned so the sane
// DefaultSensorModel never triggers it (it needs >=5 consistent hits to
// reach p=0.98) while the OverconfidentSensorModel triggers it within 2.
func NewDetector() *Detector {
	return &Detector{
		Lo: 0.02, Hi: 0.98, MaxObs: 3,
		reported: make(map[int]bool),
	}
}

// CheckScan inspects every cell of the grid after a scan and records new
// anomalies. Each cell is reported at most once.
func (d *Detector) CheckScan(g *Grid, scan int) {
	for y := 0; y < g.Height; y++ {
		for x := 0; x < g.Width; x++ {
			idx, _ := g.Index(x, y)
			n := g.obs[idx]
			if n == 0 || n > d.MaxObs || d.reported[idx] {
				continue
			}
			p := Probability(g.l[idx])
			if p <= d.Lo || p >= d.Hi {
				d.reported[idx] = true
				d.Anomalies = append(d.Anomalies, Anomaly{
					Scan: scan, X: x, Y: y, Obs: n, P: p,
				})
			}
		}
	}
}

// SaturatedFraction returns the fraction of observed cells whose
// probability is at or beyond the extreme thresholds — a whole-map
// overconfidence metric for a single point in time.
func (d *Detector) SaturatedFraction(g *Grid) (observed, saturated int, frac float64) {
	for idx := range g.l {
		if g.obs[idx] == 0 {
			continue
		}
		observed++
		p := Probability(g.l[idx])
		if p <= d.Lo || p >= d.Hi {
			saturated++
		}
	}
	if observed > 0 {
		frac = float64(saturated) / float64(observed)
	}
	return observed, saturated, frac
}
